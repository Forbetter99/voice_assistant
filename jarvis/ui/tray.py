"""
系统托盘图标 — 最小化到托盘
"""
import threading


class SystemTray:
    """Windows 系统托盘图标"""

    def __init__(self, app, icon_path: str = None):
        self.app = app
        self.icon_path = icon_path
        self._icon = None
        self._thread = None
        self._running = False

    def _create_menu(self):
        import pystray
        from jarvis.windows.integration import is_autostart_enabled, set_autostart

        app = self.app

        def toggle_window(icon, item):
            if app.window and app.window.root.winfo_viewable():
                app.window.root.withdraw()
            else:
                app.window.show()

        def toggle_wake(icon, item):
            app.toggle_wake_word()

        def toggle_autostart(icon, item):
            exe_path = getattr(app, '_exe_path', None)
            set_autostart(not is_autostart_enabled(), exe_path)

        def quit_app(icon, item):
            icon.stop()
            app.shutdown()

        return pystray.Menu(
            pystray.MenuItem("显示/隐藏", toggle_window, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "语音唤醒",
                toggle_wake,
                checked=lambda item: app.wake_word_enabled,
            ),
            pystray.MenuItem(
                "开机自启",
                toggle_autostart,
                checked=lambda item: is_autostart_enabled(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", quit_app),
        )

    def _create_icon(self):
        """创建托盘图标（从 PNG 文件，或生成默认图标）"""
        from PIL import Image, ImageDraw, ImageFont

        if self.icon_path and os.path.exists(self.icon_path):
            return Image.open(self.icon_path)

        # 生成默认图标：圆形 + "J" 字母
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 画蓝色圆形背景
        draw.ellipse([2, 2, size - 2, size - 2], fill="#89b4fa")

        # 画白色 "J" 字母
        try:
            font = ImageFont.truetype("segoeui.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), "J", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) / 2 - bbox[0]
        y = (size - th) / 2 - bbox[1]
        draw.text((x, y), "J", fill="white", font=font)

        return img

    def start(self):
        """在后台线程中启动托盘"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import pystray
        import os  # for os.path.exists in _create_icon

        icon_image = self._create_icon()
        menu = self._create_menu()
        self._icon = pystray.Icon(
            "jarvis",
            icon_image,
            "贾维斯语音助手",
            menu=menu,
        )
        self._icon.run()

    def stop(self):
        """停止托盘图标"""
        self._running = False
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass

    def update_menu(self):
        """更新托盘菜单（当状态变化时调用）"""
        if self._icon:
            try:
                self._icon.update_menu()
            except Exception:
                pass

    def notify(self, title: str, message: str, duration: int = 5):
        """显示托盘通知"""
        try:
            from plyer import notification
            notification.notify(title=title, message=message, timeout=duration)
        except Exception:
            try:
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(title, message, duration=duration, threaded=True)
            except Exception:
                pass
