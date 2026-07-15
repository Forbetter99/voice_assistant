"""
Toast 通知 — Windows 通知中心消息
"""


class ToastNotifier:
    """Windows 通知中心消息"""

    def __init__(self):
        self._enabled = True

    def notify(self, title: str, message: str, duration: int = 5):
        """显示系统通知"""
        if not self._enabled:
            return
        try:
            from plyer import notification
            notification.notify(title=title, message=message, timeout=duration)
        except Exception:
            try:
                from win10toast import ToastNotifier as Win10Toast
                toaster = Win10Toast()
                toaster.show_toast(title, message, duration=duration, threaded=True)
            except Exception:
                pass

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False
