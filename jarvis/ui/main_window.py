"""
主窗口 — Tkinter GUI（提取自 gui.py）
仅包含 UI 构造和更新方法，业务逻辑在 jarvis.app
"""
import tkinter as tk
from tkinter import ttk
import time
import threading

from jarvis.config import (
    COLORS, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
)
from jarvis.ui.audio_level import AudioLevelMeter


class ProgressBar(tk.Canvas):
    """自定义进度条"""

    def __init__(self, parent, width=300, height=16, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=COLORS["surface"], highlightthickness=0, **kwargs)
        self._width = width
        self._height = height
        self._bar = self.create_rectangle(0, 0, 0, height,
                                          fill=COLORS["blue"], outline="")

    def set(self, pct: int):
        w = int(self._width * pct / 100)
        self.coords(self._bar, 0, 0, w, self._height)


class MainWindow:
    """主窗口 UI（Tkinter）"""

    def __init__(self, app_controller=None):
        self.app = app_controller
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)

        # ── 内部状态 ──
        self._on_close_callback = None
        self._on_minimize_callback = None

        # ── 组件引用（初始化时为 None，_build_ui 中创建）──
        self.text = None
        self.log_text = None
        self.record_btn = None
        self.test_btn = None
        self.wake_btn = None
        self.register_btn = None
        self.owner_check = None
        self.owner_var = None
        self.status_bar = None
        self.progress_bar = None
        self.audio_meter = None
        self._notebook = None

        self._build_ui()

        # 样式标签
        if self.text:
            self.text.tag_configure("jarvis", foreground=COLORS["blue"])
            self.text.tag_configure("gray", foreground=COLORS["text_dark"])
            self.text.tag_configure("error", foreground=COLORS["red"])
            self.text.tag_configure("system", foreground=COLORS["yellow"])

        # 关闭按钮 → 最小化到托盘
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)

    def set_app(self, app_controller):
        """设置应用控制器引用"""
        self.app = app_controller

    def set_on_close(self, callback):
        """设置窗口关闭回调"""
        self._on_close_callback = callback

    def set_on_minimize(self, callback):
        """设置最小化回调"""
        self._on_minimize_callback = callback

    def _on_close_window(self):
        """关闭按钮：最小化到托盘"""
        if self._on_close_callback:
            self._on_close_callback()
        else:
            self.root.withdraw()

    def _build_ui(self):
        """构建所有 UI 组件"""
        # ── 标题 ──
        title = tk.Label(
            self.root, text="🤖 贾维斯", font=("微软雅黑", 20, "bold"),
            bg=COLORS["bg"], fg=COLORS["text"]
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            self.root, text="全本地语音助手", font=("微软雅黑", 10),
            bg=COLORS["bg"], fg=COLORS["text_muted"]
        )
        subtitle.pack(pady=(0, 10))

        # ── 选项卡 ──
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        style = ttk.Style()
        style.configure("TNotebook", background=COLORS["bg"])
        style.configure("TNotebook.Tab", padding=[12, 4])

        # 对话页
        chat_frame = tk.Frame(notebook, bg=COLORS["bg"])
        notebook.add(chat_frame, text="  对话 ")

        self.text = tk.Text(
            chat_frame, font=("微软雅黑", 11), wrap="word",
            bg=COLORS["chat_bg"], fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat", borderwidth=0, padx=12, pady=12,
            state="disabled"
        )
        scrollbar = tk.Scrollbar(chat_frame, command=self.text.yview, bg=COLORS["bg"])
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 日志页
        log_frame = tk.Frame(notebook, bg=COLORS["bg"])
        notebook.add(log_frame, text="  日志 ")

        self.log_text = tk.Text(
            log_frame, font=("Consolas", 10), wrap="word",
            bg=COLORS["log_bg"], fg=COLORS["text_muted"],
            insertbackground=COLORS["text_muted"],
            relief="flat", borderwidth=0, padx=12, pady=12,
            state="disabled"
        )
        log_scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview, bg=COLORS["bg"])
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

        self._notebook = notebook

        # ── 录音按钮 ──
        btn_frame = tk.Frame(self.root, bg=COLORS["bg"])
        btn_frame.pack(pady=(0, 10))

        self.record_btn = tk.Button(
            btn_frame, text="加载中...", font=("微软雅黑", 14, "bold"),
            width=16, height=2, state="disabled",
            bg=COLORS["overlay"], fg=COLORS["text"],
            activebackground=COLORS["red"],
            relief="flat", borderwidth=0, cursor="hand2",
        )
        self.record_btn.pack()

        # ── 音频电平表 ──
        self.audio_meter = AudioLevelMeter(btn_frame, width=200, height=20)
        self.audio_meter.pack(pady=(6, 0))

        # ── 第二行按钮 ──
        test_frame = tk.Frame(self.root, bg=COLORS["bg"])
        test_frame.pack(pady=(0, 5))

        self.test_btn = tk.Button(
            test_frame, text="🎧 测试录音", font=("微软雅黑", 10),
            width=12, state="disabled",
            bg=COLORS["surface"], fg=COLORS["text_muted"],
            activebackground=COLORS["blue"],
            relief="flat", borderwidth=0, cursor="hand2",
        )
        self.test_btn.pack(side="left", padx=(0, 5))

        self.register_btn = tk.Button(
            test_frame, text="🎙️ 注册声纹", font=("微软雅黑", 10),
            width=12, state="disabled",
            bg=COLORS["surface"], fg=COLORS["text_muted"],
            activebackground=COLORS["orange"],
            relief="flat", borderwidth=0, cursor="hand2",
        )
        self.register_btn.pack(side="left", padx=(0, 5))

        # 仅机主唤醒
        self.owner_var = tk.BooleanVar(value=False)
        self.owner_check = tk.Checkbutton(
            test_frame, text="仅机主唤醒", variable=self.owner_var,
            font=("微软雅黑", 9), bg=COLORS["bg"], fg=COLORS["text_muted"],
            selectcolor=COLORS["bg"], activebackground=COLORS["bg"],
            activeforeground=COLORS["text"], state="disabled",
        )
        self.owner_check.pack(side="left")

        # ── 语音唤醒按钮 ──
        self.wake_btn = tk.Button(
            btn_frame, text="🤖 语音唤醒", font=("微软雅黑", 10),
            width=12, state="disabled",
            bg=COLORS["surface"], fg=COLORS["text_muted"],
            activebackground=COLORS["green"],
            relief="flat", borderwidth=0, cursor="hand2",
        )
        self.wake_btn.pack(pady=(5, 0))

        # ── 状态栏 ──
        self.status_bar = tk.Label(
            self.root, text="正在加载模型...", font=("微软雅黑", 9),
            bg=COLORS["bg"], fg=COLORS["text_muted"]
        )
        self.status_bar.pack(pady=(0, 15))

    # ═══════════════════════════════════════════════
    #  UI 更新方法（由 App 调用，线程安全 via root.after）
    # ═══════════════════════════════════════════════

    def log(self, msg: str):
        """在日志页追加一条带时间戳的记录"""
        ts = time.strftime("%H:%M:%S")
        if self.log_text:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"[{ts}] {msg}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def append_chat(self, text, tag=None):
        """在对话页追加文本"""
        if not self.text:
            return
        self.text.configure(state="normal")
        if tag:
            self.text.insert("end", text + "\n", tag)
        else:
            self.text.insert("end", text + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def set_status(self, text: str):
        """设置状态栏文本（需在主线程调用或用 root.after）"""
        if self.status_bar:
            self.status_bar.config(text=text)

    def set_status_safe(self, text: str):
        """线程安全地更新状态栏"""
        self.root.after(0, lambda: self.set_status(text))

    def set_record_btn_text(self, text: str, bg: str = None):
        """设置录音按钮文字和背景"""
        if not self.record_btn:
            return
        config = {"text": text}
        if bg:
            config["bg"] = bg
        self.record_btn.config(**config)

    def enable_all_buttons(self, sv_is_registered: bool = False):
        """加载完成后启用所有按钮"""
        self.record_btn.config(state="normal", text="🎤 按住说话")
        self.test_btn.config(state="normal")
        self.wake_btn.config(state="normal")
        self.register_btn.config(state="normal")
        self.owner_check.config(state="normal")
        if sv_is_registered:
            self.set_status("就绪 — 声纹已注册，可开启'仅机主唤醒'")
        else:
            self.set_status("就绪 — 建议先注册声纹")

    def show_progress(self):
        """显示进度条"""
        if self.progress_bar is None:
            self.progress_bar = ProgressBar(self.root, width=300, height=14)
            self.progress_bar.pack(before=self.status_bar, pady=(0, 5))

    def hide_progress(self, delay_ms=500):
        """延迟隐藏进度条"""
        if self.progress_bar:
            self.root.after(delay_ms, lambda: self.progress_bar.pack_forget())

    def set_progress(self, pct: int):
        """设置进度条百分比"""
        if self.progress_bar:
            self.progress_bar.set(pct)

    def show(self):
        """显示窗口（从托盘恢复）"""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide(self):
        """隐藏窗口（最小化到托盘）"""
        self.root.withdraw()

    def is_visible(self) -> bool:
        """窗口是否可见"""
        return self.root.winfo_viewable()

    def start_mainloop(self):
        """启动 Tkinter 主循环"""
        self.root.mainloop()

    def after(self, ms, func):
        """Tkinter after 的快捷方式"""
        self.root.after(ms, func)

    # ═══════════════════════════════════════════════
    # 绑定事件（由 App 在初始化后调用）
    # ═══════════════════════════════════════════════

    def bind_record(self, on_press, on_release):
        """绑定录音按钮事件"""
        self.record_btn.bind("<ButtonPress-1>", on_press)
        self.record_btn.bind("<ButtonRelease-1>", on_release)

    def bind_test_record(self, on_press, on_release):
        """绑定测试录音按钮事件"""
        self.test_btn.bind("<ButtonPress-1>", on_press)
        self.test_btn.bind("<ButtonRelease-1>", on_release)

    def bind_wake_toggle(self, callback):
        """绑定语音唤醒按钮事件"""
        self.wake_btn.config(command=callback)

    def bind_register(self, callback):
        """绑定声纹注册按钮事件"""
        self.register_btn.config(command=callback)

    def bind_owner_toggle(self, callback):
        """绑定仅机主唤醒开关事件"""
        self.owner_check.config(command=callback)

    def bind_close(self, callback):
        """绑定窗口关闭事件"""
        self._on_close_callback = callback
