"""
Windows 平台集成工具 — DWM 效果（亚克力/圆角）
"""
import ctypes
from ctypes import wintypes


# ── 常量 ──
ACCENT_DISABLED = 0
ACCENT_ENABLE_GRADIENT = 1
ACCENT_ENABLE_TRANSPARENTGRADIENT = 2
ACCENT_ENABLE_BLURBEHIND = 3
ACCENT_ENABLE_ACRYLIC = 4  # Windows 10 1803+

DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_DEFAULT = 0
DWMWCP_DONOTROUND = 1
DWMWCP_ROUND = 2
DWMWCP_ROUNDSMALL = 3

WCA_ACCENT_POLICY = 19


class ACCENTPOLICY(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_uint),
        ("AccentFlags", ctypes.c_uint),
        ("GradientColor", ctypes.c_uint),
        ("AnimationId", ctypes.c_uint),
    ]


class WINCOMPATTRDATA(ctypes.Structure):
    _fields_ = [
        ("Attribute", ctypes.c_int),
        ("Data", ctypes.POINTER(ACCENTPOLICY)),
        ("SizeOfData", ctypes.c_size_t),
    ]


class DWM:
    """Windows 桌面窗口管理器效果"""

    @staticmethod
    def get_hwnd(tk_root) -> int:
        """从 Tkinter root 获取窗口 HWND"""
        return int(tk_root.winfo_id())

    @staticmethod
    def enable_acrylic(hwnd: int, color: int = 0x99B41E1E):
        """
        启用 Windows 10/11 亚克力模糊效果
        color: BGRA 格式，例如 0x99B41E1E (alpha=0x99, blue=0xB4, green=0x1E, red=0x1E)
        """
        try:
            accent = ACCENTPOLICY()
            accent.AccentState = ACCENT_ENABLE_ACRYLIC
            accent.GradientColor = color
            accent.AccentFlags = 0x20 | 0x40  # 支持暗色模式

            data = WINCOMPATTRDATA()
            data.Attribute = WCA_ACCENT_POLICY
            data.SizeOfData = ctypes.sizeof(accent)
            data.Data = ctypes.pointer(accent)

            ctypes.windll.user32.SetWindowCompositionAttribute(hwnd, ctypes.byref(data))
        except Exception as e:
            print(f"  [DWM] 启用亚克力失败: {e}")

    @staticmethod
    def enable_blur(hwnd: int):
        """启用 Windows 7/8/10 毛玻璃效果（轻量级，兼容性更好）"""
        try:
            accent = ACCENTPOLICY()
            accent.AccentState = ACCENT_ENABLE_BLURBEHIND

            data = WINCOMPATTRDATA()
            data.Attribute = WCA_ACCENT_POLICY
            data.SizeOfData = ctypes.sizeof(accent)
            data.Data = ctypes.pointer(accent)

            ctypes.windll.user32.SetWindowCompositionAttribute(hwnd, ctypes.byref(data))
        except Exception as e:
            print(f"  [DWM] 启用毛玻璃失败: {e}")

    @staticmethod
    def set_rounded_corners(hwnd: int, prefer_small: bool = False):
        """设置 Windows 11 圆角"""
        try:
            pref = DWMWCP_ROUNDSMALL if prefer_small else DWMWCP_ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_WINDOW_CORNER_PREFERENCE,
                ctypes.byref(ctypes.c_int(pref)), ctypes.sizeof(ctypes.c_int)
            )
        except Exception as e:
            print(f"  [DWM] 设置圆角失败: {e}")

    @staticmethod
    def apply_all(tk_root, acrylic: bool = True, rounded: bool = True, color: int = 0x99B41E1E):
        """一键应用所有窗口效果"""
        try:
            hwnd = DWM.get_hwnd(tk_root)
            if acrylic:
                try:
                    DWM.enable_acrylic(hwnd, color)
                except Exception:
                    DWM.enable_blur(hwnd)
            if rounded:
                DWM.set_rounded_corners(hwnd)
        except Exception as e:
            print(f"  [DWM] 应用窗口效果失败: {e}")
