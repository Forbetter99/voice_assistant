"""
音频电平表（VU 表）— 实时显示录音音量
"""
import tkinter as tk


class AudioLevelMeter(tk.Canvas):
    """实时音频电平 VU 表，使用 Canvas 绘制条形"""

    def __init__(self, parent, width=200, height=24, num_bars=20, **kwargs):
        super().__init__(
            parent, width=width, height=height,
            bg="#313244", highlightthickness=0, **kwargs
        )
        self._width = width
        self._height = height
        self._num_bars = num_bars
        self._bars = []
        self._level = 0.0

        bar_width = (width - (num_bars - 1) * 2) / num_bars
        for i in range(num_bars):
            x = i * (bar_width + 2)
            bar = self.create_rectangle(
                x, 1, x + bar_width, height - 1,
                fill="#585b70", outline="", width=0
            )
            self._bars.append(bar)

        # 颜色渐变：绿 → 黄 → 红
        self._colors = []
        green_count = num_bars * 3 // 5
        yellow_count = num_bars // 5
        red_count = num_bars - green_count - yellow_count

        for i in range(green_count):
            self._colors.append("#a6e3a1")
        for i in range(yellow_count):
            self._colors.append("#f9e2af")
        for i in range(red_count):
            self._colors.append("#f38ba8")

    def set_level(self, rms: float):
        """设置电平值（0.0 ~ 1.0），更新显示"""
        self._level = max(0.0, min(1.0, rms))
        active = int(self._level * self._num_bars)
        active = max(0, min(active, self._num_bars))

        for i, bar in enumerate(self._bars):
            color = self._colors[i] if i < active else "#585b70"
            self.itemconfig(bar, fill=color)

    def reset(self):
        """重置所有条为灰色"""
        self.set_level(0.0)
