"""
语音活动检测（VAD）— 基于 Silero VAD
提取自 gui.py
"""
import numpy as np
from jarvis.config import SAMPLE_RATE, BLOCK_SIZE


class VAD:
    """基于 Silero VAD 的语音活动检测"""

    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        torch.set_num_threads(1)
        self.model = load_silero_vad()
        self.model.eval()
        self._torch = torch

    def is_speech(self, audio: np.ndarray) -> bool:
        with self._torch.no_grad():
            prob = self.model(self._torch.from_numpy(audio), SAMPLE_RATE).item()
        return prob > 0.5
