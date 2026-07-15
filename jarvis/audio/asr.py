"""
自动语音识别（ASR）— 基于 openai-whisper
使用 CPU（GTX 960M Compute 5.0 不支持 whisper 的 CUDA 内核）
"""
import os
import numpy as np
from opencc import OpenCC

# 繁简转换器（繁体 → 简体）
_t2s = OpenCC("t2s")


class ASR:
    """基于 OpenAI Whisper 的本地语音识别"""

    def __init__(self, model_size="small"):
        import whisper
        self.model = whisper.load_model(model_size, device="cpu")
        print(f"  [ASR] whisper {model_size} loaded on CPU")

    def transcribe(self, audio: np.ndarray) -> str:
        import whisper
        audio_float = audio.astype(np.float32)
        result = whisper.transcribe(
            self.model,
            audio_float,
            language="zh",
            task="transcribe",
            fp16=False,
        )
        text = result["text"].strip()
        # 繁体转简体
        text = _t2s.convert(text)
        return text
