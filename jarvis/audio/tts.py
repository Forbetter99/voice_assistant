"""
文本转语音（TTS）— 基于 Kokoro（强制 CPU）
"""
import numpy as np
import sounddevice as sd
from jarvis.config import TTS_VOICE, TTS_SPEED, TTS_SAMPLE_RATE


class TTS:
    """基于 Kokoro 的本地语音合成"""

    def __init__(self):
        import os as _os
        from kokoro import KPipeline

        # 强制 CPU（GTX 960M 4GB 显存不够同时跑 whisper + kokoro）
        _os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.pipeline = KPipeline(lang_code="z", device="cpu")
        except Exception as e:
            print(f"  [TTS] 加载失败: {e}")
            self.pipeline = None

    def speak(self, text: str):
        if not text or not self.pipeline:
            return
        generator = self.pipeline(text, voice=TTS_VOICE, speed=TTS_SPEED)
        chunks = [audio for _, _, audio in generator]
        if not chunks:
            return
        audio = np.concatenate(chunks)
        peak = np.max(np.abs(audio))
        if peak > 0 and peak < 0.6:
            audio = audio * (0.7 / peak)
        audio = np.clip(audio, -1.0, 1.0)
        sd.play(audio, samplerate=TTS_SAMPLE_RATE)
        sd.wait()

    def speak_streaming(self, text: str):
        if not text or not self.pipeline:
            return
        generator = self.pipeline(text, voice=TTS_VOICE, speed=TTS_SPEED)
        for gs, ps, audio in generator:
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            peak = np.max(np.abs(audio))
            if peak > 0 and peak < 0.6:
                audio = audio * (0.7 / peak)
            audio = np.clip(audio, -1.0, 1.0)
            yield audio
