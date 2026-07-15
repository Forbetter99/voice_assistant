"""
声纹验证 — 基于 WeSpeaker
提取自 gui.py
"""
import os
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from jarvis.config import SPEAKER_EMBEDDING_PATH, SPEAKER_THRESHOLD


class SpeakerVerifier:
    """基于 WeSpeaker 的声纹注册与验证"""

    def __init__(self):
        self.model = None
        self.embedding = None
        self.embedding_path = SPEAKER_EMBEDDING_PATH
        self._load_model()

    def _load_model(self):
        try:
            import wespeaker
            self.model = wespeaker.load_model("chinese")
        except Exception as e:
            print(f"  [声纹] 加载失败: {e}")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def is_registered(self) -> bool:
        return self.embedding is not None or self.embedding_path.exists()

    def extract_embedding(self, audio: np.ndarray, sr: int = 16000):
        """从音频数组中提取声纹嵌入"""
        path = os.path.join(tempfile.gettempdir(), "jarvis_speaker_temp.wav")
        sf.write(path, audio, sr)
        return self.model.extract_embedding(path)

    def register(self, audio_list: list) -> str:
        """注册机主声纹：传入一个包含多段音频的列表，取平均嵌入"""
        import torch
        embs = []
        for i, audio in enumerate(audio_list):
            emb = self.extract_embedding(audio)
            embs.append(emb)
            print(f"  [声纹] 第{i+1}段嵌入已提取")
        stacked = torch.stack(embs)
        self.embedding = torch.mean(stacked, dim=0)
        self.embedding = self.embedding / torch.norm(self.embedding)
        torch.save(self.embedding, self.embedding_path)
        return f"声纹注册完成，共{len(audio_list)}段"

    def load(self) -> bool:
        """从文件加载机主声纹"""
        import torch
        if self.embedding_path.exists():
            self.embedding = torch.load(self.embedding_path, weights_only=True)
            return True
        return False

    def verify(self, audio: np.ndarray, sr: int = 16000, threshold: float = SPEAKER_THRESHOLD) -> tuple:
        """验证是否为机主，返回 (is_owner, confidence)"""
        import torch
        if not self.available:
            return True, 1.0
        if self.embedding is None:
            if not self.load():
                return True, 1.0
        emb = self.extract_embedding(audio)
        sim = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), self.embedding.unsqueeze(0)
        ).item()
        return sim >= threshold, sim

    def verify_from_file(self, wav_path: str, threshold: float = SPEAKER_THRESHOLD) -> tuple:
        """从 WAV 文件验证，返回 (is_owner, confidence)"""
        import torch
        if not self.available:
            return True, 1.0
        if self.embedding is None:
            if not self.load():
                return True, 1.0
        emb = self.model.extract_embedding(wav_path)
        sim = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), self.embedding.unsqueeze(0)
        ).item()
        return sim >= threshold, sim
