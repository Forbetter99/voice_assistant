import os
import logging

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

logger = logging.getLogger(__name__)


class STTEngine:
    """基于 Qwen3-ASR-1.7B 的语音识别（统一 ASR 后端）"""

    def __init__(self, config):
        self.config = config
        self._model = None
        logger.info("加载 ASR (Qwen3-ASR-1.7B) 语音识别模型...")

    def _get_model(self):
        if self._model is None:
            from pathlib import Path
            import torch
            from qwen_asr import Qwen3ASRModel

            # 解析本地模型路径：优先项目缓存，其次用户默认 HF 缓存
            repo = "models--Qwen--Qwen3-ASR-1.7B"
            local = None
            project_root = Path(__file__).resolve().parent.parent
            for base in (project_root / "models" / "huggingface",
                         Path.home() / ".cache" / "huggingface" / "hub"):
                ref = base / repo / "refs" / "main"
                if ref.exists():
                    snap = base / repo / "snapshots" / ref.read_text().strip()
                    if snap.exists():
                        local = str(snap)
                        break

            kwargs = {"dtype": torch.float16, "device_map": "cuda:0"} if torch.cuda.is_available() else {}
            self._model = Qwen3ASRModel.from_pretrained(local or "Qwen/Qwen3-ASR-1.7B", **kwargs)
        return self._model

    def transcribe(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            return None

        audio_float = np.asarray(audio_data, dtype=np.float32)
        model = self._get_model()
        results = model.transcribe((audio_float, 16000), language="Chinese")

        text = results[0].text.strip() if results else ""
        if text:
            logger.info(f"STT: {text}")
        return text if text else None
