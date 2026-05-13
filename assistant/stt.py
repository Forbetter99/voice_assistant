import os
import logging

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTEngine:
    def __init__(self, config):
        self.config = config
        self._model = None
        logger.info(
            f"Loading whisper model '{config.WHISPER_MODEL_SIZE}' "
            f"on {config.WHISPER_DEVICE} ({config.WHISPER_COMPUTE_TYPE})..."
        )

    def _get_model(self):
        if self._model is None:
            self._model = WhisperModel(
                model_size_or_path=self.config.WHISPER_MODEL_SIZE,
                device=self.config.WHISPER_DEVICE,
                compute_type=self.config.WHISPER_COMPUTE_TYPE,
                download_root=os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "models"
                ),
            )
        return self._model

    def transcribe(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            return None

        audio_data = audio_data.astype("float32")
        model = self._get_model()

        segments, info = model.transcribe(audio_data, beam_size=5, language="zh")

        text = "".join(segment.text for segment in segments).strip()
        if text:
            logger.info(f"STT: [{info.language} {info.language_probability:.2f}] {text}")
        return text if text else None
