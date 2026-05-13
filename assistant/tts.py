import asyncio
import os
import tempfile
import logging
import edge_tts
import pygame

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, config):
        self.config = config
        self._fallback = None
        pygame.mixer.init(frequency=24050)

    def _init_fallback(self):
        if self._fallback is None:
            import pyttsx3
            self._fallback = pyttsx3.init()
            voices = self._fallback.getProperty("voices")
            for v in voices:
                if "Chinese" in v.name or "zh" in v.id:
                    self._fallback.setProperty("voice", v.id)
                    break
        return self._fallback

    def speak(self, text):
        if not text:
            return
        try:
            asyncio.run(self._speak_edge(text))
        except Exception as e:
            logger.warning(f"edge-tts failed ({e}), falling back to pyttsx3")
            self._speak_fallback(text)

    async def _speak_edge(self, text):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            tts = edge_tts.Communicate(text, self.config.TTS_VOICE)
            await tts.save(tmp_path)
            self._play_mp3(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _speak_fallback(self, text):
        engine = self._init_fallback()
        engine.say(text)
        engine.runAndWait()

    def _play_mp3(self, path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
