import logging
import re
import threading
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(self, config, stt_engine=None):
        self.config = config
        self.stt = stt_engine
        self._running = False
        self._detected = threading.Event()

    def wait_for_wake_word(self):
        logger.info(f"等待唤醒词: '{self.config.WAKE_WORD}'")
        self._detected.clear()
        self._running = True

        sample_rate = self.config.SAMPLE_RATE
        frame_size = int(sample_rate * 0.1)

        audio_buffer = []
        speech_mode = False
        silence_count = 0
        max_silence = int(self.config.SILENCE_DURATION / 0.1)

        def callback(indata, frames, time_info, status):
            nonlocal speech_mode, silence_count, audio_buffer
            rms = np.sqrt(np.mean(indata ** 2))
            if rms > 0.025:
                if not speech_mode:
                    speech_mode = True
                    audio_buffer = [indata.copy()]
                else:
                    audio_buffer.append(indata.copy())
                silence_count = 0
            elif speech_mode:
                audio_buffer.append(indata.copy())
                silence_count += 1
                if silence_count > max_silence:
                    audio = np.concatenate(audio_buffer, axis=0).flatten()
                    speech_mode = False
                    silence_count = 0
                    audio_buffer = []
                    self._check_wake_word(audio)

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=self.config.CHANNELS,
            callback=callback,
            blocksize=frame_size,
        )

        try:
            with stream:
                self._detected.wait()
        finally:
            self._running = False

        return True

    def _check_wake_word(self, audio_data):
        if self.stt is None:
            return
        audio_float = audio_data.astype(np.float32)
        text = self.stt.transcribe(audio_float)
        if text and self._match_wake_word(text):
            logger.info(f"唤醒词检测成功: '{text}'")
            self._detected.set()

    def _match_wake_word(self, text):
        wake = self.config.WAKE_WORD
        text_clean = re.sub(r"[^一-鿿\w]", "", text)
        wake_clean = re.sub(r"[^一-鿿\w]", "", wake)
        return wake_clean in text_clean or text_clean in wake_clean
