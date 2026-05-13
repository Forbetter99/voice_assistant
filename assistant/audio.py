import threading
import numpy as np
import sounddevice as sd


class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._audio_buffer = []
        self._is_recording = False
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if self._is_recording:
            with self._lock:
                self._audio_buffer.append(indata.copy())

    def record(self, timeout=8.0, silence_duration=1.5):
        self._audio_buffer = []
        self._is_recording = True

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            blocksize=int(self.sample_rate * 0.1),
        )

        with stream:
            sd.sleep(int(timeout * 1000))

        self._is_recording = False

        with self._lock:
            if not self._audio_buffer:
                return None
            audio = np.concatenate(self._audio_buffer, axis=0)
        return audio.flatten()

    def record_until_silence(self, timeout=8.0, silence_duration=1.5):
        self._audio_buffer = []
        self._is_recording = True
        silence_frames = 0
        silence_threshold_frames = int(silence_duration / 0.1)
        has_spoken = False

        def callback(indata, frames, time_info, status):
            nonlocal silence_frames, has_spoken
            if self._is_recording:
                rms = np.sqrt(np.mean(indata ** 2))
                with self._lock:
                    self._audio_buffer.append(indata.copy())
                if rms > 0.02:
                    has_spoken = True
                    silence_frames = 0
                else:
                    if has_spoken:
                        silence_frames += 1

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
            blocksize=int(self.sample_rate * 0.1),
        )

        with stream:
            elapsed = 0
            while elapsed < timeout:
                sd.sleep(100)
                elapsed += 0.1
                if has_spoken and silence_frames >= silence_threshold_frames:
                    break

        self._is_recording = False

        with self._lock:
            if not self._audio_buffer:
                return None
            audio = np.concatenate(self._audio_buffer, axis=0)

        if not has_spoken:
            return None
        return audio.flatten()

    @staticmethod
    def play_audio(audio_data, sample_rate=16000):
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()
