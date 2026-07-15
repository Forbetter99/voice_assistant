"""
录音工具 — VAD 自动停止录音
提取自 gui.py
"""
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from jarvis.config import SAMPLE_RATE, BLOCK_SIZE, SILENCE_TIMEOUT


def record_until_silence(vad, audio_q, stop_event=None, silence_timeout=None):
    """
    从 audio_q 读取音频，VAD 检测到静音后自动停止。
    返回累积的音频 numpy 数组，或 None（没录到声音）。

    参数：
        vad: VAD 实例
        audio_q: queue.Queue，由 sd.InputStream callback 填充
        stop_event: threading.Event，用于外部停止
        silence_timeout: 静音超时秒数（默认 SILENCE_TIMEOUT）
    """
    if silence_timeout is None:
        silence_timeout = SILENCE_TIMEOUT

    chunks = []
    silent_blocks = 0
    speak_started = False
    max_silent = int(silence_timeout * SAMPLE_RATE / BLOCK_SIZE)

    while True:
        if stop_event and stop_event.is_set():
            break
        try:
            chunk = audio_q.get(timeout=0.3)
        except queue.Empty:
            continue
        flat = chunk.flatten()
        chunks.append(flat)
        speech = vad.is_speech(flat)
        if speech:
            silent_blocks = 0
            speak_started = True
        elif speak_started:
            silent_blocks += 1
        if silent_blocks > max_silent and speak_started:
            break

    if not speak_started or not chunks:
        return None
    return np.concatenate(chunks)


def create_audio_stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
    """
    创建一个 sounddevice InputStream 并返回 (stream, audio_q)。
    audio_q 会被 callback 持续填充音频块。
    """
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    stream.start()
    return stream, audio_q


def stop_stream(stream):
    """安全停止并关闭音频流"""
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def drain_queue(audio_q):
    """耗尽队列中所有音频块，返回列表"""
    chunks = []
    if audio_q is None:
        return chunks
    while not audio_q.empty():
        try:
            chunks.append(audio_q.get_nowait().flatten())
        except queue.Empty:
            break
    return chunks
