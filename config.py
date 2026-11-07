import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class Config:
    # DeepSeek API
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-chat"

    # Wake word
    WAKE_WORD = "你好小度"

    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORD_TIMEOUT = 8.0
    SILENCE_DURATION = 1.5

    # STT（统一使用本地 Qwen3-ASR-1.7B，见 assistant/stt.py）

    # TTS
    TTS_VOICE = "zh-CN-XiaoxiaoNeural"
    TTS_FALLBACK_VOICE = "zh-CN"

    # Log
    LOG_DIR = "logs"
