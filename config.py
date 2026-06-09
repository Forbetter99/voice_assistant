import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class Config:
    # DeepSeek API
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-f26a817d074543af92303575b8376e5a")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-chat"

    # Wake word
    WAKE_WORD = "你好小度"

    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    RECORD_TIMEOUT = 8.0
    SILENCE_DURATION = 1.5

    # STT
    WHISPER_MODEL_SIZE = "small"  # tiny, base, small, medium, large
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE_TYPE = "int8_float16"

    # TTS
    TTS_VOICE = "zh-CN-XiaoxiaoNeural"
    TTS_FALLBACK_VOICE = "zh-CN"

    # Log
    LOG_DIR = "logs"
