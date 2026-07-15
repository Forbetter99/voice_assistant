"""
统一配置模块（提取自 gui.py 和 config.py）
"""
import os
from pathlib import Path

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
GENERATED_DIR = PROJECT_ROOT / "generated"
LOG_DIR = PROJECT_ROOT / "logs"

# ── 模型缓存路径（全部指向项目目录，实现自包含）──
os.environ.setdefault("WHISPER_CACHE_DIR", str(MODELS_DIR / "whisper"))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
os.environ.setdefault("OLLAMA_MODELS", str(MODELS_DIR / "ollama"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── 音频设置 ──
SAMPLE_RATE = 16000
BLOCK_SIZE = 512       # Silero VAD 需要 512 samples @ 16kHz
SILENCE_TIMEOUT = 1.5  # 静音超时秒数

# ── 唤醒词 ──
WAKE_WORD = "贾维斯"

# ── LLM 后端选择: "ollama" 或 "deepseek" ──
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # 默认用 Ollama（本地）

# ── Ollama（本地）──
OLLAMA_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_MODELS_DIR = str(MODELS_DIR / "ollama")

# ── DeepSeek API（云端）──
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ── TTS ──
TTS_VOICE = "zf_xiaobei"
TTS_SPEED = 1.3
TTS_SAMPLE_RATE = 24000

# ── 声纹验证 ──
SPEAKER_EMBEDDING_PATH = MODELS_DIR / "speaker_embedding.pt"
SPEAKER_THRESHOLD = 0.5

# ── UI ──
WINDOW_TITLE = "贾维斯 语音助手"
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
WINDOW_MIN_WIDTH = 480
WINDOW_MIN_HEIGHT = 400

# ── 日志 ──
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── 颜色主题（Catppuccin Mocha）──
COLORS = {
    "bg": "#1e1e2e",
    "surface": "#313244",
    "overlay": "#585b70",
    "text": "#cdd6f4",
    "text_muted": "#a6adc8",
    "text_dark": "#6c7086",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "orange": "#fab387",
    "chat_bg": "#181825",
    "log_bg": "#11111b",
}
