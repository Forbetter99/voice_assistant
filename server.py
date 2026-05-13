import asyncio
import base64
import io
import logging
import os
import tempfile
from datetime import datetime

import edge_tts
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from scipy.io import wavfile

from assistant.intents import IntentHandler
from assistant.nlu import NLUEngine
from assistant.stt import STTEngine
from config import Config

log_dir = Config.LOG_DIR
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f"server_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("server")

config = Config()

if "your-api-key" in config.DEEPSEEK_API_KEY:
    logger.error("请先在 config.py 中配置 DEEPSEEK_API_KEY")
    raise SystemExit(1)

logger.info("正在初始化模块...")
stt = STTEngine(config)
nlu = NLUEngine(config)
handler = IntentHandler()
logger.info("所有模块初始化完成")

app = FastAPI(title="语音助手")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    old_len = len(audio)
    new_len = int(old_len * target_sr / orig_sr)
    if new_len <= 1:
        return audio
    old_idx = np.arange(old_len)
    new_idx = np.arange(new_len) * (old_len - 1) / (new_len - 1)
    return np.interp(new_idx, old_idx, audio)


async def _generate_tts(text: str) -> str:
    if not text:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    try:
        tts = edge_tts.Communicate(text, config.TTS_VOICE)
        await tts.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/api/voice")
async def process_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        return {"text": "", "response": "没有收到音频数据", "audio": ""}

    try:
        sr, wav_data = wavfile.read(io.BytesIO(audio_bytes))
    except Exception as e:
        logger.warning(f"WAV 解析失败: {e}")
        return {"text": "", "response": "音频格式错误", "audio": ""}

    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)

    samples = wav_data.astype(np.float32) / 32768.0
    samples = _resample(samples, sr, 16000)

    text = stt.transcribe(samples)
    if not text:
        logger.info("STT 未识别到语音")
        return {"text": "", "response": "抱歉，我没有听清楚", "audio": ""}

    logger.info(f"STT: {text}")

    result = nlu.understand(text)
    if not result:
        return {"text": text, "response": "抱歉，我没有理解", "audio": ""}

    intent = result.get("intent", "chat")
    entities = result.get("entities", {})
    confidence = result.get("confidence", 0)
    response_text = result.get("response", "")

    action_text = handler.execute(intent, entities, config)
    final_text = response_text or action_text

    audio_base64 = await _generate_tts(final_text)

    return {
        "text": text,
        "intent": intent,
        "confidence": confidence,
        "response": final_text,
        "audio": audio_base64,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("  语音助手 Web 服务")
    print(f"  访问地址: http://localhost:8000")
    print("  手机在同一 WiFi 下访问局域网 IP")
    print("  Ctrl+C 退出")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
