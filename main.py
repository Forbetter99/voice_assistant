import logging
import os
import sys
from datetime import datetime

from config import Config
from assistant.audio import AudioRecorder
from assistant.stt import STTEngine
from assistant.nlu import NLUEngine
from assistant.tts import TTSEngine
from assistant.wake import WakeWordDetector
from assistant.intents import IntentHandler

log_dir = Config.LOG_DIR
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f"assistant_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


def print_banner():
    print("=" * 50)
    print("  智能语音助手 v1.0")
    print(f"  唤醒词: '{Config.WAKE_WORD}'")
    print(f"  LLM: DeepSeek")
    print("=" * 50)
    print("  说唤醒词开始对话，Ctrl+C 退出")
    print()


def main():
    config = Config()
    if "your-api-key" in config.DEEPSEEK_API_KEY:
        logger.error("请先在 config.py 中配置 DEEPSEEK_API_KEY")
        print("\n❌ 请先设置 DeepSeek API Key:")
        print("   方式一: 编辑 config.py 文件")
        print("   方式二: 设置环境变量 set DEEPSEEK_API_KEY=sk-xxxx")
        sys.exit(1)

    print_banner()

    logger.info("正在初始化模块...")
    audio = AudioRecorder(
        sample_rate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
    )
    stt = STTEngine(config)
    nlu = NLUEngine(config)
    tts = TTSEngine(config)
    wake = WakeWordDetector(config, stt_engine=stt)
    handler = IntentHandler()

    logger.info("所有模块初始化完成")

    try:
        while True:
            print("\n🎤 等待唤醒...", end="", flush=True)
            wake.wait_for_wake_word()
            print("\r🔊 唤醒成功! 请说出你的指令...", flush=True)

            audio_data = audio.record_until_silence(
                timeout=config.RECORD_TIMEOUT,
                silence_duration=config.SILENCE_DURATION,
            )

            if audio_data is None:
                print("\r⏺  没有检测到语音", flush=True)
                continue

            print("\r⏺  正在识别...", flush=True)
            text = stt.transcribe(audio_data)

            if not text:
                print("\r❌ 未能识别语音", flush=True)
                tts.speak("抱歉，我没有听清楚")
                continue

            print(f"\r💬 你说: {text}", flush=True)

            print("\r🧠 正在理解意图...", flush=True)
            result = nlu.understand(text)

            if not result:
                continue

            intent = result.get("intent", "chat")
            confidence = result.get("confidence", 0)
            entities = result.get("entities", {})
            response = result.get("response", "")

            print(f"\r🎯 意图: {intent} (置信度: {confidence:.2f})", flush=True)

            print("\r⚡ 执行中...", flush=True)
            action_response = handler.execute(intent, entities, config)

            final_response = response if response else action_response
            print(f"\r🤖 助手: {final_response}", flush=True)
            tts.speak(final_response)

    except KeyboardInterrupt:
        print("\n\n👋 再见!")
        logger.info("用户退出")
    except Exception as e:
        logger.error(f"运行时错误: {e}", exc_info=True)
        print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
