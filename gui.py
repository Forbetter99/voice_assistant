"""
贾维斯语音助手 — 图形界面版
"""
import io
import os
import queue
import sys
import threading
import time
from pathlib import Path

# ── 本地模型路径 ──
_LOCAL_MODELS = Path(__file__).parent / "models"
os.environ.setdefault("HF_HUB_CACHE", str(_LOCAL_MODELS / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_LOCAL_MODELS / "torch"))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# 所有模型已缓存，离线加载避免镜像不稳定导致崩溃
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.stdout.reconfigure(line_buffering=True)

# 检测 Windows 系统代理设置是否有效
def _apply_windows_proxy():
    try:
        import winreg
        import urllib.request
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if enabled:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                proxy = f"http://{server}"
                # 验证代理是否可达，避免设置死代理导致所有请求失败
                try:
                    urllib.request.urlopen(proxy, timeout=2)
                except Exception:
                    return  # 代理不可用，不设置
                for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    os.environ.setdefault(var, proxy)
                # 有代理时直连 HuggingFace（镜像可能缺模型）
                os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    except Exception:
        pass
_apply_windows_proxy()

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk

SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # Silero VAD requires exactly 512 samples at 16kHz
SILENCE_TIMEOUT = 1.5


# ── VAD ──
class VAD:
    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        torch.set_num_threads(1)
        self.model = load_silero_vad()
        self.model.eval()
        self._torch = torch

    def is_speech(self, audio: np.ndarray) -> bool:
        with self._torch.no_grad():
            prob = self.model(self._torch.from_numpy(audio), SAMPLE_RATE).item()
        return prob > 0.5


# ── ASR ──
class ASR:
    """基于 Paraformer-zh 的本地语音识别（中文准确率 1.95% CER）"""

    def __init__(self):
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        from funasr import AutoModel
        self.model = AutoModel(
            model="paraformer-zh",
            device=device,
            disable_update=True,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        audio_float = audio.astype(np.float32)
        result = self.model.generate(input=audio_float)
        # result 格式: [{"text": "识别文字", "timestamp": [[...]]}]
        text = result[0]["text"] if result else ""
        # Paraformer 偶尔会在字间加空格，去掉
        text = text.replace(" ", "").strip()
        return text


# ── LLM（Agent 版，支持 Ollama 原生工具调用）──
class LLM:
    def __init__(self, model="qwen2.5:7b"):
        import requests
        self._requests = requests
        self.model = model
        self.history = []
        self.max_turns = 10
        self.system_prompt = (
            "你是贾维斯(Jarvis)，一个智能语音助手。"
            "你有以下工具可用：查天气、查时间、保存代码、打开网页、打开本地应用(如计算器、记事本等)。"
            "需要时直接使用对应的工具。"
            "用户的问题可能是语音识别结果，如有错别字请自动修正。"
            "用中文回答，简洁自然。使用工具后请告知用户结果。"
        )

    # ── 工具定义 ──────────────────────────────────

    @staticmethod
    def _build_tools() -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询指定城市的当前天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市中文名，例如：北京、上海、广州"
                            }
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "获取当前的日期和时间，包括年、月、日、星期、时、分、秒",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_code",
                    "description": "将代码保存到 generated 目录下的文件中。当你需要为用户保存代码时调用此工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "文件名，如 hello.py、index.html、style.css"
                            },
                            "content": {
                                "type": "string",
                                "description": "完整的代码内容"
                            },
                            "language": {
                                "type": "string",
                                "description": "编程语言，如 python、javascript、html"
                            }
                        },
                        "required": ["filename", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_browser",
                    "description": "在默认浏览器中打开一个网址",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "要打开的完整网址，如 https://www.baidu.com"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "打开 Windows 本地应用程序，如计算器、记事本、画图等",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app": {
                                "type": "string",
                                "description": "应用名称，支持：计算器、记事本、画图(画图工具)、命令提示符(cmd)、任务管理器、资源管理器、控制面板、截图工具、便签(sticky notes)、放大镜、录音机、时钟、设置"
                            }
                        },
                        "required": ["app"]
                    }
                }
            },
        ]

    # ── 工具实现 ──────────────────────────────────

    def _get_weather(self, city: str) -> str:
        try:
            resp = self._requests.get(
                f"https://wttr.in/{city}?format=%C+%t+%h+%w&lang=zh",
                timeout=10,
                headers={"User-Agent": "curl/8.0"}
            )
            if resp.status_code == 200:
                return f"{city}天气: {resp.text.strip()}"
            return f"无法获取{city}的天气信息"
        except Exception as e:
            return f"查询天气失败: {e}"

    def _get_time(self) -> str:
        import datetime
        now = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        wd = weekdays[now.weekday()]
        return f"当前时间: {now.year}年{now.month}月{now.day}日 {wd} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"

    def _save_code(self, filename: str, content: str, language: str = "") -> str:
        out_dir = Path(__file__).parent / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / filename).write_text(content, encoding="utf-8")
        msg = f"代码已保存到 {filename}"
        if language:
            msg += f" (语言: {language})"
        return msg

    def _open_browser(self, url: str) -> str:
        import webbrowser
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        try:
            webbrowser.open(url)
            return f"已打开网页: {url}"
        except Exception as e:
            return f"打开浏览器失败: {e}"

    def _open_app(self, app: str) -> str:
        """打开 Windows 本地应用程序"""
        import subprocess, os

        app_map = {
            "计算器": "calc.exe",
            "calculator": "calc.exe",
            "记事本": "notepad.exe",
            "notepad": "notepad.exe",
            "画图": "mspaint.exe",
            "画图工具": "mspaint.exe",
            "命令提示符": "cmd.exe",
            "cmd": "cmd.exe",
            "任务管理器": "taskmgr.exe",
            "task manager": "taskmgr.exe",
            "资源管理器": "explorer.exe",
            "文件资源管理器": "explorer.exe",
            "控制面板": "control",
            "截图工具": "snippingtool.exe",
            "便签": "stikynot.exe",
            "sticky notes": "stikynot.exe",
            "放大镜": "magnify.exe",
            "录音机": "voice recorder",
            "录音": "voice recorder",
            "时钟": "clock",
            "闹钟": "clock",
            "设置": "ms-settings:",
            "系统设置": "ms-settings:",
        }

        mapped = app_map.get(app.lower() if app.isascii() else app, None)
        if mapped:
            try:
                if mapped.startswith("ms-settings:"):
                    subprocess.Popen(f"start {mapped}", shell=True)
                else:
                    subprocess.Popen(mapped)
                return f"已打开: {app}"
            except Exception as e:
                return f"打开 {app} 失败: {e}"

        # 没匹配到已知应用名，尝试直接运行
        try:
            subprocess.Popen(app)
            return f"已尝试打开: {app}"
        except Exception as e:
            return f"无法打开 {app}，未知应用名。支持的计算器、记事本、画图、命令提示符等"

    def _save_code_block(self, content: str) -> str:
        """回退：从文本中检测代码块并保存"""
        import re
        blocks = re.findall(r"```(\w+)?\s*\n(.*?)```", content, re.DOTALL)
        saved = []
        for lang, code in blocks:
            code = code.strip()
            lines = code.split("\n")
            filename = "code.py"
            for prefix in ("# filename:", "// filename:", "; filename:"):
                if lines[0].lower().startswith(prefix.lower()):
                    filename = lines[0][len(prefix):].strip()
                    code = "\n".join(lines[1:]).strip()
                    break
            out_dir = Path(__file__).parent / "generated"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / filename).write_text(code, encoding="utf-8")
            saved.append(filename)
        return f"代码已保存: {', '.join(saved)}" if saved else ""

    # ── 工具调度 ──────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "get_weather":
            city = args.get("city", "")
            if not city:
                return "错误: 获取天气需要提供城市名"
            return self._get_weather(city)

        if name == "get_time":
            return self._get_time()

        if name == "save_code":
            filename = args.get("filename", "code.py")
            content = args.get("content", "")
            lang = args.get("language", "")
            if not content:
                return "错误: 没有要保存的代码内容"
            return self._save_code(filename, content, lang)

        if name == "open_browser":
            url = args.get("url", "")
            if not url:
                return "错误: 需要提供要打开的网址"
            return self._open_browser(url)

        if name == "open_app":
            app = args.get("app", "")
            if not app:
                return "错误: 需要提供应用名称"
            return self._open_app(app)

        return f"错误: 未知工具 '{name}'"

    # ── 对话 ──────────────────────────────────────

    def chat(self, user_text: str) -> str:
        import json, re
        self.history.append({"role": "user", "content": user_text})

        for _ in range(5):
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_prompt}]
                           + self.history[-self.max_turns * 2:],
                "stream": False,
                "tools": self._build_tools(),
            }

            # 请求（含 3 次重试）
            for attempt in range(3):
                try:
                    resp = self._requests.post(
                        "http://localhost:11434/api/chat", json=payload, timeout=120
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    break
                except self._requests.exceptions.ConnectionError:
                    time.sleep(1)
            else:
                return "抱歉，Ollama 连接失败，请检查是否已启动。"

            message = result.get("message", {})
            reply = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])

            # ── 主路径：Ollama 原生 tool_calls ──
            if tool_calls:
                assistant_msg = {"role": "assistant", "content": reply}
                assistant_msg["tool_calls"] = tool_calls
                self.history.append(assistant_msg)

                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    raw_args = func.get("arguments", {})

                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            raw_args = {}

                    result_content = self._execute_tool(name, raw_args)
                    self.history.append({
                        "role": "tool",
                        "content": result_content,
                    })

                continue

            # ── 回退 1a：检测原生格式 tool_call JSON ──
            # 格式：{"name":"get_weather","arguments":{"city":"北京"}}
            native_match = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', reply)
            if native_match:
                try:
                    name = native_match.group(1)
                    args = json.loads(native_match.group(2))
                    result_content = self._execute_tool(name, args)
                    self.history.append({
                        "role": "user",
                        "content": f"[工具结果] {result_content} 请据此回答用户。"
                    })
                    continue
                except (json.JSONDecodeError, Exception):
                    pass

            # ── 回退 1b：检测旧版 JSON 工具调用 ──
            tool_match = re.search(r'\{\s*"tool"\s*:\s*"([^"]+)"\s*.*?\}', reply)
            if tool_match:
                try:
                    tc = json.loads(tool_match.group(0))
                    if tc.get("tool") == "get_weather":
                        city = tc.get("city", "")
                        if city:
                            wx = self._get_weather(city)
                            self.history.append({
                                "role": "user",
                                "content": f"[天气实况] {wx} 请据此回答用户。"
                            })
                            continue
                except json.JSONDecodeError:
                    pass

            # ── 回退 2：检测代码块并保存 ──
            save_msg = self._save_code_block(reply)
            if save_msg:
                reply += f"\n\n({save_msg})"

            # 如果 reply 看起来是纯 JSON 工具调用（回退路径都没匹配到），不显示给用户
            if reply.strip().startswith("{") and reply.strip().endswith("}"):
                # 未知格式的 JSON，跳过本轮让 LLM 重试
                self.history.append({"role": "user", "content": "请用中文直接回答，不要输出 JSON。"})
                continue

            self.history.append({"role": "assistant", "content": reply})
            return reply

        return "任务已完成。"


# ── TTS ──
class TTS:
    def __init__(self):
        from kokoro import KPipeline
        try:
            self.pipeline = KPipeline(lang_code="z")
        except Exception as e:
            print(f"TTS 加载失败: {e}")
            self.pipeline = None

    def speak(self, text: str):
        """合成并播放语音（可从外部通过 sd.stop() 中断）"""
        if not text or not self.pipeline:
            return
        generator = self.pipeline(text, voice="zf_xiaobei", speed=1.0)
        chunks = [audio for _, _, audio in generator]
        if not chunks:
            return
        audio = np.concatenate(chunks)
        # 增益放大（原始音量偏低）
        peak = np.max(np.abs(audio))
        if peak > 0 and peak < 0.6:
            audio = audio * (0.7 / peak)
        audio = np.clip(audio, -1.0, 1.0)
        # 用 sounddevice 播放（支持 sd.stop() 从外部中断）
        sd.play(audio, samplerate=24000)
        sd.wait()


# ── 声纹验证（WeSpeaker）───────────────────
class SpeakerVerifier:
    """基于 WeSpeaker 的声纹注册与验证，只有机主能唤醒"""

    def __init__(self):
        self.model = None
        self.embedding = None  # 机主声纹嵌入
        self.embedding_path = Path(__file__).parent / "models" / "speaker_embedding.pt"
        self._load_model()

    def _load_model(self):
        try:
            import wespeaker
            self.model = wespeaker.load_model("chinese")
        except Exception as e:
            print(f"  [声纹] 加载失败: {e}")
            self.model = None

    @property
    def available(self) -> bool:
        """模型是否可用"""
        return self.model is not None

    @property
    def is_registered(self) -> bool:
        """是否已注册机主声纹"""
        return self.embedding is not None or self.embedding_path.exists()

    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> "torch.Tensor":
        """从音频数组中提取声纹嵌入"""
        import tempfile, os, soundfile as sf
        path = os.path.join(tempfile.gettempdir(), "jarvis_speaker_temp.wav")
        sf.write(path, audio, sr)
        return self.model.extract_embedding(path)

    def register(self, audio_list: list) -> str:
        """注册机主声纹：传入一个包含多段音频的列表，取平均嵌入"""
        import torch
        embs = []
        for i, audio in enumerate(audio_list):
            emb = self.extract_embedding(audio)
            embs.append(emb)
            print(f"  [声纹] 第{i+1}段嵌入已提取")
        # 平均多个嵌入
        stacked = torch.stack(embs)
        self.embedding = torch.mean(stacked, dim=0)
        # 归一化
        self.embedding = self.embedding / torch.norm(self.embedding)
        # 保存
        torch.save(self.embedding, self.embedding_path)
        return f"声纹注册完成，共{len(audio_list)}段"

    def load(self) -> bool:
        """从文件加载机主声纹"""
        import torch
        if self.embedding_path.exists():
            self.embedding = torch.load(self.embedding_path, weights_only=True)
            return True
        return False

    def verify(self, audio: np.ndarray, sr: int = 16000, threshold: float = 0.5) -> tuple:
        """验证是否为机主，返回 (is_owner, confidence)"""
        import torch
        if not self.available:
            return True, 1.0  # 模型不可用时默认通过
        if self.embedding is None:
            if not self.load():
                return True, 1.0  # 未注册时默认通过
        emb = self.extract_embedding(audio)
        sim = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), self.embedding.unsqueeze(0)
        ).item()
        return sim >= threshold, sim

    def verify_from_file(self, wav_path: str, threshold: float = 0.5) -> tuple:
        """从 WAV 文件验证，返回 (is_owner, confidence)"""
        import torch
        if not self.available:
            return True, 1.0
        if self.embedding is None:
            if not self.load():
                return True, 1.0
        emb = self.model.extract_embedding(wav_path)
        sim = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), self.embedding.unsqueeze(0)
        ).item()
        return sim >= threshold, sim
class ProgressBar(tk.Canvas):
    def __init__(self, parent, width=300, height=16, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg="#313244", highlightthickness=0, **kwargs)
        self._width = width
        self._bar = self.create_rectangle(0, 0, 0, height,
                                          fill="#89b4fa", outline="")

    def set(self, pct: int):
        w = int(self._width * pct / 100)
        self.coords(self._bar, 0, 0, w, self.winfo_reqheight())


# ── 加载线程 ──
def loading_screen(root, status_label, progress_bar, models):
    steps = [
        ("加载 ASR (Paraformer-zh) 语音识别模型...", 25),
        ("加载 VAD 语音检测模型...", 50),
        ("加载 LLM 语言模型...", 75),
        ("加载 TTS 语音合成模型...", 85),
    ]
    for i, (msg, val) in enumerate(steps):
        root.after(0, lambda m=msg: status_label.config(text=m))
        root.after(0, lambda v=val: progress_bar.set(v))
        root.after(0, lambda m=msg: root.app.log(m))
        root.update()
        models[i]()  # 执行加载
        time.sleep(0.3)
    root.after(0, lambda: progress_bar.set(100))
    root.after(0, lambda: status_label.config(text="点击按钮开始对话"))
    root.after(0, lambda: on_loaded(root))

    # 加载完成后隐藏进度条
    root.after(500, lambda: progress_bar.pack_forget())


def on_loaded(root):
    app = root.app
    app.record_btn.config(state="normal", text="🎤 按住说话")
    app.test_btn.config(state="normal")
    app.wake_btn.config(state="normal")
    app.register_btn.config(state="normal")
    app.owner_check.config(state="normal")
    if app.sv.is_registered:
        app.status_bar.config(text="就绪 — 声纹已注册，可开启'仅机主唤醒'")
    else:
        app.status_bar.config(text="就绪 — 建议先注册声纹")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("贾维斯 语音助手")
        self.root.geometry("600x500")
        self.root.minsize(480, 400)
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        self.vad = None
        self.asr = None
        self.llm = None
        self.tts = None
        self.recording = False
        self.audio_q = queue.Queue()
        self.conv_count = 0
        self.test_audio_q = queue.Queue()
        self.test_audio_chunks = []
        self._current_gen = 0  # 递增计数器，用于中断旧对话的处理线程
        self.wake_word_enabled = False  # 语音唤醒开关
        self._stop_wake = threading.Event()  # 用于停止唤醒监听线程
        self.conversation_mode = False  # 对话模式：唤醒后持续对话，说退出才结束
        self.sv = SpeakerVerifier()  # 声纹验证器
        self.owner_only = False  # 仅机主唤醒

        self._build_ui()

    def _build_ui(self):
        # 标题
        title = tk.Label(
            self.root, text="🤖 贾维斯", font=("微软雅黑", 20, "bold"),
            bg="#1e1e2e", fg="#cdd6f4"
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            self.root, text="全本地语音助手", font=("微软雅黑", 10),
            bg="#1e1e2e", fg="#a6adc8"
        )
        subtitle.pack(pady=(0, 10))

        # ── 选项卡 ──
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        style = ttk.Style()
        style.configure("TNotebook", background="#1e1e2e")
        style.configure("TNotebook.Tab", padding=[12, 4])

        # 对话页
        chat_frame = tk.Frame(notebook, bg="#1e1e2e")
        notebook.add(chat_frame, text="  对话 ")

        self.text = tk.Text(
            chat_frame, font=("微软雅黑", 11), wrap="word",
            bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4",
            relief="flat", borderwidth=0, padx=12, pady=12,
            state="disabled"
        )
        scrollbar = tk.Scrollbar(chat_frame, command=self.text.yview, bg="#1e1e2e")
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 日志页
        log_frame = tk.Frame(notebook, bg="#1e1e2e")
        notebook.add(log_frame, text="  日志 ")

        self.log_text = tk.Text(
            log_frame, font=("Consolas", 10), wrap="word",
            bg="#11111b", fg="#a6adc8", insertbackground="#a6adc8",
            relief="flat", borderwidth=0, padx=12, pady=12,
            state="disabled"
        )
        log_scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview, bg="#1e1e2e")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

        self._notebook = notebook

        # 录音按钮
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=(0, 10))

        self.record_btn = tk.Button(
            btn_frame, text="加载中...", font=("微软雅黑", 14, "bold"),
            width=16, height=2, state="disabled",
            bg="#585b70", fg="#cdd6f4", activebackground="#f38ba8",
            relief="flat", borderwidth=0, cursor="hand2",
        )
        self.record_btn.pack()

        test_frame = tk.Frame(self.root, bg="#1e1e2e")
        test_frame.pack(pady=(0, 5))

        self.test_btn = tk.Button(
            test_frame, text="🎧 测试录音", font=("微软雅黑", 10),
            width=12, state="disabled",
            bg="#45475a", fg="#a6adc8", activebackground="#89b4fa",
            relief="flat", borderwidth=0, cursor="hand2",
        )
        self.test_btn.pack()
        self.test_btn.bind("<ButtonPress-1>", self.start_test_record)
        self.test_btn.bind("<ButtonRelease-1>", self.stop_test_record)
        self.record_btn.bind("<ButtonPress-1>", self.start_record)
        self.record_btn.bind("<ButtonRelease-1>", self.stop_record)

        # 声纹注册按钮
        self.register_btn = tk.Button(
            test_frame, text="🎙️ 注册声纹", font=("微软雅黑", 10),
            width=12, state="disabled",
            bg="#45475a", fg="#a6adc8", activebackground="#fab387",
            relief="flat", borderwidth=0, cursor="hand2",
            command=self._start_speaker_registration
        )
        self.register_btn.pack(side="left", padx=(0, 5))

        # 仅机主唤醒
        self.owner_var = tk.BooleanVar(value=False)
        self.owner_check = tk.Checkbutton(
            test_frame, text="仅机主唤醒", variable=self.owner_var,
            font=("微软雅黑", 9), bg="#1e1e2e", fg="#a6adc8",
            selectcolor="#1e1e2e", activebackground="#1e1e2e",
            activeforeground="#cdd6f4", state="disabled",
            command=self._on_owner_toggle
        )
        self.owner_check.pack(side="left")
        if self.sv.is_registered:
            self.owner_var.set(True)
            self.owner_only = True

        # 语音唤醒按钮
        self.wake_btn = tk.Button(
            btn_frame, text="🤖 语音唤醒", font=("微软雅黑", 10),
            width=12, state="disabled",
            bg="#45475a", fg="#a6adc8", activebackground="#a6e3a1",
            relief="flat", borderwidth=0, cursor="hand2",
            command=self._toggle_wake_word
        )
        self.wake_btn.pack(pady=(5, 0))

        # 状态栏
        self.status_bar = tk.Label(
            self.root, text="正在加载模型...", font=("微软雅黑", 9),
            bg="#1e1e2e", fg="#a6adc8"
        )
        self.status_bar.pack(pady=(0, 15))

    def log(self, msg: str):
        """在日志页追加一条带时间戳的记录"""
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_test_record(self, event=None):
        """按住测试录音"""
        if self.recording:
            return
        self.recording = True
        self.test_btn.configure(text="🔴 测试录音中...", bg="#f38ba8")
        self.status_bar.config(text="测试录音中... 松手回放")
        self.log("🎤 测试录音开始")
        self.test_audio_q = queue.Queue()
        self.test_audio_chunks = []

        def callback(indata, frames, time_info, status):
            self.test_audio_q.put(indata.copy())

        self._test_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            channels=1, dtype="float32", callback=callback
        )
        self._test_stream.start()

        # 后台收集音频
        def collect():
            while self.recording:
                try:
                    chunk = self.test_audio_q.get(timeout=0.3)
                    self.test_audio_chunks.append(chunk.flatten())
                except queue.Empty:
                    pass

        threading.Thread(target=collect, daemon=True).start()

    def stop_test_record(self, event=None):
        """松开回放测试录音"""
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, "_test_stream") and self._test_stream:
            try:
                self._test_stream.stop()
                self._test_stream.close()
            except Exception:
                pass

        # 等 collect 线程耗尽队列
        time.sleep(0.5)
        while not self.test_audio_q.empty():
            try:
                self.test_audio_chunks.append(self.test_audio_q.get_nowait().flatten())
            except queue.Empty:
                break

        self.status_bar.config(text="回放中...")
        self.log("🔊 回放测试录音")

        if not self.test_audio_chunks:
            self.log("⚠️ 没有录到声音")
            self.test_btn.configure(text="🎧 测试录音", bg="#45475a")
            self.status_bar.config(text="就绪 — 按住按钮说话，松手识别")
            return

        audio = np.concatenate(self.test_audio_chunks)
        self.log(f"📊 录音时长: {len(audio)/SAMPLE_RATE:.1f}s")

        # 直接回放
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()

        self.log("✅ 回放完成")
        self.test_btn.configure(text="🎧 测试录音", bg="#45475a")
        self.status_bar.config(text="就绪 — 按住按钮说话，松手识别")

    def stop_playback(self):
        """中断当前语音播放和处理进程"""
        self._current_gen += 1
        sd.stop()
        self.log("⏹️ 中断当前对话")

    # ── 语音唤醒 ──────────────────────────────────

    def _on_owner_toggle(self):
        """仅机主唤醒开关"""
        self.owner_only = self.owner_var.get()
        if self.owner_only and not self.sv.is_registered:
            self.log("⚠️ 请先注册声纹，再开启'仅机主唤醒'")
            self.owner_var.set(False)
            self.owner_only = False
            self._start_speaker_registration()
        elif self.owner_only:
            self.log("🔒 仅机主唤醒已开启")
        else:
            self.log("🔓 仅机主唤醒已关闭")

    def _start_speaker_registration(self):
        """注册声纹：录制3段语音"""
        if self.recording:
            return
        self.log("🎙️ 开始声纹注册")
        self._regist_phrases = []
        self._regist_step = 0
        self._regist_total = 3
        self.log(f"📢 请说第1句话（共{self._regist_total}句），说完静音自动停止")
        self.status_bar.config(text=f"声纹注册 第1/{self._regist_total}句 — 请说话")
        self._start_regist_record()

    def _start_regist_record(self):
        """启动录音用于声纹注册"""
        self._stop_wake.set()
        self.recording = True
        self.audio_q = queue.Queue()
        self._regist_chunks = []

        def callback(indata, frames, time_info, status):
            self.audio_q.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            channels=1, dtype="float32", callback=callback
        )
        self._stream.start()

        def collect():
            silent_blocks = 0
            speak_started = False
            max_silent = int(SILENCE_TIMEOUT * SAMPLE_RATE / BLOCK_SIZE)
            while self.recording:
                try:
                    chunk = self.audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                self._regist_chunks.append(chunk.flatten())
                speech = self.vad.is_speech(chunk.flatten())
                if speech:
                    silent_blocks = 0
                    speak_started = True
                elif speak_started:
                    silent_blocks += 1
                if silent_blocks > max_silent and speak_started:
                    self.root.after(0, self._stop_regist_record)
                    break

        threading.Thread(target=collect, daemon=True).start()

    def _stop_regist_record(self):
        """停止注册录音，处理当前句"""
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, "_stream") and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        time.sleep(0.2)

        if not self._regist_chunks:
            self.log("⚠️ 没有录到声音，重试")
            self.root.after(500, self._start_regist_record)
            return

        audio = np.concatenate(self._regist_chunks)
        if len(audio) < SAMPLE_RATE * 0.3:
            self.log("⚠️ 录音太短，重试")
            self.root.after(500, self._start_regist_record)
            return

        self._regist_phrases.append(audio)
        self._regist_step += 1
        self.log(f"✅ 第{self._regist_step}段录音完成")

        if self._regist_step >= self._regist_total:
            # 注册
            self.log("🧬 正在提取声纹特征...")
            self.status_bar.config(text="正在处理声纹...")
            threading.Thread(target=self._do_registration, daemon=True).start()
        else:
            step = self._regist_step + 1
            self.status_bar.config(text=f"声纹注册 第{step}/{self._regist_total}句 — 请说话")
            self.log(f"📢 请说第{step}句话")
            self.root.after(500, self._start_regist_record)

    def _do_registration(self):
        """后台执行声纹注册"""
        try:
            msg = self.sv.register(self._regist_phrases)
            self.log(f"✅ {msg}")
            self.root.after(0, lambda: self.status_bar.config(text="声纹注册成功 ✅"))
            self.root.after(0, lambda: self.owner_check.config(state="normal"))
            self.root.after(0, lambda: self.owner_var.set(True))
            self.root.after(0, lambda: setattr(self, 'owner_only', True))
            self.root.after(0, lambda: self.log("🔒 已自动开启'仅机主唤醒'"))
            self.root.after(2000, lambda: self.status_bar.config(
                text="就绪 — 声纹已注册，仅机主可唤醒" if self.owner_only else "就绪"
            ))
            # 重新唤醒监听
            self.root.after(500, self._try_restart_wake)
        except Exception as e:
            self.log(f"❌ 声纹注册失败: {e}")
            self.root.after(500, self._try_restart_wake)
        finally:
            self._regist_phrases = []

    def _try_restart_wake(self):
        """尝试重启唤醒监听"""
        if self.wake_word_enabled and not self.recording:
            self._stop_wake.clear()
            self._start_wake_listener()

    def _toggle_wake_word(self):
        """开关语音唤醒功能"""
        if self.wake_word_enabled:
            self.wake_word_enabled = False
            self._stop_wake.set()
            self.wake_btn.config(text="🤖 语音唤醒", bg="#45475a", fg="#a6adc8")
            self.log("⏹️ 语音唤醒已关闭")
            # 如果在对话模式中，一并退出
            if self.conversation_mode:
                self._current_gen += 1
                self.conversation_mode = False
                self.record_btn.configure(text="🎤 按住说话", bg="#585b70")
                self.status_bar.config(text="就绪 — 按住按钮说话，松手识别")
            return

        self.wake_word_enabled = True
        self._stop_wake.clear()
        self.wake_btn.config(text="🔊 聆听中...", bg="#a6e3a1", fg="#11111b")
        self.log("🔊 语音唤醒已开启，说'贾维斯'唤醒")
        self._start_wake_listener()

    def _start_wake_listener(self):
        """启动唤醒监听线程"""
        self._wake_thread = threading.Thread(target=self._wake_word_loop, daemon=True)
        self._wake_thread.start()

    def _wake_word_loop(self):
        """后台监听唤醒词'贾维斯'"""
        import numpy as np
        import tempfile, os
        wake_q = queue.Queue()

        def callback(indata, frames, time_info, status):
            wake_q.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                channels=1, dtype="float32", callback=callback
            ):
                speech_blocks = []
                speech_active = False
                # 约 2 秒的音频用于检测
                max_check = int(2.0 * SAMPLE_RATE / BLOCK_SIZE)

                while self.wake_word_enabled and not self._stop_wake.is_set():
                    if self.recording:
                        time.sleep(0.1)
                        continue
                    try:
                        chunk = wake_q.get(timeout=0.3)
                    except queue.Empty:
                        continue

                    flat = chunk.flatten()
                    is_speech = self.vad.is_speech(flat)

                    if is_speech:
                        if not speech_active:
                            speech_active = True
                            speech_blocks = [flat]
                        else:
                            speech_blocks.append(flat)

                        # 积累够了就检测
                        if len(speech_blocks) >= max_check:
                            audio = np.concatenate(speech_blocks)
                            text = self.asr.transcribe(audio)
                            if "贾维斯" in text:
                                self._on_wake_verified(audio)
                                return
                            speech_blocks = []  # 没检测到，继续听
                    else:
                        # 静音后检查刚刚说的内容
                        if speech_active and len(speech_blocks) >= int(0.3 * SAMPLE_RATE / BLOCK_SIZE):
                            audio = np.concatenate(speech_blocks)
                            text = self.asr.transcribe(audio)
                            if "贾维斯" in text:
                                self._on_wake_verified(audio)
                                return
                        speech_blocks = []
                        speech_active = False
        except Exception as e:
            self.log(f"⚠️ 唤醒监听异常: {e}")
        finally:
            self._on_wake_loop_end()

    def _on_wake_verified(self, audio: np.ndarray):
        """检测到'贾维斯'后验证声纹（如开启）"""
        if self.owner_only and self.sv.available:
            self.log("🧬 验证声纹...")
            try:
                is_owner, confidence = self.sv.verify(audio, SAMPLE_RATE)
                if not is_owner:
                    self.log(f"⛔ 声纹不匹配 (confidence={confidence:.2f})，非机主已忽略")
                    return
                self.log(f"✅ 声纹验证通过 (confidence={confidence:.2f})")
            except Exception as e:
                self.log(f"⚠️ 声纹验证异常: {e}（默认通过）")
        self.root.after(0, self._on_wake_word_detected)

    def _on_wake_loop_end(self):
        """唤醒监听线程结束后自动重启"""
        if self.wake_word_enabled and not self._stop_wake.is_set() and not self.recording:
            self._start_wake_listener()

    def _begin_conversation_round(self):
        """对话模式下自动开始一轮录音（VAD 自动停止）"""
        if not self.conversation_mode or self.recording:
            return
        # 关闭可能残留的旧流
        if hasattr(self, "_stream") and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stop_wake.set()
        self.recording = True
        self.record_btn.configure(text="🔴 聆听中...", bg="#f38ba8")
        self.status_bar.config(text="对话模式 — 请说话")
        self.audio_q = queue.Queue()
        self.audio_chunks = []

        def callback(indata, frames, time_info, status):
            self.audio_q.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            channels=1, dtype="float32", callback=callback
        )
        self._stream.start()

        def collect():
            silent_blocks = 0
            speak_started = False
            max_silent = int(SILENCE_TIMEOUT * SAMPLE_RATE / BLOCK_SIZE)
            while self.recording and self.conversation_mode:
                try:
                    chunk = self.audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                self.audio_chunks.append(chunk.flatten())
                speech = self.vad.is_speech(chunk.flatten())
                if speech:
                    silent_blocks = 0
                    speak_started = True
                elif speak_started:
                    silent_blocks += 1
                if silent_blocks > max_silent and speak_started:
                    self.root.after(0, lambda: self.log("🔇 检测到静音，自动停止"))
                    self.root.after(0, self._stop_conversation_record)
                    break

        threading.Thread(target=collect, daemon=True).start()

    def _stop_conversation_record(self):
        """对话模式停止录音并处理"""
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, "_stream") and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        time.sleep(0.3)
        self._process_audio()

    def _exit_conversation_mode(self):
        """退出对话模式，回到唤醒待机状态"""
        self.conversation_mode = False
        self._current_gen += 1
        self.log("⏹️ 已退出对话模式")
        # _safe_reset_btn 会自动处理 UI 重置和唤醒监听重启
        self._safe_reset_btn(self._current_gen)

    def _on_wake_word_detected(self):
        """唤醒词检测到，进入对话模式"""
        self.log("🔊 检测到唤醒词 '贾维斯'，进入对话模式")
        self.status_bar.config(text="对话模式 — 说'退出'结束对话")
        # 播放提示音
        beep_t = np.linspace(0, 0.15, int(24000 * 0.15), False)
        beep_audio = np.sin(2 * np.pi * 800 * beep_t) * 0.3
        sd.play(beep_audio, samplerate=24000)
        self.conversation_mode = True
        self._begin_conversation_round()

    def _append_text(self, text, tag=None):
        self.text.configure(state="normal")
        if tag:
            self.text.insert("end", text + "\n", tag)
        else:
            self.text.insert("end", text + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def start_record(self, event=None, from_wake=False):
        """开始录音（from_wake=True 表示由唤醒词触发，对话模式持续对话）"""
        if not self.asr or self.recording:
            return

        # 手动录音时退出对话模式、关闭唤醒监听
        if not from_wake:
            self.conversation_mode = False
            self._stop_wake.set()
            self.stop_playback()

        self.recording = True
        self.record_btn.configure(text="🔴 录音中...", bg="#f38ba8")
        self.status_bar.config(text="录音中 — 说完自动识别" if from_wake else "录音中 — 松手识别")
        self.log("🔊 唤醒后录音" if from_wake else "🎤 开始录音")
        self.audio_q = queue.Queue()
        self.audio_chunks = []

        def callback(indata, frames, time_info, status):
            self.audio_q.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            channels=1, dtype="float32", callback=callback
        )
        self._stream.start()

        # 单线程：采集全部音频 + VAD 自动停止检测
        def collect():
            silent_blocks = 0
            speak_started = False
            max_silent = int(SILENCE_TIMEOUT * SAMPLE_RATE / BLOCK_SIZE)
            while self.recording:
                try:
                    chunk = self.audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                # 始终追加（不过滤）
                self.audio_chunks.append(chunk.flatten())
                # VAD 检测用于自动停止
                speech = self.vad.is_speech(chunk.flatten())
                if speech:
                    silent_blocks = 0
                    speak_started = True
                elif speak_started:
                    silent_blocks += 1
                if silent_blocks > max_silent and speak_started:
                    self.root.after(0, lambda: self.log("🔇 检测到静音，自动停止"))
                    self.root.after(0, self.stop_record_auto)
                    break

        threading.Thread(target=collect, daemon=True).start()

    def stop_record_auto(self):
        """VAD 自动触发停止"""
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, "_stream") and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        time.sleep(0.3)
        self._process_audio()

    def stop_record(self, event=None):
        """松开停止录音"""
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, "_stream") and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self.log("✋ 手动停止录音")
        time.sleep(0.3)
        self._process_audio()

    def _process_audio(self):
        self._current_gen += 1
        my_gen = self._current_gen

        self.record_btn.configure(text="⏳ 处理中...", bg="#585b70")
        # 保持按钮可用，让用户可以随时中断
        self.record_btn.configure(state="normal")
        self.status_bar.config(text="识别中...")

        # 耗尽队列中可能残留的音频
        if hasattr(self, "audio_q"):
            while not self.audio_q.empty():
                try:
                    self.audio_chunks.append(self.audio_q.get_nowait().flatten())
                except queue.Empty:
                    break

        if not self.audio_chunks:
            self.log("⚠️ 没有录到音频")
            self._append_text("[没有录到声音，请重试]", "gray")
            self._safe_reset_btn(my_gen)
            return

        audio = np.concatenate(self.audio_chunks)
        if len(audio) < SAMPLE_RATE * 0.3:
            self.log("⚠️ 录音太短，已忽略")
            self._append_text(f"[录音太短，已忽略]", "gray")
            self._safe_reset_btn(my_gen)
            return

        self.log("📝 开始语音识别...")

        def work():
            if my_gen != self._current_gen:
                return
            user_text = ""
            try:
                text = self.asr.transcribe(audio)
                user_text = text
                if not text:
                    self.root.after(0, lambda: self.log("⚠️ 语音识别无结果"))
                    self.root.after(0, lambda: self._append_text(f"[没听清，请再说一遍]", "gray"))
                    self._safe_reset_btn(my_gen)
                    return
                self.root.after(0, lambda: self.log(f"🧑 识别结果: {text}"))
                self.root.after(0, lambda: self._append_text(f"🧑 你: {text}"))
                self.root.after(0, lambda: self.status_bar.config(text="思考中..."))
                self.root.after(0, lambda: self.log("💭 请求 LLM..."))

                if my_gen != self._current_gen:
                    return

                reply = self.llm.chat(text)

                if my_gen != self._current_gen:
                    return

                self.root.after(0, lambda: self.log(f"🤖 LLM 回复: {reply}"))
                self.root.after(0, lambda: self._append_text(f"🤖 贾维斯: {reply}", "jarvis"))
                self.root.after(0, lambda: self.status_bar.config(text="播放语音..."))
                self.root.after(0, lambda: self.log("🔊 合成语音并播放..."))

                self.tts.speak(reply)

                if my_gen != self._current_gen:
                    return

                self.root.after(0, lambda: self.log("✅ 播放完成"))
            except Exception as e:
                self.root.after(0, lambda e=e: self.log(f"❌ 处理出错: {e}"))
            finally:
                if not self.conversation_mode or my_gen != self._current_gen:
                    self._safe_reset_btn(my_gen)
                elif any(w in user_text for w in ["退出", "再见", "结束", "没有问题了", "就这些"]):
                    self.log(f"⏹️ 检测到退出词 '{user_text}'，退出对话模式")
                    self.root.after(0, self._exit_conversation_mode)
                else:
                    self.root.after(0, self._begin_conversation_round)

        threading.Thread(target=work, daemon=True).start()

    def _reset_btn(self):
        self.record_btn.configure(state="normal", text="🎤 按住说话", bg="#585b70")

    def _safe_reset_btn(self, gen):
        """仅在指定代仍为当前代时重置按钮状态（防止被中断后误重置）"""
        def reset():
            if gen == self._current_gen:
                self.record_btn.configure(text="🎤 按住说话")
                self.status_bar.config(text="就绪 — 按住按钮说话，松手识别")
                # 如果开启了唤醒，重新启动监听线程
                if self.wake_word_enabled and not self.recording:
                    self._start_wake_listener()
        self.root.after(0, reset)


def main():
    root = tk.Tk()
    app = App(root)
    root.app = app

    # 样式标签
    app.text.tag_configure("jarvis", foreground="#89b4fa")
    app.text.tag_configure("gray", foreground="#6c7086")

    # 模型引用，供按序加载
    models = [
        lambda: setattr(app, 'asr', ASR()),
        lambda: setattr(app, 'vad', VAD()),
        lambda: setattr(app, 'llm', LLM()),
        lambda: setattr(app, 'tts', TTS()),
    ]

    status_label = app.status_bar
    progress_bar = ProgressBar(root, width=300, height=14)
    progress_bar.pack(before=app.status_bar, pady=(0, 5))

    threading.Thread(
        target=loading_screen,
        args=(root, status_label, progress_bar, models),
        daemon=True
    ).start()

    root.mainloop()


if __name__ == "__main__":
    main()
