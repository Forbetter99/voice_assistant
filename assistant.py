"""
Jarvis 语音助手 - 全本地运行
唤醒 -> VAD(静音检测) -> ASR(语音识别) -> LLM(理解) -> TTS(语音合成)

注意：必须先加载 ASR (CTranslate2) 再加载 VAD/PyTorch，
否则同进程内会 segfault。
"""
import io
import queue
import sys
import time
import threading
from pathlib import Path

# 确保 stdout 立即刷新（后台运行时可能被缓冲）
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

# ── 配置 ──────────────────────────────────────────
import os
# ── 本地模型路径 ──
_LOCAL_MODELS = Path(__file__).parent / "models"
os.environ.setdefault("HF_HUB_CACHE", str(_LOCAL_MODELS / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_LOCAL_MODELS / "torch"))
# 国内访问 HuggingFace 镜像，避免联网超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 自动读取 Windows 系统代理（走 VPN 时无需手动配环境变量）
def _apply_windows_proxy():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if enabled:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                proxy = f"http://{server}"
                for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    os.environ.setdefault(var, proxy)
                # 有代理时直连 HuggingFace（镜像可能缺模型）
                os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    except Exception:
        pass
_apply_windows_proxy()

SAMPLE_RATE = 16000          # ASR / VAD 采样率
BLOCK_SIZE = 800             # 50ms @ 16kHz
SILENCE_TIMEOUT = 1.5        # 静音超过此秒数则结束录音
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

# 对话系统提示词
SYSTEM_PROMPT = """你是贾维斯(Jarvis)，一个智能语音助手。你有以下能力：
- 查询天气
- 查看当前时间
- 保存代码到文件
- 打开网页
- 打开本地应用（计算器、记事本、画图等）
- 控制窗口（最小化、关闭、切换等）
- 系统操作（锁屏、音量调节、截图等）

需要时直接使用对应的工具。用户的问题可能是语音识别结果，如有错别字请自动修正。
用中文回答，简洁自然。使用工具后请告知用户结果。"""


# ═══════════════════════════════════════════════════
#  1. VAD - 语音活动检测
# ═══════════════════════════════════════════════════
class VAD:
    """基于 Silero VAD 的语音活动检测（本地加载）"""

    def __init__(self):
        print("  [VAD] 加载模型中...")
        from silero_vad import load_silero_vad
        import torch
        torch.set_num_threads(1)
        self.model = load_silero_vad()
        self.model.eval()
        self._torch = torch

    def is_speech(self, audio: np.ndarray) -> bool:
        """返回 audio 是否包含人声"""
        with self._torch.no_grad():
            prob = self.model(self._torch.from_numpy(audio), SAMPLE_RATE).item()
        return prob > 0.5


# ═══════════════════════════════════════════════════
#  2. ASR - 语音识别
# ═══════════════════════════════════════════════════
class ASR:
    """基于 Paraformer-zh 的本地语音识别（中文准确率 1.95% CER）"""

    def __init__(self):
        import torch, os
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  [ASR] 加载 Paraformer-zh (device={device})...")
        # 强制离线，不联网检查更新
        os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
        # 本地缓存路径
        local_path = os.path.join(
            os.path.expanduser("~"), ".cache", "modelscope", "hub", "models",
            "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        )
        if not os.path.exists(local_path):
            local_path = "paraformer-zh"

        from funasr import AutoModel
        self.model = AutoModel(
            model=local_path,
            device=device,
            disable_update=True,
            hub="ms",
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """语音转文字，返回识别文本"""
        audio_float = audio.astype(np.float32)
        result = self.model.generate(input=audio_float)
        text = result[0]["text"] if result else ""
        text = text.replace(" ", "").strip()
        return text


# ═══════════════════════════════════════════════════
#  3. LLM - 大语言模型 (Ollama) — 原生工具调用
# ═══════════════════════════════════════════════════
class LLM:
    """通过 Ollama API 调用本地 LLM（支持 Ollama 原生 tool calling）"""

    def __init__(self, model=MODEL_NAME):
        self.model = model
        self.history = []
        self.max_turns = 10

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
                    "description": "打开 Windows 本地应用程序（自动搜索），如 微信(WeChat)、QQ、浏览器(chrome/edge)、Word、Excel、VSCode、计算器、记事本等",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app": {
                                "type": "string",
                                "description": "应用名称或关键词，如 微信、WeChat、QQ、QQ.exe、chrome、Word、VSCode、计算器、记事本 等"
                            }
                        },
                        "required": ["app"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "control_window",
                    "description": "控制已打开的窗口，如最小化、关闭、切换等",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "操作：minimize(最小化)、close(关闭窗口)、activate(切换/激活)、minimize_all(全部最小化)、show_desktop(显示桌面)"
                            },
                            "target": {
                                "type": "string",
                                "description": "目标窗口标题关键词（可选，如'浏览器'、'记事本'、'微信'）"
                            }
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "system_action",
                    "description": "系统操作：控制音量、锁屏、截图等",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "操作：lock(锁屏)、screenshot(截图)、volume_up(音量增)、volume_down(音量减)、mute(静音)、unmute(取消静音)、shutdown(关机)、restart(重启)、sleep(睡眠)"
                            },
                            "value": {
                                "type": "string",
                                "description": "参数（可选），如音量增/减数字"
                            }
                        },
                        "required": ["action"]
                    }
                }
            },
        ]

    # ── 工具实现 ──────────────────────────────────

    def _get_weather(self, city: str) -> str:
        try:
            resp = requests.get(
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
        """打开 Windows 本地应用程序（自动搜索已安装软件）"""
        import subprocess, os

        APP_MAP = {
            "微信": "WeChat.exe", "wechat": "WeChat.exe",
            "qq": "QQ.exe", "qq.exe": "QQ.exe",
            "qq音乐": "QQMusic.exe",
            "钉钉": "DingTalk.exe", "dingtalk": "DingTalk.exe",
            "飞书": "Feishu.exe", "feishu": "Feishu.exe",
            "腾讯会议": "wemeetapp.exe",
            "chrome": "chrome.exe", "谷歌": "chrome.exe", "谷歌浏览器": "chrome.exe",
            "edge": "msedge.exe", "浏览器": "msedge.exe", "microsoft edge": "msedge.exe",
            "火狐": "firefox.exe", "firefox": "firefox.exe",
            "vscode": "Code.exe", "vs code": "Code.exe", "visual studio code": "Code.exe",
            "word": "WINWORD.EXE",
            "excel": "EXCEL.EXE",
            "ppt": "POWERPNT.EXE", "powerpoint": "POWERPNT.EXE",
            "outlook": "OUTLOOK.EXE",
            "wps": "wps.exe", "wps office": "wps.exe",
            "steam": "steam.exe",
            "网易云": "cloudmusic.exe", "网易云音乐": "cloudmusic.exe",
            "百度网盘": "baidunetdisk.exe",
            "迅雷": "thunder.exe",
            "阿里云盘": "aDrive.exe",
            "剪映": "CapCut.exe", "剪映专业版": "CapCut.exe",
            "pycharm": "pycharm64.exe",
            "idea": "idea64.exe",
            "typora": "Typora.exe",
            "xmind": "Xmind.exe",
            "winrar": "WinRAR.exe",
            "git": "git-bash.exe", "git bash": "git-bash.exe",
            "计算器": "calc.exe", "calculator": "calc.exe",
            "记事本": "notepad.exe", "notepad": "notepad.exe",
            "画图": "mspaint.exe", "画图工具": "mspaint.exe",
            "命令提示符": "cmd.exe", "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "任务管理器": "taskmgr.exe", "task manager": "taskmgr.exe",
            "资源管理器": "explorer.exe", "文件资源管理器": "explorer.exe",
            "控制面板": "control",
            "截图工具": "snippingtool.exe",
            "便签": "stikynot.exe", "sticky notes": "stikynot.exe",
            "放大镜": "magnify.exe",
            "录音机": "voice recorder", "录音": "voice recorder",
            "时钟": "clock", "闹钟": "clock",
            "设置": "ms-settings:", "系统设置": "ms-settings:",
        }

        key = app.lower() if app.isascii() else app
        exe_name = APP_MAP.get(key)

        if exe_name:
            try:
                subprocess.Popen(f'cmd /c start "" "{exe_name}"', shell=True)
                return f"已打开: {app}"
            except:
                pass

        try:
            subprocess.Popen(f'cmd /c start "" "{app}"', shell=True)
            return f"已打开: {app}"
        except:
            pass

        search_dirs = [
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
            os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs"),
        ]
        for sd in search_dirs:
            if not os.path.exists(sd):
                continue
            for root, dirs, files in os.walk(sd):
                for f in files:
                    if f.endswith(".lnk") and (key in f.lower().replace(".lnk", "").replace(" ", "")):
                        try:
                            os.startfile(os.path.join(root, f))
                            return f"已打开: {app}"
                        except:
                            continue
                if len(dirs) > 100:
                    break

        return f"未找到: {app}"

    def _control_window(self, action: str, target: str = "") -> str:
        """控制已打开的窗口"""
        import subprocess
        try:
            if action in ("全部最小化", "minimize_all"):
                subprocess.Popen("powershell -Command (New-Object -ComObject Shell.Application).MinimizeAll()", shell=True)
                return "已全部最小化"
            if action in ("显示桌面", "show_desktop"):
                subprocess.Popen("powershell -Command (New-Object -ComObject Shell.Application).ToggleDesktop()", shell=True)
                return "已显示桌面"
            if not target:
                return f"已执行: {action}"
            try:
                import pygetwindow as gw
                wins = gw.getWindowsWithTitle(target)
                if not wins:
                    return f"未找到标题包含 '{target}' 的窗口"
                w = wins[0]
                act = action
                if act in ("最小化", "minimize"): w.minimize()
                elif act in ("最大化", "maximize"): w.maximize()
                elif act in ("关闭", "close", "关闭窗口"): w.close()
                elif act in ("切换", "激活", "activate", "switch"): w.activate()
                elif act in ("还原", "restore"): w.restore()
                else: return f"不支持的操作: {act}"
                return f"已{action}窗口: {target}"
            except ImportError:
                if action in ("关闭", "close", "关闭窗口"):
                    subprocess.run(f"powershell -Command \"Get-Process | Where-Object {{$_.MainWindowTitle -like '*{target}*'}} | Stop-Process\"", shell=True, timeout=5)
                    return f"已尝试关闭: {target}"
                return f"需要安装 pygetwindow 库来{action}窗口"
        except Exception as e:
            return f"窗口控制失败: {e}"

    def _system_action(self, action: str, value: str = "") -> str:
        """系统操作：音量、锁屏、截图等"""
        import subprocess
        try:
            if action == "锁屏" or action == "lock":
                subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True)
                return "已锁屏"
            if action == "截图" or action == "screenshot":
                subprocess.run("SnippingTool.exe", shell=True)
                return "已打开截图工具"
            if action == "静音" or action == "mute":
                subprocess.run("powershell -Command (New-Object -ComObject WScript.Shell).SendKeys([char]173)", shell=True)
                return "已静音"
            if action == "取消静音" or action == "unmute":
                subprocess.run("powershell -Command (New-Object -ComObject WScript.Shell).SendKeys([char]173)", shell=True)
                return "已取消静音"
            if "音量" in action or "vol" in action.lower():
                step = int(value) if value else 10
                key = "175" if ("增" in action or "up" in action.lower() or "大" in action) else "174"
                for _ in range(step // 2):
                    subprocess.run(f"powershell -Command (New-Object -ComObject WScript.Shell).SendKeys([char]{key})", shell=True, capture_output=True)
                return f"音量{'增加' if key=='175' else '减少'}{step}"
            if action == "关机": subprocess.run("shutdown /s /t 5", shell=True); return "将在5秒后关机"
            if action == "重启": subprocess.run("shutdown /r /t 5", shell=True); return "将在5秒后重启"
            if action == "睡眠": subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True); return "已进入睡眠"
            return f"未知系统操作: {action}"
        except Exception as e:
            return f"系统操作失败: {e}"

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

        if name == "control_window":
            action = args.get("action", "")
            target = args.get("target", "")
            if not action:
                return "错误: 需要提供操作"
            return self._control_window(action, target)

        if name == "system_action":
            action = args.get("action", "")
            value = args.get("value", "")
            if not action:
                return "错误: 需要提供操作"
            return self._system_action(action, value)

        return f"错误: 未知工具 '{name}'"

    # ── 对话 ──────────────────────────────────────

    def chat(self, user_text: str) -> str:
        import json, re

        self.history.append({"role": "user", "content": user_text})

        for _ in range(5):
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
                           + self.history[-self.max_turns * 2:],
                "stream": True,
                "tools": self._build_tools(),
            }

            # 请求（含 3 次重试，streaming）
            for attempt in range(3):
                try:
                    resp = requests.post(OLLAMA_URL, json=payload, timeout=120, stream=True)
                    resp.raise_for_status()
                    break
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
            else:
                return "抱歉，我与大脑的连接断开了，请检查 Ollama 是否在运行。"

            # 流式读取响应
            full_content = ""
            tool_calls = None
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {})
                    if delta.get("content"):
                        full_content += delta["content"]
                    if delta.get("tool_calls"):
                        tool_calls = delta.get("tool_calls")
                    if chunk.get("done"):
                        break

            reply = full_content

            # ── 主路径：Ollama 原生 tool_calls ──
            if tool_calls:
                assistant_msg = {"role": "assistant", "content": reply}
                assistant_msg["tool_calls"] = tool_calls
                self.history.append(assistant_msg)

                silent_tools = {"open_app", "control_window", "system_action"}
                last_name = ""
                last_result = ""

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
                    last_name = name
                    last_result = result_content

                # 操作类工具直接返回结果，不用 LLM 再废话
                if last_name in silent_tools:
                    return last_result
                continue

            # ── 回退 1a：检测原生格式 tool_call JSON ──
            native_match = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', reply)
            if native_match:
                try:
                    name = native_match.group(1)
                    args = json.loads(native_match.group(2))
                    result_content = self._execute_tool(name, args)
                    if name in {"open_app", "control_window", "system_action"}:
                        self.history.append({"role": "user", "content": result_content})
                        return result_content
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

            # 如果 reply 是纯 JSON 工具调用但仍未匹配，让 LLM 重试
            if reply.strip().startswith("{") and reply.strip().endswith("}"):
                self.history.append({"role": "user", "content": "请用中文直接回答，不要输出 JSON。"})
                continue

            self.history.append({"role": "assistant", "content": reply})
            return reply

        return "任务已完成。"


# ═══════════════════════════════════════════════════
#  4. TTS - 语音合成
# ═══════════════════════════════════════════════════
class TTS:
    """基于 Kokoro 的本地语音合成"""

    def __init__(self):
        print("  [TTS] 加载模型中...")
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code="z")

    def speak(self, text: str, block: bool = False):
        """合成并播放语音（异步播放，block=True 则阻塞等待播放完成）

        异步模式下，播放后台进行，可通过 sd.stop() 中断。
        """
        if not text:
            return

        generator = self.pipeline(text, voice="zf_xiaobei", speed=1.3)
        chunks = []
        for gs, ps, audio in generator:
            chunks.append(audio)

        if not chunks:
            return

        audio = np.concatenate(chunks)
        sd.play(audio, samplerate=24000)
        if block:
            sd.wait()


# ═══════════════════════════════════════════════════
#  5. 录音器 - 带 VAD 的语音采集
# ═══════════════════════════════════════════════════
class Recorder:
    """持续录音直到检测到静音结束"""

    def __init__(self, vad: VAD):
        self.vad = vad
        self.q = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    def record_until_silence(self) -> np.ndarray:
        """录制直到静音超时，返回完整音频 (16kHz float32)"""
        chunks = []
        silent_blocks = 0
        speak_started = False
        max_silent = int(SILENCE_TIMEOUT * SAMPLE_RATE / BLOCK_SIZE)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        ):
            while True:
                chunk = self.q.get()
                flat = chunk.flatten()
                speech = self.vad.is_speech(flat)

                if speech:
                    silent_blocks = 0
                    speak_started = True
                elif speak_started:
                    silent_blocks += 1
                else:
                    silent_blocks = 0

                if speak_started:
                    chunks.append(flat)

                if silent_blocks > max_silent:
                    break

        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ═══════════════════════════════════════════════════
#  Main - 主循环
# ═══════════════════════════════════════════════════
def main():
    print("=" * 45)
    print(" 贾维斯语音助手 — 全本地运行")
    print("=" * 45)
    print()
    print("正在加载模块（首次加载需下载模型）...")
    print()

    # 重要：先加载 ASR (CTranslate2)，再加载 VAD/Kokoro (PyTorch)
    # CTranslate2 必须在 PyTorch 之前初始化，否则 segfault
    asr = ASR()
    vad = VAD()
    llm = LLM()
    tts = TTS()
    recorder = Recorder(vad)

    print()
    print("✓ 贾维斯已就绪！")
    print("  - 按 Enter 开始说话")
    print("  - 静音 1.5 秒后自动识别")
    print("  - 输入 q 退出")
    print()

    while True:
        cmd = input("回车录音 / q退出 > ").strip().lower()
        if cmd == "q":
            sd.stop()  # 中断可能正在播放的语音
            tts.speak("贾维斯为您服务，再见。", block=True)
            break

        sd.stop()  # 中断上一轮可能还在异步播放的语音
        print("  [录音中... 说完自动识别]")
        audio = recorder.record_until_silence()

        if len(audio) < SAMPLE_RATE * 0.3:
            print("  (录音太短，已忽略)")
            continue

        print("  [识别中...]")
        text = asr.transcribe(audio)
        if not text:
            print("  (没听清，请再说一遍)")
            continue
        print(f"  你: {text}")

        print("  [思考中...]")
        reply = llm.chat(text)
        print(f"  贾维斯: {reply}")

        print("  [合成语音...]")
        tts.speak(reply)  # 异步播放，不阻塞，可由下次 Enter 中断
        print()


if __name__ == "__main__":
    main()
