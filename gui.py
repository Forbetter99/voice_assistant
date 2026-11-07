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
    """基于 Qwen3-ASR-1.7B 的语音识别（支持 22 种中文方言）"""

    def __init__(self):
        import os
        from pathlib import Path
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # 解析本地模型路径：优先项目缓存，其次用户默认 HF 缓存
        repo = "models--Qwen--Qwen3-ASR-1.7B"
        local = None
        for base in (_LOCAL_MODELS / "huggingface",
                     Path.home() / ".cache" / "huggingface" / "hub"):
            ref = base / repo / "refs" / "main"
            if ref.exists():
                snap = base / repo / "snapshots" / ref.read_text().strip()
                if snap.exists():
                    local = str(snap)
                    break
        import torch
        from qwen_asr import Qwen3ASRModel
        # ASR 用 8bit 量化 + GPU 加速：8bit 把显存从 fp16 的 ~4.9GB 降到 ~3GB，
        # 腾出的空间给 LLM 上 GPU（见 LLM.chat 里 options.num_gpu=24）。
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kwargs = {
                "dtype": torch.float16,
                "device_map": "cuda:0",
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            }
        else:
            kwargs = {}
        self.model = Qwen3ASRModel.from_pretrained(local or "Qwen/Qwen3-ASR-1.7B", **kwargs)

    def transcribe(self, audio: np.ndarray) -> str:
        audio_float = audio.astype(np.float32)
        results = self.model.transcribe(
            (audio_float, 16000), language="Chinese"
        )
        return results[0].text.strip() if results else ""


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
            "你有以下工具可用：查天气、查时间、保存代码、打开网页、网页搜索(web_search，"
            "支持B站/百度/谷歌/知乎等，B站可 play=true 直接播放第一个视频)、"
            "打开本地应用(计算器/记事本等)、播放音乐(QQ音乐)、媒体控制(暂停/下一首)、"
            "控制窗口(最小化/关闭/切换)、系统操作(锁屏/音量/截图等)。"
            "需要时直接使用对应的工具。"
            "注意：用户每次要求打开/启动某应用时，都必须重新调用 open_app 工具，"
            "不要因为历史里已经打开过就跳过。"
            "用户的问题可能是语音识别结果，如有错别字请自动修正。"
            "用中文回答，简洁自然。使用工具后请告知用户结果。"
        )

    # 中文 → exe 名映射（Windows 注册的是 exe 名，不是中文名；微信 4.0 是 Weixin.exe）
    APP_MAP = {
        "微信": "Weixin.exe", "wechat": "Weixin.exe", "weixin": "Weixin.exe",
        "qq": "QQ.exe", "qq.exe": "QQ.exe",
        "qq音乐": "QQMusic.exe", "qqmusic": "QQMusic.exe",
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

    # 关闭时进程名可能与“打开用 exe 名”不同（UWP 应用尤其如此），这里单独覆盖（不带 .exe）
    CLOSE_OVERRIDE = {
        "计算器": "CalculatorApp", "calculator": "CalculatorApp",
    }

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
                    "name": "web_search",
                    "description": "在网页上搜索。可指定网站：bilibili(B站)、百度、谷歌、知乎、微博、淘宝、YouTube、豆瓣。"
                                   "play=true 时对 B站 会直接打开搜索到的第一个视频播放，例如“用网页打开B站搜索发如雪并播放”",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词，如 发如雪"
                            },
                            "site": {
                                "type": "string",
                                "description": "网站，默认 bilibili。可选：bilibili、百度、谷歌、知乎、微博、淘宝、youtube、豆瓣"
                            },
                            "play": {
                                "type": "boolean",
                                "description": "是否直接播放第一个视频结果（仅 B站 有效）"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "打开 Windows 本地应用程序（自动搜索，支持任何已安装软件），如 微信(WeChat)、QQ、浏览器(chrome/edge)、Word、Excel、VSCode、计算器、记事本等",
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
                    "description": "系统操作：控制音量、锁屏、关机、截图等",
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
            {
                "type": "function",
                "function": {
                    "name": "play_music",
                    "description": "在音乐软件里搜索并播放指定歌曲，例如“播放晴天”。会自动打开 QQ音乐 并播放第一首结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "song": {
                                "type": "string",
                                "description": "歌曲名（可带歌手），如 晴天、周杰伦的晴天"
                            },
                            "app": {
                                "type": "string",
                                "description": "音乐软件，默认 qq音乐"
                            }
                        },
                        "required": ["song"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "media_control",
                    "description": "控制音乐播放：暂停/继续播放、上一首、下一首",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "操作：播放暂停(play)、暂停(pause)、继续(play)、上一首(prev)、下一首(next)"
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

    # ── 网页搜索 / B站播放 ─────────────────────────

    WEB_SITES = {
        "bilibili": ["bilibili", "b站", "哔哩哔哩", "哔哩", "小破站"],
        "baidu": ["百度"],
        "google": ["谷歌", "google"],
        "youtube": ["youtube", "油管"],
        "zhihu": ["知乎"],
        "weibo": ["微博"],
        "taobao": ["淘宝"],
        "douban": ["豆瓣"],
    }

    @staticmethod
    def _strip_em(text: str) -> str:
        """去掉 B站 搜索接口返回标题里的 <em> 高亮标签。"""
        import re
        return re.sub(r"</?em[^>]*>", "", text or "").strip()

    def _bili_search(self, query: str, play: bool = False) -> str:
        """B站：只搜索→打开搜索页；play=True→取第一个视频直接打开并播放。"""
        import urllib.parse
        search_url = f"https://search.bilibili.com/all?keyword={urllib.parse.quote(query)}"
        if not play:
            self._open_browser(search_url)
            return f"已在浏览器打开 B站搜索「{query}」"

        try:
            import requests
            h = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
                "Referer": "https://www.bilibili.com/",
            }
            s = requests.Session()
            s.get("https://www.bilibili.com/", headers=h, timeout=8)  # 拿 buvid3 等 cookie
            r = s.get("https://api.bilibili.com/x/web-interface/search/type",
                      params={"search_type": "video", "keyword": query, "page": 1},
                      headers={**h, "Referer": "https://search.bilibili.com/"}, timeout=8)
            data = r.json()
            result = (data.get("data") or {}).get("result") or []
            if data.get("code") == 0 and result:
                first = result[0]
                bvid = first.get("bvid")
                title = self._strip_em(first.get("title", ""))
                author = first.get("author", "")
                if bvid:
                    self._open_browser(f"https://www.bilibili.com/video/{bvid}/")
                    return f"已在浏览器播放 B站视频《{title}》（{author}）"
        except Exception:
            pass

        # 接口失败（断网/风控）就退回搜索页
        self._open_browser(search_url)
        return f"没拿到 B站搜索结果，已打开搜索页「{query}」"

    def _open_site_search(self, query: str, site: str = "bilibili", play: bool = False) -> str:
        """在指定网站搜索；B站支持直接播放第一个视频。"""
        import urllib.parse
        query = (query or "").strip()
        if not query:
            return "请告诉我搜索什么"
        site = (site or "bilibili").strip().lower()
        # 中文站点名映射到 key
        for key, aliases in self.WEB_SITES.items():
            if site == key or site in aliases:
                site = key
                break
        if site == "bilibili":
            return self._bili_search(query, play)
        q = urllib.parse.quote(query)
        urls = {
            "baidu": f"https://www.baidu.com/s?wd={q}",
            "google": f"https://www.google.com/search?q={q}",
            "youtube": f"https://www.youtube.com/results?search_query={q}",
            "zhihu": f"https://www.zhihu.com/search?type=content&q={q}",
            "weibo": f"https://s.weibo.com/weibo?q={q}",
            "taobao": f"https://s.taobao.com/search?q={q}",
            "douban": f"https://www.douban.com/search?q={q}",
        }
        url = urls.get(site)
        if not url:
            return f"不支持的网站: {site}"
        self._open_browser(url)
        return f"已在浏览器搜索「{query}」（{site}）"

    def _fast_web(self, text: str):
        """识别“用网页/在B站搜索X并播放”这类指令，返回 (站点, 关键词, 是否播放) 或 None。"""
        import re
        if not text:
            return None
        tl = text.lower()

        site = None
        for key, aliases in self.WEB_SITES.items():
            if any(a in tl for a in aliases):
                site = key
                break
        if not site:
            return None

        # 必须是“网页语境”：搜索类动词 / “百度一下” / “B站播放X” 这类直说
        verb = re.search(r"(搜索|搜一下|搜一搜|搜搜|搜|查一下|查查|查找|查询|找一下|找找|找)\s*[:：]?\s*(.+)", text)
        one_xia = re.search(r"(?:百度|谷歌|知乎|b站|bilibili|哔哩哔哩)\s*(?:一下|下)\s*(.+)", tl)
        direct = re.search(r"(?:播放|看|放)\s*(.+)", text)
        play = bool(re.search(r"播放|放一下|放放|听一下|看一下|打开看", tl))

        if verb:
            query = verb.group(2)
        elif one_xia:
            query = text[one_xia.start(1):]
        elif direct and site in ("bilibili", "youtube"):
            # “打开B站播放发如雪”“在B站看发如雪”
            query = direct.group(1)
            play = True
        else:
            return None
        # 去掉尾部动作词和站点名残留
        query = re.sub(r"[，,。\s]*(?:并|然后|再|并且)?\s*(?:播放|放一下|放放|听一下|看一下|打开看|打开)\s*$", "", query)
        for a in self.WEB_SITES[site]:
            query = re.sub(re.escape(a), "", query, flags=re.I)
        query = query.strip(" 的了吧啊呀，,。.！!？?《》\"'“”")
        if not query:
            return None
        return site, query, play

    def _fast_open_app(self, text: str):
        """识别“打开/启动/开启 <已知应用>”命令；命中返回应用名，否则 None。

        在 chat() 里绕过 LLM 直接打开应用：更快，且避免 LLM 因历史里已“打开过”
        而第二次不再调用工具（“关闭后再次打开失败”的根因）。
        """
        import re
        if not re.search(r"(打开|启动|开启|运行)", text or ""):
            return None
        # 别名从长到短匹配，避免“腾讯会议”被“会议”类短路、以及“qq音乐”被“qq”抢先；
        # ASCII 忽略大小写（“打开QQ”要能命中“qq”）
        tl = (text or "").lower()
        for name in sorted(self.APP_MAP.keys(), key=len, reverse=True):
            if name and name.lower() in tl:
                return name
        return None

    def _fast_close_app(self, text: str):
        """识别“关闭/退出/结束 <已知应用>”命令；命中返回应用名，否则 None。"""
        import re
        if not re.search(r"(关闭|退出|结束|关掉)", text or ""):
            return None
        tl = (text or "").lower()
        names = sorted(set(self.APP_MAP.keys()) | set(self.CLOSE_OVERRIDE.keys()), key=len, reverse=True)
        for name in names:
            if name and name.lower() in tl:
                return name
        return None

    # ── 音乐播放（QQ音乐 / 网易云）──────────────────

    @staticmethod
    def _fg_pid() -> int:
        """当前前台窗口所属进程 PID。"""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(ctypes.c_ulong)]
        hwnd = user32.GetForegroundWindow()
        pid = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value

    @classmethod
    def _force_foreground(cls, hwnd, pid: int = 0) -> bool:
        """把窗口强制拉到前台并校验：前台窗口是它，或（给了 pid 时）前台属于同一进程的弹窗。"""
        import ctypes, time
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE（顺带解除最小化）
        if user32.GetForegroundWindow() == hwnd:
            return True
        fg = user32.GetForegroundWindow()
        cur_tid = kernel32.GetCurrentThreadId()
        fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        if fg_tid:
            user32.AttachThreadInput(cur_tid, fg_tid, True)
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if fg_tid:
                user32.AttachThreadInput(cur_tid, fg_tid, False)
        time.sleep(0.4)
        if user32.GetForegroundWindow() == hwnd:
            return True
        # 同一进程的弹窗也算“已在前台”
        return bool(pid) and cls._fg_pid() == pid

    @staticmethod
    def _pids_of(proc_name: str) -> set:
        import subprocess
        try:
            out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {proc_name}", "/FO", "CSV"],
                                 capture_output=True, timeout=10).stdout.decode("gbk", "ignore")
            pids = set()
            for line in out.splitlines()[1:]:
                parts = line.split('","')
                if len(parts) > 1 and parts[1].strip('"').isdigit():
                    pids.add(int(parts[1].strip('"')))
            return pids
        except Exception:
            return set()

    def _qq_hwnd(self):
        """返回 (hwnd, pid)：QQ音乐主窗口（面积最大、可见的 TXGuiFoundation）。"""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
                                       wintypes.LPARAM]
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(ctypes.c_ulong)]

        pids = self._pids_of("QQMusic.exe")
        if not pids:
            return None, 0
        best = [None, 0, 0]

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(h, _):
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(h, cls, 256)
            if cls.value != "TXGuiFoundation" or not user32.IsWindowVisible(h):
                return True
            pid = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
            if pid.value not in pids:
                return True
            r = wintypes.RECT()
            user32.GetWindowRect(h, ctypes.byref(r))
            area = (r.right - r.left) * (r.bottom - r.top)
            if area > best[1]:
                best[0], best[1], best[2] = h, area, pid.value
            return True

        user32.EnumWindows(cb, 0)
        return best[0], best[2]

    def _qq_window(self):
        """返回 QQ音乐主窗口（面积最大的 TXGuiFoundation 且进程属于 QQMusic.exe）。"""
        try:
            import uiautomation as auto
        except ImportError:
            return None
        auto.InitializeUIAutomationInCurrentThread()
        hwnd, _pid = self._qq_hwnd()
        if not hwnd:
            return None
        try:
            return auto.ControlFromHandle(hwnd)
        except Exception:
            return None

    def _find_music_exe(self) -> str:
        """定位 QQMusic.exe 的绝对路径。"""
        import os, shutil
        # 已知路径
        for p in (r"D:\QQMusic\QQMusic.exe", r"C:\QQMusic\QQMusic.exe"):
            if os.path.exists(p):
                return p
        # PATH
        p = shutil.which("QQMusic.exe")
        if p:
            return p
        # 开始菜单快捷方式解析
        import subprocess
        for sd in (os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
                   os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs")):
            if not os.path.isdir(sd):
                continue
            for root, _dirs, files in os.walk(sd):
                for f in files:
                    if f.lower().endswith(".lnk") and "qq音乐" in f:
                        lnk = os.path.join(root, f)
                        ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut('"
                              + lnk.replace("'", "''") + "'); $s.TargetPath")
                        try:
                            t = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                               capture_output=True, timeout=15).stdout.decode("gbk", "ignore").strip()
                            if t and os.path.exists(t):
                                return t
                        except Exception:
                            pass
        # 安装目录扫描
        for base in (r"C:\Program Files", r"C:\Program Files (x86)", "D:\\", "E:\\"):
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                if root[len(base):].count(os.sep) > 3:
                    dirs[:] = []
                    continue
                if "QQMusic.exe" in files:
                    return os.path.join(root, "QQMusic.exe")
        return ""

    @staticmethod
    def _title_matches(title: str, song: str) -> bool:
        """判断窗口标题（当前播放歌曲）是否是目标歌曲。"""
        import re
        t = (title or "").lower()
        core = re.sub(r"(的|这首歌|歌曲|音乐|qq音乐|网易云|来一首|放一首)", "", (song or "").lower()).strip()
        if not core:
            return False
        for n in (len(core), 5, 4, 3, 2):
            if n >= 2 and core[:n] in t:
                return True
        return False

    @staticmethod
    def _double_click(x: int, y: int):
        """Win32 双击（比 uiautomation 的两次 Click 更稳）。"""
        import ctypes, time
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        user32.mouse_event.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                       wintypes.DWORD, wintypes.ULONG]
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.15)
        for _ in range(2):
            user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
            time.sleep(0.03)
            user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
            time.sleep(0.08)

    def _play_music_qq(self, song: str) -> str:
        """在 QQ音乐 里搜索并播放：UIA 定位搜索框 → 粘贴 → 回车 → 按 UIA 行列坐标双击第一条。"""
        import time, os
        try:
            import uiautomation as auto
        except ImportError:
            return "缺少 uiautomation 库，无法操作 QQ音乐（pip install uiautomation）"
        auto.InitializeUIAutomationInCurrentThread()

        hwnd, pid = self._qq_hwnd()
        if not hwnd:
            exe = self._find_music_exe()
            if not exe:
                return "没找到 QQ音乐，请确认已安装"
            try:
                os.startfile(exe)
            except Exception as e:
                return f"启动 QQ音乐失败: {e}"
            for _ in range(20):
                time.sleep(1)
                hwnd, pid = self._qq_hwnd()
                if hwnd:
                    break
            if not hwnd:
                return "QQ音乐启动超时"

        if not self._force_foreground(hwnd, pid):
            return "QQ音乐没有成功切到前台，已取消操作（避免误点其他窗口）"
        time.sleep(0.5)
        # 先关掉可能存在的弹窗（会员/更新提示等）
        auto.SendKeys("{Esc}", waitTime=0.15)
        auto.SendKeys("{Esc}", waitTime=0.15)

        win = auto.ControlFromHandle(hwnd)
        r = win.BoundingRectangle

        # 2) 在结果列表里找出每一行（“时长：mm:ss”单元格），按纵向排序
        def scan_rows():
            rows, header_x = [], [None]

            def walk(c, depth=0):
                if depth > 16:
                    return
                for ch in c.GetChildren():
                    try:
                        nm = (ch.Name or "").strip()
                        br = ch.BoundingRectangle
                        if nm and br.width() > 0:
                            if nm.startswith("时长：") or nm.startswith("时长:"):
                                if (br.left > r.left + r.width() * 0.5
                                        and br.top > r.top + r.height() * 0.3):
                                    rows.append((br.top, nm))
                            elif nm in ("歌名/歌手", "歌曲标题") and header_x[0] is None:
                                header_x[0] = br.left
                    except Exception:
                        pass
                    walk(ch, depth + 1)

            walk(win)
            rows.sort()
            return rows, header_x[0]

        # 记录旧结果，用于判断搜索结果是否已经刷新
        before_rows, _ = scan_rows()
        before = tuple(nm for _t, nm in before_rows)

        # 1) UIA 找到搜索框实体，点它的中心（不再用屏幕比例估算）
        ex = ey = None
        try:
            edit = win.EditControl(searchDepth=16)
            er = edit.BoundingRectangle
            if er.width() > 0 and er.height() > 0:
                ex, ey = (er.left + er.right) // 2, (er.top + er.bottom) // 2
        except Exception:
            edit = None
        if ex is None:
            ex, ey = r.left + int(r.width() * 0.428), r.top + int(r.height() * 0.047)
        auto.Click(ex, ey, waitTime=0.3)
        time.sleep(0.3)
        auto.SendKeys("{Ctrl}a", waitTime=0.1)
        auto.SetClipboardText(song)
        time.sleep(0.15)
        auto.SendKeys("{Ctrl}v", waitTime=0.3)
        time.sleep(0.9)
        auto.SendKeys("{Enter}", waitTime=0.2)

        # 等结果刷新：行的“时长”内容变了才算新结果（最多 8 秒）
        rows, header_x = [], None
        for _ in range(20):
            time.sleep(0.4)
            rows, header_x = scan_rows()
            if rows and tuple(nm for _t, nm in rows) != before:
                break
        time.sleep(0.4)
        rows, header_x = scan_rows()

        # 歌曲名列：从“歌名/歌手”表头往右偏一点
        x = (header_x + int(r.width() * 0.10)) if header_x else (r.left + int(r.width() * 0.317))

        for top, _nm in rows[:3]:
            if not self._force_foreground(hwnd, pid):
                break
            self._double_click(x, top + 10)
            time.sleep(2.5)
            w2 = self._qq_window()
            title = (w2.Name if w2 else "") or ""
            if self._title_matches(title, song):
                return f"正在播放: {title.strip()}"

        # 3) 兜底：老的比例坐标
        for fy in (0.487, 0.44, 0.53):
            if not self._force_foreground(hwnd, pid):
                break
            self._double_click(r.left + int(r.width() * 0.45), r.top + int(r.height() * fy))
            time.sleep(2.5)
            w2 = self._qq_window()
            title = (w2.Name if w2 else "") or ""
            if self._title_matches(title, song):
                return f"正在播放: {title.strip()}"

        return f"已在 QQ音乐 搜索「{song}」，但没能自动点播放，请手动点一下第一首"

    def _play_music(self, song: str, app: str = "qq音乐") -> str:
        song = (song or "").strip().strip("《》\"'“”")
        if not song:
            return "请告诉我歌名"
        app = (app or "").lower()
        if "网易" in app or "cloud" in app or "netease" in app:
            return f"暂不支持自动播放网易云音乐，已为你打开 QQ音乐播放「{song}」\n" + self._play_music_qq(song)
        return self._play_music_qq(song)

    def _fast_play_music(self, text: str):
        """识别“播放<歌名>”命令，返回歌名；不是播放指令则返回 None。"""
        import re
        if not text:
            return None
        # 注意不要用裸“放”，否则“别放了”会被当成播放歌曲“了”
        m = re.search(r"(?:播放|放一首|放一下|放首|来一首|来首|点播|我想听|想听)\s*(.+)", text)
        if not m:
            return None
        song = m.group(1)
        for junk in ("qq音乐", "QQ音乐", "网易云音乐", "网易云", "酷狗音乐", "酷我音乐",
                     "上面的", "这首歌", "一首歌曲", "歌曲", "的音乐", "音乐", "在", "用"):
            song = song.replace(junk, " ")
        song = song.strip(" \t，,。.！!？?《》\"'“”")
        if not song or song in ("继续", "一下", "吧", "啊"):
            return None
        return song

    def _fast_media_control(self, text: str):
        """识别纯媒体控制指令（不带歌名时），如“暂停”“下一首”。"""
        t = (text or "").strip().rstrip("。.!！?？")
        if t in ("暂停", "暂停播放", "暂停一下", "停一下", "别放了", "停"):
            return "暂停"
        if t in ("继续", "继续播放", "恢复", "恢复播放", "接着放", "继续放", "播放"):
            return "继续"
        if t in ("下一首", "下一首歌", "切歌", "换一首", "下一曲", "下一个"):
            return "下一首"
        if t in ("上一首", "上一首歌", "上一曲", "上一个"):
            return "上一首"
        return None

    def _media_control(self, action: str) -> str:
        """系统媒体键：播放/暂停、上一首、下一首。"""
        import ctypes, time
        VK = {"播放": 0xB3, "暂停": 0xB3, "播放暂停": 0xB3, "继续": 0xB3, "play": 0xB3,
              "下一首": 0xB0, "next": 0xB0, "上一首": 0xB1, "prev": 0xB1}
        vk = VK.get((action or "").lower())
        if vk is None:
            return f"不支持的媒体操作: {action}"
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(vk, 0, 2, 0)
        return f"已执行: {action}"

    def _open_app(self, app: str) -> str:
        """打开 Windows 本地应用程序（自动搜索已安装软件）"""
        import subprocess, os

        # 先在映射表里找（映射表见类属性 APP_MAP）
        APP_MAP = self.APP_MAP
        app = (app or "").strip()
        key = app.lower() if app.isascii() else app
        exe_name = APP_MAP.get(key)
        import shutil

        def _start(target: str) -> bool:
            # os.startfile 走 ShellExecute：目标不存在/打不开会抛异常，能可靠判断成败
            try:
                os.startfile(target)
                return True
            except Exception:
                return False

        def _resolve(name: str):
            # PATH（系统自带应用在这里，如 calc.exe / notepad.exe）
            p = shutil.which(name)
            if p:
                return p
            # App Paths 注册表（绝大多数第三方应用都注册了）
            exe = name if name.lower().endswith(".exe") else name + ".exe"
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for sub in (r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\\",
                            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\\"):
                    try:
                        k = winreg.OpenKey(hive, sub + exe)
                        v, _ = winreg.QueryValueEx(k, None)
                        winreg.CloseKey(k)
                        if v and os.path.exists(v):
                            return v
                    except OSError:
                        continue
            return None

        def _keywords():
            kws = set()
            for c in (key, app, exe_name):
                if c:
                    kw = c.lower().replace(" ", "").replace(".exe", "")
                    if kw:
                        kws.add(kw)
            # 把 exe 名反查回所有中文别名，传英文名也能命中中文快捷方式
            if exe_name:
                for _k, _v in APP_MAP.items():
                    if _v == exe_name:
                        kws.add(_k.lower().replace(" ", "").replace(".exe", ""))
            return kws

        # 1) 协议 / URI（ms-settings:、shell: 等）
        if exe_name and exe_name.startswith(("ms-", "shell:")):
            try:
                subprocess.run(f'cmd /c start "" "{exe_name}"', shell=True, timeout=10)
                return f"已打开: {app}"
            except Exception:
                pass

        # 2) 解析到绝对路径（PATH 或 App Paths）后打开
        for name in filter(None, (exe_name, app)):
            path = _resolve(name)
            if path and _start(path):
                return f"已打开: {app}"

        # 3) 直接给的是完整路径
        if os.path.exists(app) and _start(app):
            return f"已打开: {app}"

        # 4) 搜索开始菜单 .lnk（覆盖绝大多数已安装桌面软件）
        kws = _keywords()
        _SKIP_MARK = ("卸载", "uninstall", "移除", "remove", "修复", "repair", "官网", "website", "帮助", "help")
        lnk_dirs = [
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
            os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs"),
        ]
        best = None  # (score, path)，精确匹配优先
        for sd in lnk_dirs:
            if not os.path.isdir(sd):
                continue
            for root, _dirs, files in os.walk(sd):
                for f in files:
                    if not f.lower().endswith(".lnk"):
                        continue
                    lnk_name = os.path.splitext(f)[0].lower().replace(" ", "")
                    if any(m in lnk_name for m in _SKIP_MARK):
                        continue
                    for kw in kws:
                        if not kw or kw not in lnk_name:
                            continue
                        score = 0 if lnk_name == kw else (1 if lnk_name.startswith(kw) else 2)
                        if best is None or score < best[0]:
                            best = (score, os.path.join(root, f))
        if best is not None and _start(best[1]):
            return f"已打开: {app}"

        # 5) 搜索常见安装目录里的同名 exe
        exe_names = set()
        for c in filter(None, (exe_name, app, key)):
            c = c.lower().strip()
            if not c:
                continue
            exe_names.add(c if c.endswith(".exe") else c + ".exe")
        install_dirs = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs"),
        ]
        for base in install_dirs:
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                depth = root[len(base):].count(os.sep)
                if depth >= 4:
                    dirs[:] = []
                for f in files:
                    if f.lower() in exe_names:
                        if _start(os.path.join(root, f)):
                            return f"已打开: {app}"

        return f"未找到: {app}"

    def _close_app(self, app: str) -> str:
        """关闭运行中的应用：优先按进程名 taskkill（优雅→强制），兜底按窗口标题。"""
        import subprocess, time

        app = (app or "").strip()
        key = app.lower() if app.isascii() else app

        # 进程名候选：UWP 覆盖表优先，其次 APP_MAP 的 exe 名去 .exe
        procs = []
        if key in self.CLOSE_OVERRIDE:
            procs.append(self.CLOSE_OVERRIDE[key])
        exe = self.APP_MAP.get(key)
        if exe and exe.lower().endswith(".exe"):
            base = exe[:-4]
            if base and base.lower() not in [p.lower() for p in procs]:
                procs.append(base)

        if not procs:
            # 无已知进程名（如“设置”“控制面板”等）→ 退回窗口标题方式
            return self._control_window("close", app)

        def _running(proc: str) -> bool:
            try:
                out = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {proc}.exe", "/FO", "CSV"],
                    capture_output=True, timeout=10,
                ).stdout.decode("gbk", errors="ignore")
                return proc.lower() in out.lower()
            except Exception:
                return False

        for proc in procs:
            if not _running(proc):
                continue
            subprocess.run(["taskkill", "/IM", f"{proc}.exe"], capture_output=True, timeout=10)
            time.sleep(1.2)
            if _running(proc):
                subprocess.run(["taskkill", "/F", "/IM", f"{proc}.exe"], capture_output=True, timeout=10)
            return f"已关闭: {app}"
        return f"未找到运行中的: {app}"

    def _control_window(self, action: str, target: str = "") -> str:
        """控制已打开的窗口"""
        import subprocess, time
        try:
            # 全局操作
            if action in ("全部最小化", "minimize_all"):
                subprocess.Popen("powershell -Command (New-Object -ComObject Shell.Application).MinimizeAll()", shell=True)
                return "已全部最小化"
            if action in ("显示桌面", "show_desktop"):
                subprocess.Popen("powershell -Command (New-Object -ComObject Shell.Application).ToggleDesktop()", shell=True)
                return "已显示桌面"
            if not target:
                return f"已执行: {action}"

            # 用 pygetwindow 控制指定窗口
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
            # ── 锁屏 ──
            if action == "锁屏" or action == "lock":
                subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True)
                return "已锁屏"

            # ── 截图 ──
            if action == "截图" or action == "screenshot":
                subprocess.run("SnippingTool.exe", shell=True)
                return "已打开截图工具"

            # ── 音量控制 ──
            if action == "静音" or action == "mute":
                subprocess.run("nircmd muteappvolume sound 1", shell=True, capture_output=True) or \
                subprocess.run("powershell -Command (New-Object -ComObject WScript.Shell).SendKeys([char]173)", shell=True)
                return "已静音"

            if action == "取消静音" or action == "unmute":
                subprocess.run("nircmd muteappvolume sound 0", shell=True, capture_output=True) or \
                subprocess.run("powershell -Command (New-Object -ComObject WScript.Shell).SendKeys([char]173)", shell=True)
                return "已取消静音"

            if "音量" in action or "vol" in action.lower():
                # 音量增/减 + 数值
                step = 10
                if value:
                    try:
                        step = int(value)
                    except ValueError:
                        pass

                if "增" in action or "up" in action.lower() or "大" in action:
                    for _ in range(step // 2):
                        subprocess.run("powershell -Command (New-Object -ComObject WScript.Shell).SendKeys([char]175)", shell=True, capture_output=True)
                    return f"音量增加{step}"

                if "减" in action or "down" in action.lower() or "小" in action:
                    for _ in range(step // 2):
                        subprocess.run("powershell -Command (New-Object -ComObject WScript.Shell).SendKeys([char]174)", shell=True, capture_output=True)
                    return f"音量减少{step}"

                # 设置具体音量值
                try:
                    target_vol = int(action.replace("音量", "").strip())
                    subprocess.run(f"powershell -Command \"$k=[Math]::Round({target_vol}/100*65535); (New-Object -ComObject WScript.Shell).SendKeys([char]173)\"", shell=True, capture_output=True)
                    return f"音量已设置为{target_vol}"
                except:
                    pass

            # ── 关机/重启/睡眠 ──
            if action == "关机" or action == "shutdown":
                subprocess.run("shutdown /s /t 5", shell=True)
                return "将在5秒后关机"

            if action == "重启" or action == "restart":
                subprocess.run("shutdown /r /t 5", shell=True)
                return "将在5秒后重启"

            if action == "睡眠" or action == "sleep":
                subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
                return "已进入睡眠"

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

        if name == "web_search":
            query = args.get("query", "")
            if not query:
                return "错误: 需要提供搜索关键词"
            return self._open_site_search(query, args.get("site", "bilibili"),
                                          bool(args.get("play", False)))

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

        if name == "play_music":
            song = args.get("song", "")
            if not song:
                return "错误: 需要提供歌名"
            return self._play_music(song, args.get("app", "qq音乐"))

        if name == "media_control":
            action = args.get("action", "")
            if not action:
                return "错误: 需要提供操作"
            return self._media_control(action)

        return f"错误: 未知工具 '{name}'"

    # ── 对话 ──────────────────────────────────────

    def chat(self, user_text: str) -> str:
        import json, re
        self.history.append({"role": "user", "content": user_text})

        # 快速通道：网页搜索（“用网页打开B站搜索发如雪并播放”）。
        # 必须早于“播放歌曲”和“打开应用”，否则会被当成 QQ音乐 播放 / 打开浏览器。
        web = self._fast_web(user_text)
        if web:
            site, query, play = web
            result = self._open_site_search(query, site, play)
            self.history.append({"role": "assistant", "content": result})
            return result

        # 快速通道：播放歌曲（必须早于“打开应用”，否则“打开qq音乐播放晴天”只会打开音乐软件）
        song = self._fast_play_music(user_text)
        if song:
            result = self._play_music(song)
            self.history.append({"role": "assistant", "content": result})
            return result

        # 快速通道：媒体控制（暂停/继续/上一首/下一首）
        media = self._fast_media_control(user_text)
        if media:
            result = self._media_control(media)
            self.history.append({"role": "assistant", "content": result})
            return result

        # 快速通道：明确的“打开某应用”命令直接执行，跳过 LLM。
        # 既省一次 LLM 往返，又规避“历史里已打开过 → LLM 不再调用工具”的问题。
        fast_app = self._fast_open_app(user_text)
        if fast_app:
            result = self._open_app(fast_app)
            self.history.append({"role": "assistant", "content": result})
            return result

        # 快速通道：明确的“关闭/退出某应用”命令直接执行，跳过 LLM。
        close_app = self._fast_close_app(user_text)
        if close_app:
            result = self._close_app(close_app)
            self.history.append({"role": "assistant", "content": result})
            return result

        for _ in range(5):
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_prompt}]
                           + self.history[-self.max_turns * 2:],
                "stream": True,
                "tools": self._build_tools(),
                # LLM 上 GPU：7b 共 29 层，卸 24 层到显存（ASR 已改 8bit 腾出空间），
                # 实测 ~28 tok/s（CPU 只有 ~9 tok/s）；留 5 层 + KV 缓存给 6GB 余量，避免 OOM
                "options": {"num_gpu": 24},
            }

            # 请求（含 3 次重试，streaming）
            for attempt in range(3):
                try:
                    resp = self._requests.post(
                        "http://localhost:11434/api/chat", json=payload, timeout=120, stream=True
                    )
                    resp.raise_for_status()
                    break
                except self._requests.exceptions.ConnectionError:
                    time.sleep(1)
            else:
                return "抱歉，Ollama 连接失败，请检查是否已启动。"

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

                silent_tools = {"open_app", "control_window", "system_action", "get_time", "get_weather"}
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
        generator = self.pipeline(text, voice="zf_xiaobei", speed=1.3)
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
        ("加载 ASR (Qwen3-ASR-1.7B) 语音识别模型...", 25),
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
        self._stop_speech = threading.Event()  # 用于中断 TTS 播放
        self._user_interrupted = False  # 用于记录是否真是用户打断

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
        self._stop_speech.set()
        sd.stop()
        self.log("⏹️ 中断当前对话")

    # ── TTS 播放中插嘴打断 ──────────────────────

    def _play_tts_with_interrupt(self, text: str) -> bool:
        """流式播放 TTS（生成一段播一段）+ 后台监听打断词，返回 True=用户打断"""
        if not text or not self.tts or not self.tts.pipeline:
            return False

        self._stop_speech.clear()
        self._user_interrupted = False

        # 启动打断监听线程
        listener_thread = threading.Thread(target=self._interrupt_listener_loop, daemon=True)
        listener_thread.start()

        # 流式合成+播放：生成一段立即播放，不等全部生成
        generator = self.tts.pipeline(text, voice="zf_xiaobei", speed=1.3)
        for gs, ps, audio in generator:
            if self._stop_speech.is_set():
                sd.stop()
                break

            # Kokoro 返回 torch tensor，转 numpy
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()

            # 增益放大
            peak = np.max(np.abs(audio))
            if peak > 0 and peak < 0.6:
                audio = audio * (0.7 / peak)
            audio = np.clip(audio, -1.0, 1.0)

            sd.play(audio, samplerate=24000)
            # 轮询等待播放结束（可被打断）
            while sd.get_stream() and sd.get_stream().active:
                if self._stop_speech.is_set():
                    sd.stop()
                    break
                time.sleep(0.05)

        self._stop_speech.set()
        listener_thread.join(timeout=1.0)

        return self._user_interrupted

    def _interrupt_listener_loop(self):
        """TTS 播放时后台监听打断词：结束/停止/闭嘴"""
        import numpy as np
        stop_q = queue.Queue()
        chunks = []
        speech_active = False

        def callback(indata, frames, time_info, status):
            if not self._stop_speech.is_set():
                stop_q.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                channels=1, dtype="float32", callback=callback
            ):
                max_chunks = int(1.5 * SAMPLE_RATE / BLOCK_SIZE)  # ~1.5秒检测一次

                while not self._stop_speech.is_set():
                    try:
                        chunk = stop_q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    flat = chunk.flatten()
                    is_speech = self.vad.is_speech(flat)

                    if is_speech:
                        speech_active = True
                        chunks.append(flat)

                        if len(chunks) >= max_chunks:
                            audio = np.concatenate(chunks)
                            text = self.asr.transcribe(audio)
                            chunks = []
                            # 打断词：只停 TTS，不退出对话
                            if any(w in text for w in ["停", "闭嘴", "别说了", "停下"]):
                                self._stop_speech.set()
                                self.root.after(0, lambda: self.log(f"⏹️ 语音打断: '{text}'"))
                                return
                            # 退出词：停 TTS + 退出对话模式
                            if any(w in text for w in ["结束", "退出", "再见"]):
                                self._user_interrupted = True
                                self._stop_speech.set()
                                self.root.after(0, lambda: self.log(f"⏹️ 退出对话: '{text}'"))
                                return
                    else:
                        if speech_active and len(chunks) >= int(0.3 * SAMPLE_RATE / BLOCK_SIZE):
                            # 静音了，检查刚说的
                            audio = np.concatenate(chunks)
                            text = self.asr.transcribe(audio)
                            chunks = []
                            speech_active = False
                            if any(w in text for w in ["停", "闭嘴", "别说了", "停下"]):
                                self._stop_speech.set()
                                self.root.after(0, lambda: self.log(f"⏹️ 语音打断: '{text}'"))
                                return
                            if any(w in text for w in ["结束", "退出", "再见"]):
                                self._user_interrupted = True
                                self._stop_speech.set()
                                self.root.after(0, lambda: self.log(f"⏹️ 退出对话: '{text}'"))
                                return
                        chunks = []
                        speech_active = False
        except Exception as e:
            if not self._stop_speech.is_set():
                self.log(f"⚠️ 打断监听异常: {e}")

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
        my_gen = self._current_gen
        self.log("⏹️ 已退出对话模式")
        self.root.after(0, lambda: self._reset_after_exit(my_gen))

    def _reset_after_exit(self, gen):
        """退出对话模式后的 UI 重置（带 gen 保护）"""
        if gen != self._current_gen:
            return
        self.record_btn.configure(text="🎤 按住说话", bg="#585b70")
        self.status_bar.config(text="就绪 — 按住按钮说话，松手识别")
        self.record_btn.configure(state="normal")
        if self.wake_word_enabled and not self.recording:
            self._start_wake_listener()

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
            interrupted = False
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

                # 用可打断的播放替代直接 TTS，返回是否真是用户打断
                interrupted = self._play_tts_with_interrupt(reply)

                if my_gen != self._current_gen:
                    return

                if not interrupted:
                    self.root.after(0, lambda: self.log("✅ 播放完成"))
            except Exception as e:
                self.root.after(0, lambda e=e: self.log(f"❌ 处理出错: {e}"))
            finally:
                if not self.conversation_mode or my_gen != self._current_gen:
                    self._safe_reset_btn(my_gen)
                elif interrupted or any(w in user_text for w in ["退出", "再见", "结束", "没有问题了", "就这些"]):
                    self.log(f"⏹️ 退出对话模式")
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
