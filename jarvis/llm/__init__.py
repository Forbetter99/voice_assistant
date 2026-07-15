"""
LLM 客户端 — 基于 Ollama（本地）
使用 Python ollama 库连接本地 Ollama 服务
"""
import json
import re
from pathlib import Path
from jarvis.config import OLLAMA_URL, OLLAMA_MODEL


class LLM:
    def __init__(self, model=OLLAMA_MODEL):
        import ollama
        self._ollama = ollama
        self.client = ollama.Client(host=OLLAMA_URL)
        self.model = model
        self.history = []
        self.max_turns = 10
        self.available = False
        # 检查连接
        try:
            self.client.list()
            self.available = True
        except Exception:
            self.available = False

        self.system_prompt = (
            "你是贾维斯(Jarvis)，一个智能语音助手。"
            "你有以下工具可用：查天气、查时间、保存代码、打开网页、"
            "打开本地应用(计算器/记事本等)、控制窗口(最小化/关闭/切换)、"
            "系统操作(锁屏/音量/截图等)。"
            "需要时直接使用对应的工具。"
            "用户的问题可能是语音识别结果，如有错别字请自动修正。"
            "用中文回答，简洁自然。使用工具后请告知用户结果。"
        )

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
                        "properties": {"city": {"type": "string", "description": "城市中文名"}},
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "获取当前的日期和时间",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_code",
                    "description": "将代码保存到文件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "content": {"type": "string"},
                            "language": {"type": "string"}
                        },
                        "required": ["filename", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_browser",
                    "description": "在默认浏览器中打开网址",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "打开 Windows 本地应用",
                    "parameters": {
                        "type": "object",
                        "properties": {"app": {"type": "string"}},
                        "required": ["app"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "control_window",
                    "description": "控制窗口",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "target": {"type": "string"}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "system_action",
                    "description": "系统操作",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["action"]
                    }
                }
            },
        ]

    def _get_weather(self, city: str) -> str:
        import requests as r
        try:
            resp = r.get(f"https://wttr.in/{city}?format=%C+%t+%h+%w&lang=zh", timeout=10, headers={"User-Agent": "curl/8.0"})
            return f"{city}天气: {resp.text.strip()}" if resp.status_code == 200 else f"无法获取{city}的天气信息"
        except Exception as e:
            return f"查询天气失败: {e}"

    def _get_time(self) -> str:
        import datetime
        now = datetime.datetime.now()
        wd = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"][now.weekday()]
        return f"当前时间: {now.year}年{now.month}月{now.day}日 {wd} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"

    def _save_code(self, filename: str, content: str, language: str = "") -> str:
        from jarvis.config import GENERATED_DIR
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        (GENERATED_DIR / filename).write_text(content, encoding="utf-8")
        return f"代码已保存到 {filename}"

    def _open_browser(self, url: str) -> str:
        import webbrowser
        if not url.startswith("http"): url = "https://" + url
        webbrowser.open(url)
        return f"已打开网页: {url}"

    def _open_app(self, app: str) -> str:
        import subprocess, os
        APP_MAP = {
            "微信":"WeChat.exe","chrome":"chrome.exe","edge":"msedge.exe","浏览器":"msedge.exe",
            "vscode":"Code.exe","word":"WINWORD.EXE","excel":"EXCEL.EXE",
            "计算器":"calc.exe","记事本":"notepad.exe","cmd":"cmd.exe","powershell":"powershell.exe",
            "任务管理器":"taskmgr.exe","资源管理器":"explorer.exe","设置":"ms-settings:",
        }
        key = app.lower() if app.isascii() else app
        exe = APP_MAP.get(key)
        if exe:
            try: subprocess.Popen(f'cmd /c start "" "{exe}"', shell=True); return f"已打开: {app}"
            except: pass
        try: subprocess.Popen(f'cmd /c start "" "{app}"', shell=True); return f"已打开: {app}"
        except: pass
        return f"未找到: {app}"

    def _control_window(self, action: str, target: str = "") -> str:
        import subprocess
        try:
            if action in ("minimize_all","全部最小化"):
                subprocess.Popen("powershell -Command (New-Object -ComObject Shell.Application).MinimizeAll()", shell=True)
                return "已全部最小化"
            if action in ("show_desktop","显示桌面"):
                subprocess.Popen("powershell -Command (New-Object -ComObject Shell.Application).ToggleDesktop()", shell=True)
                return "已显示桌面"
            if not target: return f"已执行: {action}"
            try:
                import pygetwindow as gw
                wins = gw.getWindowsWithTitle(target)
                if not wins: return f"未找到窗口"
                w = wins[0]
                if action in ("最小化","minimize"): w.minimize()
                elif action in ("关闭","close"): w.close()
                elif action in ("激活","activate","switch"): w.activate()
                else: return f"不支持: {action}"
                return f"已{action}: {target}"
            except: return f"需要 pygetwindow 库"
        except Exception as e:
            return f"窗口控制失败: {e}"

    def _system_action(self, action: str, value: str = "") -> str:
        import subprocess
        try:
            if action in ("锁屏","lock"): subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True); return "已锁屏"
            if action in ("截图","screenshot"): subprocess.run("SnippingTool.exe", shell=True); return "已打开截图工具"
            if "音量" in action:
                k = "[char]175" if any(w in action for w in ["增","up","大"]) else "[char]174"
                subprocess.run(f'powershell -Command (New-Object -ComObject WScript.Shell).SendKeys({k})', shell=True)
                return "音量已调整"
            if action in ("关机","shutdown"): subprocess.run("shutdown /s /t 5", shell=True); return "将在5秒后关机"
            if action in ("重启","restart"): subprocess.run("shutdown /r /t 5", shell=True); return "将在5秒后重启"
            if action in ("睡眠","sleep"): subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True); return "已进入睡眠"
            return f"未知操作: {action}"
        except Exception as e:
            return f"系统操作失败: {e}"

    def _execute_tool(self, name: str, args: dict) -> str:
        h = {
            "get_weather": lambda: self._get_weather(args.get("city","")),
            "get_time": lambda: self._get_time(),
            "save_code": lambda: self._save_code(args.get("filename","code.py"), args.get("content",""), args.get("language","")),
            "open_browser": lambda: self._open_browser(args.get("url","")),
            "open_app": lambda: self._open_app(args.get("app","")),
            "control_window": lambda: self._control_window(args.get("action",""), args.get("target","")),
            "system_action": lambda: self._system_action(args.get("action",""), args.get("value","")),
        }
        fn = h.get(name)
        return fn() if fn else f"未知工具 '{name}'"

    def chat(self, user_text: str) -> str:
        if not self.available:
            return "Ollama 未安装。请从 https://ollama.com/download 下载安装，然后运行: ollama pull qwen2.5:7b"

        self.history.append({"role": "user", "content": user_text})

        for _ in range(5):
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}]
                            + self.history[-self.max_turns * 2:],
                    tools=self._build_tools(),
                )
            except Exception as e:
                return f"Ollama 调用失败: {e}"

            msg = resp["message"]

            if msg.get("tool_calls"):
                self.history.append({"role": "assistant", "content": msg.get("content",""), "tool_calls": msg["tool_calls"]})
                silent_tools = {"open_app", "control_window", "system_action"}
                last_name, last_result = "", ""

                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try: raw_args = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
                    except: raw_args = {}
                    result = self._execute_tool(name, raw_args)
                    self.history.append({"role": "tool", "content": result})
                    last_name, last_result = name, result

                if last_name in silent_tools:
                    return last_result
                continue

            reply = msg.get("content", "")
            self.history.append({"role": "assistant", "content": reply})
            return reply

        return "任务已完成。"
