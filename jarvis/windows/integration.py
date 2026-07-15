"""
Windows 集成工具 — 开机自启、单实例、Ollama 进程管理
"""
import os
import time
import subprocess
import winreg
import threading
import requests
from pathlib import Path
from jarvis.config import OLLAMA_URL, OLLAMA_MODELS_DIR


# ── 开机自启 ──

REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
REG_NAME = "JarvisAssistant"


def set_autostart(enabled: bool, exe_path: str = None):
    """设置开机自启"""
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY, 0, winreg.KEY_SET_VALUE)
    if enabled and exe_path:
        winreg.SetValueEx(key, REG_NAME, 0, winreg.REG_SZ, exe_path)
    else:
        try:
            winreg.DeleteValue(key, REG_NAME)
        except FileNotFoundError:
            pass
    winreg.CloseKey(key)


def is_autostart_enabled() -> bool:
    """检查是否已设置开机自启"""
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY, 0, winreg.KEY_READ)
        try:
            val, _ = winreg.QueryValueEx(key, REG_NAME)
            return bool(val)
        except FileNotFoundError:
            return False
        finally:
            winreg.CloseKey(key)
    except Exception:
        return False


# ── 单实例 ──

class SingleInstance:
    """使用命名互斥体防止多实例"""

    def __init__(self, name: str = "JarvisVoiceAssistant"):
        try:
            import win32event
            import win32api
            import winerror
            self.mutex = win32event.CreateMutex(None, False, name)
            last_err = win32api.GetLastError()
            # WAIT_ABANDONED (0x80) = 上次进程崩溃，互斥体被遗弃，自动释放即可
            # ERROR_ALREADY_EXISTS = 已有实例在运行
            self.already_running = (last_err == winerror.ERROR_ALREADY_EXISTS and last_err != 0x80)
            self._win32event = win32event
        except ImportError:
            # 无 pywin32 时降级
            self.already_running = False
            self.mutex = None

    def release(self):
        if self.mutex is not None:
            try:
                self._win32event.CloseHandle(self.mutex)
            except Exception:
                pass


# ── Ollama 进程管理 ──

class OllamaManager:
    """管理 Ollama 子进程的生命周期"""

    def __init__(self):
        self.process = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """检查 Ollama 是否正在运行"""
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def ensure_running(self, max_wait: float = 30.0) -> bool:
        """确保 Ollama 正在运行，如果未运行则启动之。返回是否成功"""
        if self.is_running:
            return True

        with self._lock:
            if self.is_running:
                return True

            # 设置 Ollama 模型路径
            env = os.environ.copy()
            env["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR

            try:
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except FileNotFoundError:
                print("  [Ollama] 未找到 ollama 命令，请先安装 Ollama")
                return False

            # 等待就绪
            start = time.time()
            while time.time() - start < max_wait:
                if self.is_running:
                    print(f"  [Ollama] 启动成功（{time.time()-start:.1f}s）")
                    return True
                time.sleep(0.5)

            print("  [Ollama] 启动超时")
            return False

    def shutdown(self):
        """关闭 Ollama 进程"""
        with self._lock:
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                self.process = None

    def __enter__(self):
        self.ensure_running()
        return self

    def __exit__(self, *args):
        self.shutdown()
