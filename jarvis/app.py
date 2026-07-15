"""
应用控制器 — 语音助手的核心业务逻辑
整合所有模块：ASR、VAD、LLM、TTS、声纹验证、GUI、系统托盘、Ollama
"""
import os
import sys
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import tkinter as tk

from jarvis.config import (
    SAMPLE_RATE, BLOCK_SIZE, SILENCE_TIMEOUT,
    OLLAMA_URL, COLORS,
)
from jarvis.audio.vad import VAD
from jarvis.audio.asr import ASR
from jarvis.audio.tts import TTS
from jarvis.audio.recorder import (
    create_audio_stream, stop_stream, drain_queue, record_until_silence,
)
from jarvis.llm import LLM
from jarvis.speaker_verification import SpeakerVerifier
from jarvis.ui.main_window import MainWindow, ProgressBar
from jarvis.ui.tray import SystemTray
from jarvis.ui.toast import ToastNotifier
from jarvis.windows.dwm import DWM
from jarvis.windows.integration import OllamaManager, SingleInstance


# ── 系统代理检测（从 gui.py 提取）──
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
                try:
                    urllib.request.urlopen(proxy, timeout=2)
                except Exception:
                    return
                for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    os.environ.setdefault(var, proxy)
                os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    except Exception:
        pass


class Application:
    """贾维斯语音助手应用控制器"""

    def __init__(self):
        # ── 管道组件（初始化时为 None，loading_screen 中加载）──
        self.vad: VAD = None
        self.asr: ASR = None
        self.llm: LLM = None
        self.tts: TTS = None
        self.sv: SpeakerVerifier = None

        # ── 录音状态 ──
        self.recording = False
        self.audio_q = queue.Queue()
        self.audio_chunks = []
        self._stream = None
        self._test_stream = None
        self.test_audio_q = queue.Queue()
        self.test_audio_chunks = []

        # ── 处理状态 ──
        self._current_gen = 0  # 递增计数器，用于中断旧对话的处理线程

        # ── 唤醒词 ──
        self.wake_word_enabled = False
        self._stop_wake = threading.Event()
        self._wake_thread = None

        # ── 对话模式 ──
        self.conversation_mode = False

        # ── 声纹 ──
        self.owner_only = False

        # ── TTS 打断 ──
        self._stop_speech = threading.Event()
        self._user_interrupted = False

        # ── 子系统 ──
        self.ollama_mgr = OllamaManager()
        self.single_instance = SingleInstance()
        self.toaster = ToastNotifier()
        self.tray = None
        self._exe_path = None  # 安装路径，供开机自启使用

        # ── 检查单实例（遗弃的互斥体会被自动释放）──
        if self.single_instance.already_running:
            print("应用已在运行中，跳过（互斥体可能残留）")

        # ── 创建 UI ──
        self.window = MainWindow(app_controller=self)
        self.root = self.window.root
        self.root.app = self  # 兼容旧代码

        # ── 绑定事件 ──
        self._bind_events()

    def _bind_events(self):
        """绑定 UI 事件到业务逻辑"""
        self.window.bind_record(self._on_record_press, self._on_record_release)
        self.window.bind_test_record(self._on_test_record_press, self._on_test_record_release)
        self.window.bind_wake_toggle(self._toggle_wake_word)
        self.window.bind_register(self._start_speaker_registration)
        self.window.bind_owner_toggle(self._on_owner_toggle)
        self.window.set_on_close(self._on_window_close)

    # ═══════════════════════════════════════════════
    #  生命周期
    # ═══════════════════════════════════════════════

    def initialize(self):
        """先加载模型，再启动界面（避免 tkinter + KPipeline 线程冲突）"""
        print("Jarvis Voice Assistant - Loading models...")
        print()

        # 启动 Ollama（后台）
        threading.Thread(target=self._check_ollama, daemon=True).start()

        # 声纹加载（轻量）
        self.sv = SpeakerVerifier()
        if self.sv.is_registered:
            self.owner_only = True

        # 顺序加载模型（主线程）
        models = [
            ("加载 ASR 语音识别模型...", lambda: setattr(self, 'asr', ASR())),
            ("加载 VAD 语音检测模型...", lambda: setattr(self, 'vad', VAD())),
            ("加载 LLM 语言模型...", lambda: setattr(self, 'llm', LLM())),
            ("加载 TTS 语音合成模型...", lambda: setattr(self, 'tts', TTS())),
        ]

        for msg, loader in models:
            self.window.set_status(msg)
            self.root.update()
            try:
                loader()
            except Exception as e:
                print(f"  [load] {msg} FAILED: {e}")

        print()
        print("模型加载完成！")
        print()

        # 启用按钮
        self.window.enable_all_buttons(sv_is_registered=self.sv.is_registered)

        # 应用窗口效果
        try:
            DWM.apply_all(self.window.root, acrylic=True, rounded=True)
        except Exception as e:
            print(f"  [DWM] {e}")

        # 启动系统托盘（失败不影响使用）
        try:
            self.tray = SystemTray(self)
            self.tray.start()
        except Exception as e:
            print(f"  [Tray] 托盘启动失败: {e}")

    def _init_ollama(self):
        """初始化 Ollama（后台线程）"""
        try:
            self.ollama_mgr.ensure_running()
        except Exception:
            pass

    def _check_ollama(self):
        """检查 Ollama 状态并启动（后台线程）"""
        if self.ollama_mgr.is_running:
            return
        if self.ollama_mgr.ensure_running():
            print("  [Ollama] 服务已就绪")
        else:
            print("  [Ollama] 未安装。请访问 https://ollama.com/download 下载安装")
            print("  [Ollama] 安装后运行: ollama pull qwen2.5:7b")

    def run(self):
        """启动应用主循环"""
        # 先显示窗口再加载
        self.root.deiconify()
        self.root.update()
        self.initialize()
        self.root.mainloop()

    def shutdown(self):
        """安全关闭应用"""
        self.log("🛑 正在关闭...")
        self.wake_word_enabled = False
        self._stop_wake.set()
        self._stop_speech.set()
        sd.stop()

        if self.ollama_mgr:
            self.ollama_mgr.shutdown()

        self.single_instance.release()
        self.root.quit()
        self.root.destroy()
        os._exit(0)

    # ═══════════════════════════════════════════════
    #  日志 / UI 更新助手
    # ═══════════════════════════════════════════════

    def log(self, msg: str):
        """记录日志（线程安全）"""
        self.window.root.after(0, lambda: self.window.log(msg))

    def append_chat(self, text, tag=None):
        """追加聊天文本（线程安全）"""
        self.window.root.after(0, lambda: self.window.append_chat(text, tag))

    def set_status(self, text: str):
        """更新状态栏（线程安全）"""
        self.window.set_status_safe(text)

    # ═══════════════════════════════════════════════
    #  窗口事件
    # ═══════════════════════════════════════════════

    def _on_window_close(self):
        """关闭按钮：最小化到托盘"""
        self.window.hide()
        if self.tray:
            self.tray.notify(
                "贾维斯语音助手",
                "仍在后台运行，点击托盘图标恢复"
            )

    def toggle_window(self):
        """切换窗口显示/隐藏"""
        if self.window.is_visible():
            self.window.hide()
        else:
            self.window.show()

    # ═══════════════════════════════════════════════
    #  录音
    # ═══════════════════════════════════════════════

    def _on_record_press(self, event=None):
        """按下录音按钮"""
        if not self.asr or self.recording:
            return
        # 手动录音时退出对话模式、关闭唤醒监听
        self.conversation_mode = False
        self._stop_wake.set()
        self._stop_current_processing()

        self.recording = True
        self.audio_q = queue.Queue()
        self.audio_chunks = []
        self.window.set_record_btn_text("🔴 录音中...", bg=COLORS["red"])
        self.window.set_status("录音中 — 松手识别")
        self.log("🎤 开始录音")

        self._stream, self.audio_q = create_audio_stream()

        # 后台 VAD 自动停止 + 采集
        def collect():
            silent_blocks = 0
            speak_started = False
            max_silent = int(SILENCE_TIMEOUT * SAMPLE_RATE / BLOCK_SIZE)
            while self.recording:
                try:
                    chunk = self.audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                self.audio_chunks.append(chunk.flatten())
                # VAD 检测用于自动停止
                if self.vad:
                    speech = self.vad.is_speech(chunk.flatten())
                else:
                    speech = False
                if speech:
                    silent_blocks = 0
                    speak_started = True
                elif speak_started:
                    silent_blocks += 1
                if silent_blocks > max_silent and speak_started:
                    self.log("🔇 检测到静音，自动停止")
                    self.window.root.after(0, self._stop_record_auto)
                    break

        threading.Thread(target=collect, daemon=True).start()

    def _on_record_release(self, event=None):
        """松开录音按钮"""
        if not self.recording:
            return
        self.recording = False
        stop_stream(self._stream)
        self._stream = None
        self.log("✋ 手动停止录音")
        time.sleep(0.3)
        self._process_audio()

    def _stop_record_auto(self):
        """VAD 自动触发停止"""
        if not self.recording:
            return
        self.recording = False
        stop_stream(self._stream)
        self._stream = None
        time.sleep(0.3)
        self._process_audio()

    def _stop_current_processing(self):
        """中断当前处理（播放等）"""
        self._current_gen += 1
        self._stop_speech.set()
        sd.stop()

    # ═══════════════════════════════════════════════
    #  测试录音
    # ═══════════════════════════════════════════════

    def _on_test_record_press(self, event=None):
        """按下测试录音按钮"""
        if self.recording:
            return
        self.recording = True
        self.test_audio_q = queue.Queue()
        self.test_audio_chunks = []
        self.window.test_btn.configure(text="🔴 测试录音中...", bg=COLORS["red"])
        self.set_status("测试录音中... 松手回放")
        self.log("🎤 测试录音开始")

        self._test_stream, self.test_audio_q = create_audio_stream()

        def collect():
            while self.recording:
                try:
                    chunk = self.test_audio_q.get(timeout=0.3)
                    self.test_audio_chunks.append(chunk.flatten())
                except queue.Empty:
                    pass

        threading.Thread(target=collect, daemon=True).start()

    def _on_test_record_release(self, event=None):
        """松开测试录音按钮"""
        if not self.recording:
            return
        self.recording = False
        stop_stream(self._test_stream)
        self._test_stream = None

        time.sleep(0.5)
        chunks = drain_queue(self.test_audio_q)
        self.test_audio_chunks.extend(c.flatten() if hasattr(c, 'flatten') else c for c in chunks)

        self.set_status("回放中...")
        self.log("🔊 回放测试录音")

        if not self.test_audio_chunks:
            self.log("⚠️ 没有录到声音")
            self.window.test_btn.configure(text="🎧 测试录音", bg=COLORS["surface"])
            self.set_status("就绪 — 按住按钮说话，松手识别")
            return

        audio = np.concatenate(self.test_audio_chunks)
        self.log(f"📊 录音时长: {len(audio)/SAMPLE_RATE:.1f}s")
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()
        self.log("✅ 回放完成")
        self.window.test_btn.configure(text="🎧 测试录音", bg=COLORS["surface"])
        self.set_status("就绪 — 按住按钮说话，松手识别")

    # ═══════════════════════════════════════════════
    #  音频处理（ASR → LLM → TTS）
    # ═══════════════════════════════════════════════

    def _process_audio(self):
        """处理录制的音频：ASR → LLM → TTS"""
        self._current_gen += 1
        my_gen = self._current_gen

        self.window.set_record_btn_text("⏳ 处理中...", bg=COLORS["overlay"])
        self.set_status("识别中...")

        # 耗尽队列
        leftovers = drain_queue(self.audio_q)
        self.audio_chunks.extend(leftovers)

        if not self.audio_chunks:
            self.log("⚠️ 没有录到音频")
            self.append_chat("[没有录到声音，请重试]", "gray")
            self._safe_reset_btn(my_gen)
            return

        audio = np.concatenate(self.audio_chunks)
        if len(audio) < SAMPLE_RATE * 0.3:
            self.log("⚠️ 录音太短，已忽略")
            self.append_chat("[录音太短，已忽略]", "gray")
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
                    self.log("⚠️ 语音识别无结果")
                    self.append_chat("[没听清，请再说一遍]", "gray")
                    self._safe_reset_btn(my_gen)
                    return

                self.log(f"🧑 识别结果: {text}")
                self.append_chat(f"🧑 你: {text}")

                if my_gen != self._current_gen:
                    return

                self.set_status("思考中...")
                self.log("💭 请求 LLM...")

                reply = self.llm.chat(text)

                if my_gen != self._current_gen:
                    return

                self.log(f"🤖 LLM 回复: {reply}")
                self.append_chat(f"🤖 贾维斯: {reply}", "jarvis")
                self.set_status("播放语音...")
                self.log("🔊 合成语音并播放...")

                interrupted = self._play_tts_with_interrupt(reply)

                if my_gen != self._current_gen:
                    return

                if not interrupted:
                    self.log("✅ 播放完成")
            except Exception as e:
                self.log(f"❌ 处理出错: {e}")
            finally:
                if not self.conversation_mode or my_gen != self._current_gen:
                    self._safe_reset_btn(my_gen)
                elif interrupted or any(w in user_text for w in ["退出", "再见", "结束", "没有问题了", "就这些"]):
                    self.log("⏹️ 退出对话模式")
                    self.root.after(0, self._exit_conversation_mode)
                else:
                    self.root.after(0, self._begin_conversation_round)

        threading.Thread(target=work, daemon=True).start()

    def _safe_reset_btn(self, gen):
        """仅在指定代仍为当前代时重置按钮状态"""
        def reset():
            if gen == self._current_gen:
                self.window.set_record_btn_text("🎤 按住说话")
                self.set_status("就绪 — 按住按钮说话，松手识别")
                if self.wake_word_enabled and not self.recording:
                    self._start_wake_listener()
        self.window.after(0, reset)

    # ═══════════════════════════════════════════════
    #  TTS 打断
    # ═══════════════════════════════════════════════

    def _play_tts_with_interrupt(self, text: str) -> bool:
        """流式播放 TTS + 后台监听打断词，返回 True=用户打断"""
        if not text or not self.tts or not self.tts.pipeline:
            return False

        self._stop_speech.clear()
        self._user_interrupted = False

        listener_thread = threading.Thread(target=self._interrupt_listener_loop, daemon=True)
        listener_thread.start()

        generator = self.tts.pipeline(text, voice="zf_xiaobei", speed=1.3)
        for gs, ps, audio in generator:
            if self._stop_speech.is_set():
                sd.stop()
                break

            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            peak = np.max(np.abs(audio))
            if peak > 0 and peak < 0.6:
                audio = audio * (0.7 / peak)
            audio = np.clip(audio, -1.0, 1.0)

            sd.play(audio, samplerate=24000)
            while sd.get_stream() and sd.get_stream().active:
                if self._stop_speech.is_set():
                    sd.stop()
                    break
                time.sleep(0.05)

        self._stop_speech.set()
        listener_thread.join(timeout=1.0)

        return self._user_interrupted

    def _interrupt_listener_loop(self):
        """TTS 播放时后台监听打断词"""
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
                max_chunks = int(1.5 * SAMPLE_RATE / BLOCK_SIZE)

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
                            if any(w in text for w in ["停", "闭嘴", "别说了", "停下"]):
                                self._stop_speech.set()
                                self.root.after(0, lambda t=text: self.log(f"⏹️ 语音打断: '{t}'"))
                                return
                            if any(w in text for w in ["结束", "退出", "再见"]):
                                self._user_interrupted = True
                                self._stop_speech.set()
                                self.root.after(0, lambda t=text: self.log(f"⏹️ 退出对话: '{t}'"))
                                return
                    else:
                        if speech_active and len(chunks) >= int(0.3 * SAMPLE_RATE / BLOCK_SIZE):
                            audio = np.concatenate(chunks)
                            text = self.asr.transcribe(audio)
                            chunks = []
                            speech_active = False
                            if any(w in text for w in ["停", "闭嘴", "别说了", "停下"]):
                                self._stop_speech.set()
                                self.root.after(0, lambda t=text: self.log(f"⏹️ 语音打断: '{t}'"))
                                return
                            if any(w in text for w in ["结束", "退出", "再见"]):
                                self._user_interrupted = True
                                self._stop_speech.set()
                                self.root.after(0, lambda t=text: self.log(f"⏹️ 退出对话: '{t}'"))
                                return
                        chunks = []
                        speech_active = False
        except Exception as e:
            if not self._stop_speech.is_set():
                self.log(f"⚠️ 打断监听异常: {e}")

    # ═══════════════════════════════════════════════
    #  语音唤醒
    # ═══════════════════════════════════════════════

    def toggle_wake_word(self):
        """开关语音唤醒（外部调用，如托盘菜单）"""
        if self.wake_word_enabled:
            self._disable_wake_word()
        else:
            self._enable_wake_word()

    def _toggle_wake_word(self):
        """开关语音唤醒（UI 按钮回调）"""
        self.toggle_wake_word()

    def _enable_wake_word(self):
        self.wake_word_enabled = True
        self._stop_wake.clear()
        self.window.wake_btn.config(text="🔊 聆听中...", bg=COLORS["green"], fg="#11111b")
        self.log("🔊 语音唤醒已开启，说'贾维斯'唤醒")
        self._start_wake_listener()
        if self.tray:
            self.tray.update_menu()

    def _disable_wake_word(self):
        self.wake_word_enabled = False
        self._stop_wake.set()
        self.window.wake_btn.config(text="🤖 语音唤醒", bg=COLORS["surface"], fg=COLORS["text_muted"])
        self.log("⏹️ 语音唤醒已关闭")
        if self.conversation_mode:
            self._current_gen += 1
            self.conversation_mode = False
            self.window.set_record_btn_text("🎤 按住说话", bg=COLORS["overlay"])
            self.set_status("就绪 — 按住按钮说话，松手识别")
        if self.tray:
            self.tray.update_menu()

    def _start_wake_listener(self):
        self._wake_thread = threading.Thread(target=self._wake_word_loop, daemon=True)
        self._wake_thread.start()

    def _wake_word_loop(self):
        """后台监听唤醒词'贾维斯'"""
        import numpy as np
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

                        if len(speech_blocks) >= max_check:
                            audio = np.concatenate(speech_blocks)
                            text = self.asr.transcribe(audio)
                            if "贾维斯" in text:
                                self._on_wake_verified(audio)
                                return
                            speech_blocks = []
                    else:
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

    def _on_wake_word_detected(self):
        """唤醒词检测到，进入对话模式"""
        self.log("🔊 检测到唤醒词 '贾维斯'，进入对话模式")
        self.set_status("对话模式 — 说'退出'结束对话")
        self._play_beep()
        self.conversation_mode = True
        self._begin_conversation_round()

        if not self.window.is_visible():
            self.window.show()
            if self.tray:
                self.tray.notify("贾维斯", "已唤醒，进入对话模式")

    def _play_beep(self):
        """播放简短提示音"""
        beep_t = np.linspace(0, 0.15, int(24000 * 0.15), False)
        beep_audio = np.sin(2 * np.pi * 800 * beep_t) * 0.3
        sd.play(beep_audio, samplerate=24000)

    # ═══════════════════════════════════════════════
    #  对话模式
    # ═══════════════════════════════════════════════

    def _begin_conversation_round(self):
        """对话模式下自动开始一轮录音（VAD 自动停止）"""
        if not self.conversation_mode or self.recording:
            return
        stop_stream(self._stream)
        self._stream = None
        self._stop_wake.set()
        self.recording = True
        self.audio_q = queue.Queue()
        self.audio_chunks = []
        self.window.set_record_btn_text("🔴 聆听中...", bg=COLORS["red"])
        self.set_status("对话模式 — 请说话")

        self._stream, self.audio_q = create_audio_stream()

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
                    self.log("🔇 检测到静音，自动停止")
                    self.root.after(0, self._stop_conversation_record)
                    break

        threading.Thread(target=collect, daemon=True).start()

    def _stop_conversation_record(self):
        """对话模式停止录音并处理"""
        if not self.recording:
            return
        self.recording = False
        stop_stream(self._stream)
        self._stream = None
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
        if gen != self._current_gen:
            return
        self.window.set_record_btn_text("🎤 按住说话", bg=COLORS["overlay"])
        self.set_status("就绪 — 按住按钮说话，松手识别")
        if self.wake_word_enabled and not self.recording:
            self._start_wake_listener()

    # ═══════════════════════════════════════════════
    #  声纹注册
    # ═══════════════════════════════════════════════

    def _on_owner_toggle(self):
        """仅机主唤醒开关"""
        self.owner_only = self.window.owner_var.get()
        if self.owner_only and not self.sv.is_registered:
            self.log("⚠️ 请先注册声纹，再开启'仅机主唤醒'")
            self.window.owner_var.set(False)
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
        self.set_status(f"声纹注册 第1/{self._regist_total}句 — 请说话")
        self._start_regist_record()

    def _start_regist_record(self):
        """启动录音用于声纹注册"""
        self._stop_wake.set()
        self.recording = True
        self.audio_q = queue.Queue()
        self._regist_chunks = []
        self._stream, self.audio_q = create_audio_stream()

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
        stop_stream(self._stream)
        self._stream = None
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
            self.log("🧬 正在提取声纹特征...")
            self.set_status("正在处理声纹...")
            threading.Thread(target=self._do_registration, daemon=True).start()
        else:
            step = self._regist_step + 1
            self.set_status(f"声纹注册 第{step}/{self._regist_total}句 — 请说话")
            self.log(f"📢 请说第{step}句话")
            self.root.after(500, self._start_regist_record)

    def _do_registration(self):
        """后台执行声纹注册"""
        try:
            msg = self.sv.register(self._regist_phrases)
            self.log(f"✅ {msg}")
            self.root.after(0, lambda: self.set_status("声纹注册成功 ✅"))
            self.root.after(0, lambda: self.window.owner_var.set(True))
            self.root.after(0, lambda: setattr(self, 'owner_only', True))
            self.root.after(0, lambda: self.log("🔒 已自动开启'仅机主唤醒'"))
            self.root.after(2000, lambda: self.set_status(
                "就绪 — 声纹已注册，仅机主可唤醒" if self.owner_only else "就绪"
            ))
            self.root.after(500, self._try_restart_wake)
        except Exception as e:
            self.log(f"❌ 声纹注册失败: {e}")
            self.root.after(500, self._try_restart_wake)
        finally:
            self._regist_phrases = []

    def _try_restart_wake(self):
        if self.wake_word_enabled and not self.recording:
            self._stop_wake.clear()
            self._start_wake_listener()


# ── 入口函数 ──

def main():
    """应用入口"""
    _apply_windows_proxy()

    # stdout 行缓冲
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    app = Application()
    app.run()


if __name__ == "__main__":
    main()
