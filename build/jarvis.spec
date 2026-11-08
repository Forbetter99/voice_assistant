# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 — 贾维斯语音助手 Windows 桌面版

入口是项目根目录的 gui.py（现存主程序），不是旧重构分支的 jarvis/__main__.py。
models/ 目录（约 13GB 模型）不打进 exe，需随安装包放在程序同级的 models/ 下。
"""
import os
from pathlib import Path

block_cipher = None

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

a = Analysis(
    [str(PROJECT_ROOT / 'gui.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / 'build' / 'icon.ico'), '.'),
        (str(PROJECT_ROOT / 'build' / 'icon.png'), '.'),
    ],
    hiddenimports=[
        # Silero VAD
        'silero_vad',
        'silero_vad.utils_vad',
        'silero_vad.model',
        # ASR: Qwen3-ASR
        'qwen_asr',
        'transformers',
        'transformers.models',
        'bitsandbytes',
        'accelerate',
        # Kokoro TTS
        'kokoro',
        'kokoro.pipeline',
        # 音频
        'sounddevice',
        '_sounddevice_data',
        'soundfile',
        # UI 自动化（QQ 音乐 / B站）
        'uiautomation',
        'comtypes',
        # PIL for tray icon
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        # pystray 托盘
        'pystray',
        # pywin32
        'win32api',
        'win32event',
        'win32gui',
        'win32con',
        'winerror',
        # plyer for toast
        'plyer',
        'plyer.platforms.win',
        'plyer.platforms.win.notification',
    ],
    hookspath=[],
    hooksconfig={},
    excludes=[
        # 只排除确定用不到的大块头，避免误伤 torch / scipy / torchaudio
        'matplotlib',
        'notebook',
        'IPython',
        'jupyter',
        'jupyter_client',
        'jupyter_core',
        'nbformat',
        'nbconvert',
        'pip',
        'tkinter.tix',
        'test',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Jarvis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # 不显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'build' / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Jarvis',
)
