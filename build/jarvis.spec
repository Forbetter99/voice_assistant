# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 — 贾维斯语音助手 Windows 桌面版
"""
import os
from pathlib import Path

block_cipher = None

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

a = Analysis(
    [str(PROJECT_ROOT / 'jarvis' / '__main__.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / 'build' / 'icon.ico'), '.'),
    ],
    hiddenimports=[
        # Silero VAD
        'silero_vad',
        'silero_vad.utils_vad',
        'silero_vad.model',
        # Funasr / Paraformer
        'funasr',
        'funasr.models',
        'funasr.runtime_utils',
        # Kokoro
        'kokoro',
        'kokoro.pipeline',
        # Sounddevice
        'sounddevice',
        '_sounddevice_data',
        # PIL for tray icon
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        # pystray
        'pystray',
        # pywin32
        'win32api',
        'win32event',
        'winerror',
        # plyer for toast
        'plyer',
        'plyer.platforms.win',
        'plyer.platforms.win.notification',
    ],
    hookspath=[],
    hooksconfig={},
    excludes=[
        # 排除不需要的 torch 组件（可节省 ~1GB）
        'torchvision',
        'torchaudio',
        'torch._inductor',
        'torch._dynamo',
        'torch.onnx',
        'torch.distributed',
        'torch.testing',
        'torch.jit',
        'torch.profiler',
        'torch.backends._coreml',
        'torch.backends._mps',
        'torch.backends.quantized',
        'torch.nn.parallel',
        'torch.quantization',
        # 排除不需要的库
        'matplotlib',
        'scipy',
        'notebook',
        'IPython',
        'jupyter',
        'jupyter_client',
        'jupyter_core',
        'nbformat',
        'nbconvert',
        'setuptools',
        'pip',
        'tkinter.tix',
        'tkinter.ttk',
        'test',
        'unittest',
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
