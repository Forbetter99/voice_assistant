"""
贾维斯语音助手 — 图形界面版（桌面入口）
薄封装，实际逻辑在 jarvis.app 包中
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from jarvis.app import main

if __name__ == "__main__":
    main()
