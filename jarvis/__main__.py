"""
入口点：python -m jarvis
"""
import sys
from pathlib import Path

# 确保项目根目录在 path 中
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from jarvis.app import main

main()
