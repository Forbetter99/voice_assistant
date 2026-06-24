#!/bin/bash
# Jarvis 语音助手 - 一键安装脚本 (Windows Git Bash)

echo "=== Jarvis 语音助手 安装脚本 ==="
echo ""

# 检查 Ollama
if ! command -v ollama &>/dev/null && ! [ -f "/c/Program Files/Ollama/ollama.exe" ]; then
    echo "[1/4] 安装 Ollama..."
    winget install Ollama.Ollama --silent
    echo "   Ollama 安装完成，请启动 Ollama（开始菜单搜索 Ollama）"
    echo "   然后运行: ollama pull qwen2.5:7b"
else
    echo "[1/4] ✓ Ollama 已安装"
fi

# 拉取模型
echo "[2/4] 拉取 LLM 模型（约 4.5GB，首次下载较慢）..."
ollama pull qwen2.5:7b 2>/dev/null || echo "   请稍后手动运行: ollama pull qwen2.5:7b"

# 安装 Python 依赖
echo "[3/4] 安装 Python 依赖..."
cd "$(dirname "$0")"
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

echo ""
echo "[4/4] 安装完成！"
echo ""
echo "=== 使用方式 ==="
echo "1. 确保 Ollama 在后台运行"
echo "2. 运行: python assistant.py"
echo "3. 按 Enter 开始说话，静音自动结束"
echo "================="
