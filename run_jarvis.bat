@echo off
cd /d "%~dp0"

:: ── Ollama 使用项目目录的模型 ──
set OLLAMA_MODELS=%~dp0models\ollama

:: ── 先关闭已有 Ollama 进程（含托盘图标），确保环境变量生效 ──
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM ollama.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: ── 重新启动 Ollama 服务 ──
echo 启动 Ollama 服务...
start /B ollama serve >nul 2>&1
timeout /t 4 /nobreak >nul

echo 启动贾维斯（SenseVoice 语音识别）...
"D:\anaconda3\envs\jarvis\python.exe" gui.py
pause
