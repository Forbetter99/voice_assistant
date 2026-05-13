@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo       语音助手 Web 服务启动器
echo ========================================
echo.

:: Check API Key
if "%DEEPSEEK_API_KEY%"=="" (
    echo [提示] 未检测到环境变量 DEEPSEEK_API_KEY
    echo.
    set /p key="请输入你的 DeepSeek API Key: "
    set DEEPSEEK_API_KEY=%key%
)

echo.
echo 正在启动服务器...
echo 手机在同一 WiFi 下访问: http://本机IP:8000
echo 按 Ctrl+C 退出
echo.

python server.py

pause
