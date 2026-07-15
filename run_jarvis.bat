@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   贾维斯 语音助手 — Windows 桌面版
echo ============================================
echo.

:: ── 检测已安装版本 ──
if exist "%LOCALAPPDATA%\Jarvis\Jarvis.exe" (
    echo 启动已安装版本...
    start "" "%LOCALAPPDATA%\Jarvis\Jarvis.exe"
    exit /b
)

:: ── 从源代码运行 ──
echo 从源代码启动...

:: 查找 Python
set PYTHON_CMD=

if exist "D:\anaconda3\envs\jarvis\python.exe" (
    set PYTHON_CMD="D:\anaconda3\envs\jarvis\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
    ) else (
        echo [错误] 未找到 Python，请先安装 Python 3.10+
        pause
        exit /b 1
    )
)

echo 启动中...
%PYTHON_CMD% -m jarvis
pause
