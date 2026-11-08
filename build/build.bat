@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   贾维斯语音助手 — 构建工具
echo ============================================
echo.

:: ── 选择解释器：优先 jarvis conda 环境 ──
set PYTHON=python
if exist "D:\anaconda3\envs\jarvis\python.exe" set PYTHON=D:\anaconda3\envs\jarvis\python.exe
echo 使用解释器: %PYTHON%
echo.

:: ── 步骤 1：生成图标 ──
echo [1/4] 生成应用图标...
"%PYTHON%" generate_icon.py
if %errorlevel% neq 0 (
    echo [警告] 图标生成失败，将使用默认图标
)
echo.

:: ── 步骤 2：检查依赖 ──
echo [2/4] 检查构建依赖...
"%PYTHON%" -c "import PyInstaller" >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 PyInstaller，请安装：pip install pyinstaller
    pause
    exit /b 1
)
echo    PyInstaller 已找到
echo.

:: ── 步骤 3：PyInstaller 打包 ──
echo [3/4] 使用 PyInstaller 打包...
echo    这可能需要 5-15 分钟...
echo.

"%PYTHON%" -m PyInstaller jarvis.spec --clean
if %errorlevel% neq 0 (
    echo [错误] 打包失败
    pause
    exit /b 1
)
echo.

:: ── 步骤 4：检查结果 ──
echo [4/4] 检查构建结果...
if exist "..\dist\Jarvis\Jarvis.exe" (
    for %%f in ("..\dist\Jarvis\Jarvis.exe") do echo    打包成功！文件大小: %%~zf 字节
    echo.
    echo    输出目录: %~dp0..\dist\Jarvis\
    echo    可执行文件: %~dp0..\dist\Jarvis\Jarvis.exe
) else (
    echo [错误] 未找到输出文件
    pause
    exit /b 1
)
echo.

echo ============================================
echo   构建完成！
echo ============================================
echo.
echo 下一步：
echo   1. 运行 dist\Jarvis\Jarvis.exe 测试
echo   2. 如需制作安装包，打开 Inno Setup 编译 installer.iss
echo.
pause
