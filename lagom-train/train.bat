@echo off
title Lagom Training - QLoRA Fine-Tuning
cd /d "%~dp0"

echo ========================================================
echo   Lagom Humanizer - QLoRA Training
echo   Model  : Qwen/Qwen2.5-3B-Instruct (4-bit NF4)
echo   Preset : local8gb  ^(LoRA r=32, alpha=64^)
echo   Epochs : 3
echo ========================================================
echo.

REM Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Make sure it is on your PATH.
    pause
    exit /b 1
)

REM Install / upgrade deps silently
echo [*] Checking dependencies...
pip install -q -r requirements.txt

echo.
echo [*] Starting training...
echo     Output will be saved to: outputs\adapter
echo     API will auto-discover it when you run api\start.bat
echo.

python train.py --preset local8gb --epochs 3 --output-dir outputs

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed. See error above.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Training complete!
echo   Adapter saved to: lagom-train\outputs\adapter
echo   You can now run api\start.bat to launch the API.
echo ========================================================
echo.
pause
