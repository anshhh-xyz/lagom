@echo off
title Lagom Humanizer API Server
cd /d "%~dp0"

echo ========================================================
echo   Lagom Humanizer API Server (FastAPI + Uvicorn)
echo   Listening on: http://127.0.0.1:8000
echo   API Docs:     http://127.0.0.1:8000/docs
echo ========================================================

REM Install API deps
echo [*] Checking API dependencies...
pip install -q -r requirements.txt

REM Install model/inference deps from lagom-train
echo [*] Checking model dependencies...
pip install -q -r ..\lagom-train\requirements.txt

REM Kill any existing process on port 8000 cleanly via PowerShell
echo [*] Freeing port 8000 if already in use...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul

echo [*] Starting Lagom API Server...
python -m uvicorn main:app --host 127.0.0.1 --port 8000
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server failed to start. Exit code: %ERRORLEVEL%
    echo Make sure training has been completed first:
    echo   Run lagom-train\train.bat to train the model.
    echo.
)
pause
