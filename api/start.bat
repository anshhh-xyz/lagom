@echo off
title Lagom Humanizer API Server
cd /d "%~dp0"

echo ========================================================
echo   Lagom Humanizer API Server (FastAPI + Uvicorn)
echo   Listening on: http://127.0.0.1:8000
echo   API Docs:     http://127.0.0.1:8000/docs
echo ========================================================

REM Kill any existing process on port 8000 cleanly via PowerShell
echo [*] Freeing port 8000 if already in use...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul

echo [*] Starting Lagom API Server...
python -m uvicorn main:app --host 127.0.0.1 --port 8000
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server failed to start. Exit code: %ERRORLEVEL%
    echo Make sure all dependencies are installed:
    echo   pip install -r ..\lagom-train\requirements.txt
    echo.
)
pause
