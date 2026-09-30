@echo off
title Lagom Humanizer API Server
cd /d "%~dp0"
echo ========================================================
echo   Lagom Humanizer API Server (FastAPI + Uvicorn)
echo   Listening on: http://127.0.0.1:8000
echo   API Docs:     http://127.0.0.1:8000/docs
echo ========================================================
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
