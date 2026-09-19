@echo off
title Deepfake Audio Detector - FastAPI Server
cd /d "%~dp0"
echo =======================================================
echo Starting Deepfake Audio Detection Server (FastAPI)
echo Web Dashboard: http://localhost:8000
echo Swagger Docs:   http://localhost:8000/docs
echo =======================================================
python -m uvicorn server:app --host 0.0.0.0 --port 8000
pause
