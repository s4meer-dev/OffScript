@echo off
title Multimodal Deepfake Detector - FastAPI Server
cd /d "%~dp0"
echo =======================================================
echo Starting Multimodal Deepfake Detection Server (FastAPI)
echo Web Dashboard: http://localhost:8000
echo Swagger Docs:   http://localhost:8000/docs
echo =======================================================
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
