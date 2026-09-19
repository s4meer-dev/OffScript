@echo off
title Deepfake Audio Detector - Gradio UI
cd /d "%~dp0"
echo =======================================================
echo Starting Deepfake Audio Detection Gradio App
echo URL: http://localhost:7860
echo =======================================================
python app_gradio.py
pause
