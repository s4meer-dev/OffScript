"""
FastAPI Server for Multimodal Audio-Visual Deepfake Detection & Localization.
Integrates Speech Detection (Wav2Vec2) and Computer Vision / Temporal Localization (AV-Deepfake1M).
Hosts REST API, demo samples, and an interactive Web Dashboard.
"""

import io
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from multimodal_detector import UnifiedDeepfakeDetector

app = FastAPI(
    title="Unified Audio-Visual Deepfake Detection API",
    description="Multimodal deepfake detection and temporal localization powered by Wav2Vec2 and AV-Deepfake1M benchmarks",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

detector: Optional[UnifiedDeepfakeDetector] = None


@app.on_event("startup")
async def startup_event():
    global detector
    print("Initializing Unified Deepfake Detector on startup...")
    detector = UnifiedDeepfakeDetector()
    print("Unified Detector ready for incoming requests.")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/health", summary="Health Check")
async def health_check():
    return {
        "status": "online",
        "model_ready": detector is not None,
        "device": str(detector.device) if detector else "uninitialized",
        "multimodal_ready": True,
        "modalities": ["audio", "video", "audio_visual"],
        "frameworks": ["Wav2Vec2", "AV-Deepfake1M (Computer Vision & Temporal Localization)"],
    }


@app.get("/api/info", summary="Model & Architecture Information")
async def model_info():
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")
    return {
        "system_name": "Unified Multimodal Audio-Visual Deepfake Detection & Localization Platform",
        "audio_model": {
            "name": "mo-thecreator/Deepfake-audio-detection",
            "architecture": detector.audio_detector.model.config.architectures,
            "target_sample_rate": 16000,
            "labels": detector.audio_detector.model.config.id2label,
        },
        "visual_model": {
            "name": "DeepfakeVisionDetector",
            "architecture": "Biometric Facial ROI, Spatial Laplacian Boundary Analysis, 2D FFT Frequency, Temporal Motion Flicker, MobileNet",
            "target_fps": detector.vision_detector.target_fps,
            "max_frames": detector.vision_detector.max_frames,
        },
        "taxonomy": {
            "standard": "AV-Deepfake1M 4-Class Taxonomy",
            "classes": ["real", "audio_modified", "visual_modified", "both_modified"],
        },
        "temporal_localization": {
            "supported": True,
            "outputs": ["fake_segments", "audio_fake_segments", "visual_fake_segments"],
        },
        "device": str(detector.device),
    }


@app.get("/samples/{filename}", summary="Download Demo Sample")
async def get_sample_file(filename: str):
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    media_type = "video/mp4" if filename.endswith((".mp4", ".webm")) else "audio/wav"
    return FileResponse(file_path, media_type=media_type)


@app.post("/api/predict", summary="Classify Audio or Video File")
async def predict_media(
    file: UploadFile = File(..., description="Media file (WAV, MP3, FLAC, OGG, MP4, WebM, AVI, MOV, etc.)"),
    audio_threshold: Optional[float] = Query(None, ge=0.50, le=0.99, description="Audio deepfake threshold"),
    visual_threshold: Optional[float] = Query(None, ge=0.40, le=0.95, description="Visual deepfake threshold"),
    threshold: Optional[float] = Query(None, ge=0.50, le=0.99, description="Legacy threshold alias"),
):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Resolve threshold parameters
        a_thresh = audio_threshold if audio_threshold is not None else (threshold if threshold is not None else 0.85)
        v_thresh = visual_threshold if visual_threshold is not None else 0.65

        result = detector.predict(
            content,
            filename=file.filename,
            audio_threshold=a_thresh,
            visual_threshold=v_thresh,
        )
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process media: {str(e)}")


@app.get("/", response_class=HTMLResponse, summary="Interactive Multimodal Web Dashboard")
async def index():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio-Visual Deepfake Detection & Localization</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #131b2e;
            --surface-hover: #1a253e;
            --border: #1e2c4a;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.4);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.4);
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.4);
            --purple: #a855f7;
            --purple-glow: rgba(168, 85, 247, 0.4);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.4);
            --text: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: rgba(19, 27, 46, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge {
            background: #1e293b;
            color: #94a3b8;
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 9999px;
            border: 1px solid var(--border);
            font-weight: 600;
        }

        .container {
            max-width: 1060px;
            margin: 0 auto;
            padding: 32px 20px;
            flex: 1;
            width: 100%;
        }

        .hero {
            text-align: center;
            margin-bottom: 24px;
        }

        .hero h1 {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.025em;
        }

        .hero p {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 760px;
            margin: 0 auto;
        }

        .samples-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .sample-chip {
            background: #1e293b;
            border: 1px solid var(--border);
            padding: 7px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            color: var(--text);
        }

        .sample-chip:hover {
            border-color: var(--accent);
            background: #27354f;
            transform: translateY(-1px);
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }

        .controls-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            padding: 12px 16px;
            background: rgba(11, 15, 25, 0.5);
            border-radius: 10px;
            border: 1px solid var(--border);
            flex-wrap: wrap;
            gap: 12px;
        }

        .mode-select {
            background: #1e293b;
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 16px;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tab-btn.active {
            background: #1e293b;
            color: #fff;
        }

        .dropzone {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 36px 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(11, 15, 25, 0.5);
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent);
            background: rgba(59, 130, 246, 0.05);
            box-shadow: 0 0 20px var(--accent-glow);
        }

        .dropzone svg {
            width: 44px;
            height: 44px;
            color: var(--accent);
            margin-bottom: 10px;
        }

        .dropzone p {
            font-size: 1rem;
            margin-bottom: 6px;
        }

        .dropzone span {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .mic-box {
            display: none;
            text-align: center;
            padding: 28px 20px;
        }

        .mic-btn {
            width: 76px;
            height: 76px;
            border-radius: 50%;
            background: #1e293b;
            border: 2px solid var(--border);
            color: #ef4444;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            margin-bottom: 12px;
        }

        .mic-btn.recording {
            background: #ef4444;
            color: #fff;
            box-shadow: 0 0 30px var(--danger-glow);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.08); }
            100% { transform: scale(1); }
        }

        /* Previews */
        .preview-box {
            display: none;
            margin-top: 20px;
            padding: 16px;
            background: rgba(11, 15, 25, 0.6);
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }

        video, audio {
            width: 100%;
            border-radius: 8px;
            background: #000;
        }

        video {
            max-height: 360px;
        }

        .btn-analyze {
            width: 100%;
            margin-top: 18px;
            background: linear-gradient(135deg, #3b82f6, #6366f1);
            border: none;
            border-radius: 10px;
            color: white;
            padding: 14px;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px var(--accent-glow);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn-analyze:hover {
            filter: brightness(1.1);
            transform: translateY(-1px);
        }

        .btn-analyze:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Results Card */
        .results-box {
            display: none;
            animation: fadeIn 0.4s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .verdict-banner {
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 24px;
            border: 1px solid transparent;
        }

        .verdict-both {
            background: rgba(239, 68, 68, 0.15);
            border-color: var(--danger);
            box-shadow: 0 0 30px var(--danger-glow);
        }

        .verdict-visual {
            background: rgba(168, 85, 247, 0.15);
            border-color: var(--purple);
            box-shadow: 0 0 30px var(--purple-glow);
        }

        .verdict-audio {
            background: rgba(245, 158, 11, 0.15);
            border-color: var(--warning);
            box-shadow: 0 0 30px var(--warning-glow);
        }

        .verdict-real {
            background: rgba(16, 185, 129, 0.15);
            border-color: var(--success);
            box-shadow: 0 0 30px var(--success-glow);
        }

        .taxonomy-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .verdict-title {
            font-size: 1.7rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .verdict-subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
            max-width: 700px;
            margin: 0 auto;
        }

        /* Dual Gauges Grid */
        .gauges-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }

        @media (max-width: 640px) {
            .gauges-grid { grid-template-columns: 1fr; }
        }

        .gauge-card {
            background: rgba(11, 15, 25, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }

        .gauge-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .meter-bar {
            height: 12px;
            background: #1e293b;
            border-radius: 9999px;
            overflow: hidden;
            display: flex;
        }

        .meter-fill-fake {
            background: linear-gradient(90deg, #f87171, #ef4444);
            height: 100%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .meter-fill-real {
            background: linear-gradient(90deg, #34d399, #10b981);
            height: 100%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }

        .stat-item {
            background: rgba(11, 15, 25, 0.5);
            padding: 12px 16px;
            border-radius: 10px;
            border: 1px solid var(--border);
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .stat-value {
            font-size: 1.15rem;
            font-weight: 700;
        }

        /* Temporal Timeline Visualization */
        .timeline-section {
            background: rgba(11, 15, 25, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 24px;
        }

        .timeline-bar-container {
            position: relative;
            height: 36px;
            background: #1e293b;
            border-radius: 8px;
            margin: 14px 0 10px;
            overflow: hidden;
            display: flex;
            align-items: center;
        }

        .timeline-segment {
            position: absolute;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            color: white;
            transition: all 0.2s;
            cursor: pointer;
        }

        .timeline-seg-audio {
            background: rgba(245, 158, 11, 0.85);
            border-left: 2px solid #fbbf24;
            border-right: 2px solid #fbbf24;
        }

        .timeline-seg-visual {
            background: rgba(239, 68, 68, 0.85);
            border-left: 2px solid #f87171;
            border-right: 2px solid #f87171;
        }

        .timeline-legend {
            display: flex;
            gap: 16px;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 8px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }

        /* Forensic Cards Grid */
        .forensics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        @media (max-width: 768px) {
            .forensics-grid { grid-template-columns: 1fr; }
        }

        .forensic-card {
            background: rgba(11, 15, 25, 0.5);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
        }

        .forensic-card h4 {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            padding: 6px 0;
            border-bottom: 1px solid rgba(30, 44, 74, 0.5);
        }

        .metric-row:last-child {
            border-bottom: none;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            text-align: center;
            padding: 24px;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        footer a {
            color: var(--accent);
            text-decoration: none;
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
            Multimodal Deepfake Detector
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <a href="/docs" target="_blank" class="badge" style="text-decoration: none;">API Docs</a>
            <span class="badge" id="deviceBadge">Loading...</span>
        </div>
    </header>

    <div class="container">
        <div class="hero">
            <h1>Audio-Visual Deepfake Detection & Localization</h1>
            <p>Unified multimodal inspection powered by <strong>Wav2Vec2</strong> speech transformers and <strong>AV-Deepfake1M</strong> computer vision benchmarks.</p>
        </div>

        <!-- Reference Benchmarks -->
        <div class="samples-bar">
            <span style="color: var(--text-muted); font-size: 0.85rem;">Reference Benchmarks:</span>
            <button class="sample-chip" onclick="loadSample('real_human_voice.wav', 'Verified Real Voice')">
                <span>🎧</span> Real Voice (WAV)
            </button>
            <button class="sample-chip" onclick="loadSample('ai_voice_clone.wav', 'Verified AI Voice')">
                <span>🤖</span> Voice Clone (WAV)
            </button>
        </div>

        <div class="card">
            <div class="controls-row">
                <div>
                    <label style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 4px;">
                        Forensic Threshold Calibration:
                    </label>
                    <select id="modeSelect" class="mode-select">
                        <option value="mobile" selected>🛡️ Real-World / Mobile Mode (Audio 85%, Visual 65%)</option>
                        <option value="balanced">⚖️ Balanced Forensic Mode (Audio 75%, Visual 60%)</option>
                        <option value="strict">🔍 Strict Binary Mode (Audio 50%, Visual 50%)</option>
                    </select>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-muted); max-width: 420px;">
                    Supports both pure audio (WAV, MP3, FLAC, M4A) and full video media (MP4, WebM, AVI, MOV). Automatically executes visual and acoustic forensics.
                </div>
            </div>

            <div class="tabs">
                <button class="tab-btn active" id="tabUpload" onclick="switchTab('upload')">Upload Media (Audio or Video)</button>
                <button class="tab-btn" id="tabRecord" onclick="switchTab('record')">Record Voice Live (WAV)</button>
            </div>

            <!-- Upload Area -->
            <div id="uploadBox">
                <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                    </svg>
                    <p><strong>Click to browse</strong> or drag & drop a file</p>
                    <span>Supports Video (MP4, WebM, AVI, MOV, MKV) and Audio (WAV, MP3, FLAC, OGG, M4A)</span>
                </div>
                <input type="file" id="fileInput" accept="audio/*,video/*" style="display: none;">
            </div>

            <!-- Mic Box -->
            <div id="micBox" class="mic-box">
                <button class="mic-btn" id="micBtn" onclick="toggleRecording()">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="23"/>
                        <line x1="8" y1="23" x2="16" y2="23"/>
                    </svg>
                </button>
                <div id="recordStatus" style="font-weight: 600;">Click to start recording</div>
                <div id="recordTimer" style="color: var(--text-muted); font-size: 0.9rem; margin-top: 4px;">00:00 (Speak naturally for 3-5 seconds)</div>
            </div>

            <!-- Preview Box -->
            <div class="preview-box" id="previewBox">
                <div class="preview-header">
                    <strong id="previewName">file.mp4</strong>
                    <span id="previewSize" style="color: var(--text-muted)"></span>
                </div>
                <video id="videoPlayer" controls style="display: none;"></video>
                <audio id="audioPlayer" controls style="display: none;"></audio>
            </div>

            <button class="btn-analyze" id="analyzeBtn" onclick="runAnalysis()" disabled>
                <span>Run Multimodal Forensic Analysis</span>
            </button>
        </div>

        <!-- Results Card -->
        <div class="card results-box" id="resultCard">
            <!-- Verdict Banner -->
            <div class="verdict-banner" id="verdictBanner">
                <div class="taxonomy-badge" id="taxonomyBadge">AV-Deepfake1M: REAL</div>
                <div class="verdict-title" id="verdictTitle">Authentic Media</div>
                <div class="verdict-subtitle" id="verdictSubtitle">Authentic video frames & genuine speech verified.</div>
            </div>

            <!-- Dual Gauges -->
            <div class="gauges-grid">
                <div class="gauge-card">
                    <div class="gauge-header">
                        <span>🎙️ Speech Authenticity (Wav2Vec2)</span>
                        <span id="audioScoreText">--</span>
                    </div>
                    <div class="meter-bar">
                        <div class="meter-fill-fake" id="audioFakeBar" style="width: 0%"></div>
                        <div class="meter-fill-real" id="audioRealBar" style="width: 100%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 6px; color: var(--text-muted);">
                        <span id="audioFakeLabel">Fake: 0%</span>
                        <span id="audioRealLabel">Real: 100%</span>
                    </div>
                </div>

                <div class="gauge-card">
                    <div class="gauge-header">
                        <span>👁️ Visual / Facial Authenticity (CV)</span>
                        <span id="visualScoreText">--</span>
                    </div>
                    <div class="meter-bar">
                        <div class="meter-fill-fake" id="visualFakeBar" style="width: 0%"></div>
                        <div class="meter-fill-real" id="visualRealBar" style="width: 100%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 6px; color: var(--text-muted);">
                        <span id="visualFakeLabel">Fake: 0%</span>
                        <span id="visualRealLabel">Real: 100%</span>
                    </div>
                </div>
            </div>

            <!-- Stats Grid -->
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">AV-Deepfake1M Class</div>
                    <div class="stat-value" id="statClass" style="color: var(--accent);">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Overall Confidence</div>
                    <div class="stat-value" id="statConfidence">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Media Type</div>
                    <div class="stat-value" id="statMediaType">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Duration / Latency</div>
                    <div class="stat-value" id="statDurationLatency">--</div>
                </div>
            </div>

            <!-- Temporal Timeline Section -->
            <div class="timeline-section" id="timelineSection">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 0.95rem;">⏱️ Temporal Tampering Timeline Localization</strong>
                    <span id="timelineStats" style="font-size: 0.8rem; color: var(--text-muted);">0 fake segments detected</span>
                </div>
                <div class="timeline-bar-container" id="timelineBar">
                    <!-- Dynamic segments injected here -->
                </div>
                <div class="timeline-legend">
                    <div class="legend-item">
                        <div class="legend-dot" style="background: rgba(245, 158, 11, 0.85);"></div>
                        <span>Audio Tampering Segment</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: rgba(239, 68, 68, 0.85);"></div>
                        <span>Visual / Facial Tampering Segment</span>
                    </div>
                </div>
            </div>

            <!-- Detailed Forensics -->
            <div class="forensics-grid">
                <div class="forensic-card">
                    <h4><span>🎙️</span> Acoustic Forensics (Speech Analysis)</h4>
                    <div class="metric-row">
                        <span>Acoustic Prediction</span>
                        <strong id="mAudioPred">--</strong>
                    </div>
                    <div class="metric-row">
                        <span>Telephony / 4kHz Cutoff</span>
                        <strong id="mAudioBandwidth">Normal Wideband</strong>
                    </div>
                    <div class="metric-row">
                        <span>Active Speech Ratio</span>
                        <strong id="mAudioSpeech">--</strong>
                    </div>
                    <div class="metric-row">
                        <span>Audio Fake Segments</span>
                        <strong id="mAudioSegs">0</strong>
                    </div>
                </div>

                <div class="forensic-card">
                    <h4><span>👁️</span> Computer Vision Forensics (Visual Analysis)</h4>
                    <div class="metric-row">
                        <span>Visual Prediction</span>
                        <strong id="mVisualPred">--</strong>
                    </div>
                    <div class="metric-row">
                        <span>Faces / Frames Analyzed</span>
                        <strong id="mVisualFrames">--</strong>
                    </div>
                    <div class="metric-row">
                        <span>Blending Discontinuity</span>
                        <strong id="mVisualBoundary">--</strong>
                    </div>
                    <div class="metric-row">
                        <span>2D FFT High-Freq Anomaly</span>
                        <strong id="mVisualFFT">--</strong>
                    </div>
                    <div class="metric-row">
                        <span>Temporal Motion Flicker</span>
                        <strong id="mVisualFlicker">--</strong>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer>
        Unified Audio-Visual Deepfake Detector &bull; Powered by <strong>Wav2Vec2</strong> &amp; <strong>AV-Deepfake1M</strong> &bull; FastAPI Backend
    </footer>

    <script>
        let currentFile = null;
        let isRecording = false;
        let recordInterval = null;
        let recordSeconds = 0;
        let audioContext = null;
        let scriptProcessor = null;
        let micStream = null;
        let pcmBuffers = [];

        fetch('/api/health')
            .then(res => res.json())
            .then(data => {
                document.getElementById('deviceBadge').innerText = 'Ready (' + data.device.toUpperCase() + ')';
            })
            .catch(() => {
                document.getElementById('deviceBadge').innerText = 'Offline';
            });

        function switchTab(tab) {
            document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
            document.getElementById('tabRecord').classList.toggle('active', tab === 'record');
            document.getElementById('uploadBox').style.display = tab === 'upload' ? 'block' : 'none';
            document.getElementById('micBox').style.display = tab === 'record' ? 'block' : 'none';
        }

        async function loadSample(filename, label) {
            try {
                const btn = document.getElementById('analyzeBtn');
                btn.disabled = true;
                btn.innerHTML = '<span class="spinner"></span> <span>Loading sample...</span>';

                const res = await fetch(`/samples/${filename}`);
                if (!res.ok) throw new Error('Sample fetch failed');
                const blob = await res.blob();
                
                currentFile = new File([blob], filename, { type: filename.endsWith('.mp4') ? 'video/mp4' : 'audio/wav' });
                displayPreview(currentFile);

                btn.disabled = false;
                btn.innerHTML = '<span>Run Multimodal Forensic Analysis</span>';
                runAnalysis();
            } catch (err) {
                alert('Could not load demo sample: ' + err.message);
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('analyzeBtn').innerHTML = '<span>Run Multimodal Forensic Analysis</span>';
            }
        }

        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');

        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length) {
                handleSelectedFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleSelectedFile(e.target.files[0]);
            }
        });

        function handleSelectedFile(file) {
            currentFile = file;
            displayPreview(file);
            document.getElementById('analyzeBtn').disabled = false;
        }

        function displayPreview(file) {
            document.getElementById('previewName').innerText = file.name;
            document.getElementById('previewSize').innerText = (file.size / 1024).toFixed(1) + ' KB';
            
            const isVid = file.type.startsWith('video/') || /\\.(mp4|webm|avi|mov|mkv)$/i.test(file.name);
            const videoPlayer = document.getElementById('videoPlayer');
            const audioPlayer = document.getElementById('audioPlayer');

            if (isVid) {
                videoPlayer.src = URL.createObjectURL(file);
                videoPlayer.style.display = 'block';
                audioPlayer.style.display = 'none';
            } else {
                audioPlayer.src = URL.createObjectURL(file);
                audioPlayer.style.display = 'block';
                videoPlayer.style.display = 'none';
            }
            document.getElementById('previewBox').style.display = 'block';
        }

        async function toggleRecording() {
            const micBtn = document.getElementById('micBtn');
            const status = document.getElementById('recordStatus');
            const timer = document.getElementById('recordTimer');

            if (!isRecording) {
                try {
                    micStream = await navigator.mediaDevices.getUserMedia({
                        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
                    });

                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const source = audioContext.createMediaStreamSource(micStream);
                    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                    pcmBuffers = [];

                    scriptProcessor.onaudioprocess = (e) => {
                        if (!isRecording) return;
                        const channelData = e.inputBuffer.getChannelData(0);
                        pcmBuffers.push(new Float32Array(channelData));
                    };

                    source.connect(scriptProcessor);
                    scriptProcessor.connect(audioContext.destination);

                    isRecording = true;
                    micBtn.classList.add('recording');
                    status.innerText = "Recording voice... Click to stop";
                    recordSeconds = 0;
                    recordInterval = setInterval(() => {
                        recordSeconds++;
                        const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                        const s = String(recordSeconds % 60).padStart(2, '0');
                        timer.innerText = `${m}:${s} (Speak naturally for 3-5s)`;
                    }, 1000);

                } catch (err) {
                    alert("Microphone access failed: " + err.message);
                }
            } else {
                isRecording = false;
                micBtn.classList.remove('recording');
                status.innerText = "Encoding lossless WAV...";
                clearInterval(recordInterval);

                if (scriptProcessor) scriptProcessor.disconnect();
                if (micStream) micStream.getTracks().forEach(t => t.stop());

                let totalSamples = 0;
                for (let b of pcmBuffers) totalSamples += b.length;
                let merged = new Float32Array(totalSamples);
                let offset = 0;
                for (let b of pcmBuffers) {
                    merged.set(b, offset);
                    offset += b.length;
                }

                const wavBlob = encodeWAV(merged, audioContext.sampleRate);
                currentFile = new File([wavBlob], "microphone_voice.wav", { type: "audio/wav" });
                displayPreview(currentFile);
                document.getElementById('analyzeBtn').disabled = false;
                status.innerText = "Voice recording ready for analysis";
            }
        }

        function encodeWAV(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * 2, true);

            let p = 44;
            for (let i = 0; i < samples.length; i++) {
                let s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                p += 2;
            }
            return new Blob([view], { type: 'audio/wav' });
        }

        function writeString(view, offset, str) {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        }

        async function runAnalysis() {
            if (!currentFile) return;

            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> <span>Running Multimodal Forensics...</span>';

            const mode = document.getElementById('modeSelect').value;
            let aThresh = 0.85;
            let vThresh = 0.65;
            if (mode === 'balanced') { aThresh = 0.75; vThresh = 0.60; }
            else if (mode === 'strict') { aThresh = 0.50; vThresh = 0.50; }

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                const response = await fetch(`/api/predict?audio_threshold=${aThresh}&visual_threshold=${vThresh}`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Prediction failed');

                displayResults(data);
            } catch (err) {
                alert("Analysis failed: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>Run Multimodal Forensic Analysis</span>';
            }
        }

        function displayResults(data) {
            const card = document.getElementById('resultCard');
            const banner = document.getElementById('verdictBanner');
            const taxBadge = document.getElementById('taxonomyBadge');
            const title = document.getElementById('verdictTitle');
            const subtitle = document.getElementById('verdictSubtitle');

            const avClass = data.overall_verdict || "real";
            taxBadge.innerText = "AV-DEEPFAKE1M: " + avClass.toUpperCase();

            title.innerText = data.verdict_title || (data.is_fake ? "Deepfake Detected" : "Authentic Media");
            subtitle.innerText = `Classification: ${avClass.replace('_', ' ').toUpperCase()} • Confidence: ${(data.overall_confidence * 100).toFixed(1)}%`;

            // Verdict banner styling based on AV-Deepfake1M taxonomy
            if (avClass === "both_modified") {
                banner.className = "verdict-banner verdict-both";
                taxBadge.style.background = "rgba(239, 68, 68, 0.3)";
                taxBadge.style.color = "#f87171";
            } else if (avClass === "visual_modified") {
                banner.className = "verdict-banner verdict-visual";
                taxBadge.style.background = "rgba(168, 85, 247, 0.3)";
                taxBadge.style.color = "#c084fc";
            } else if (avClass === "audio_modified") {
                banner.className = "verdict-banner verdict-audio";
                taxBadge.style.background = "rgba(245, 158, 11, 0.3)";
                taxBadge.style.color = "#fbbf24";
            } else {
                banner.className = "verdict-banner verdict-real";
                taxBadge.style.background = "rgba(16, 185, 129, 0.3)";
                taxBadge.style.color = "#34d399";
            }

            // Dual Gauges
            const audioData = data.audio_analysis || {};
            const visualData = data.visual_analysis || {};

            if (audioData.probabilities) {
                const aFake = (audioData.probabilities.fake * 100).toFixed(1);
                const aReal = (audioData.probabilities.real * 100).toFixed(1);
                document.getElementById('audioScoreText').innerText = `${aReal}% Authentic`;
                document.getElementById('audioFakeBar').style.width = aFake + '%';
                document.getElementById('audioRealBar').style.width = aReal + '%';
                document.getElementById('audioFakeLabel').innerText = `Fake: ${aFake}%`;
                document.getElementById('audioRealLabel').innerText = `Real: ${aReal}%`;
            } else {
                document.getElementById('audioScoreText').innerText = "Silent / N/A";
                document.getElementById('audioFakeBar').style.width = "0%";
                document.getElementById('audioRealBar').style.width = "100%";
            }

            if (visualData.probabilities) {
                const vFake = (visualData.probabilities.fake * 100).toFixed(1);
                const vReal = (visualData.probabilities.real * 100).toFixed(1);
                document.getElementById('visualScoreText').innerText = `${vReal}% Authentic`;
                document.getElementById('visualFakeBar').style.width = vFake + '%';
                document.getElementById('visualRealBar').style.width = vReal + '%';
                document.getElementById('visualFakeLabel').innerText = `Fake: ${vFake}%`;
                document.getElementById('visualRealLabel').innerText = `Real: ${vReal}%`;
            } else {
                document.getElementById('visualScoreText').innerText = "N/A (Audio Only)";
                document.getElementById('visualFakeBar').style.width = "0%";
                document.getElementById('visualRealBar').style.width = "100%";
            }

            // Stats grid
            document.getElementById('statClass').innerText = avClass.toUpperCase().replace('_', ' ');
            document.getElementById('statConfidence').innerText = (data.overall_confidence * 100).toFixed(1) + '%';
            document.getElementById('statMediaType').innerText = data.media_type.toUpperCase();
            document.getElementById('statDurationLatency').innerText = `${data.duration_seconds}s (${data.inference_time_seconds}s latency)`;

            // Temporal Timeline Localization Rendering
            const bar = document.getElementById('timelineBar');
            bar.innerHTML = '';
            const totalDur = Math.max(1.0, data.duration_seconds || 1.0);
            const events = data.timeline_events || [];

            document.getElementById('timelineStats').innerText = `${(data.fake_segments || []).length} localized tampering segment(s)`;

            if (events.length > 0) {
                events.forEach(ev => {
                    const leftPct = (ev.start / totalDur) * 100;
                    const widthPct = Math.max(2.0, ((ev.end - ev.start) / totalDur) * 100);
                    const segEl = document.createElement('div');
                    segEl.className = 'timeline-segment ' + (ev.modality === 'audio' ? 'timeline-seg-audio' : 'timeline-seg-visual');
                    segEl.style.left = leftPct + '%';
                    segEl.style.width = widthPct + '%';
                    segEl.title = `[${ev.start}s - ${ev.end}s] ${ev.description}`;
                    segEl.innerText = `${ev.start}s`;
                    bar.appendChild(segEl);
                });
            } else {
                bar.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem; margin-left: 14px;">No temporal tampering detected (Consistent Authentic Stream)</span>';
            }

            // Forensic breakdown details
            document.getElementById('mAudioPred').innerText = (audioData.prediction || "N/A").toUpperCase();
            if (audioData.bandwidth_analysis && audioData.bandwidth_analysis.is_narrowband) {
                document.getElementById('mAudioBandwidth').innerText = "📞 Narrowband (<4kHz)";
                document.getElementById('mAudioBandwidth').style.color = "var(--warning)";
            } else {
                document.getElementById('mAudioBandwidth').innerText = "Normal Wideband";
                document.getElementById('mAudioBandwidth').style.color = "var(--success)";
            }

            if (audioData.speech_ratio !== undefined) {
                document.getElementById('mAudioSpeech').innerText = `${Math.round(audioData.speech_ratio * 100)}% (${audioData.active_speech_seconds}s)`;
            } else {
                document.getElementById('mAudioSpeech').innerText = "N/A";
            }
            document.getElementById('mAudioSegs').innerText = (data.audio_fake_segments || []).length;

            document.getElementById('mVisualPred').innerText = (visualData.prediction || "N/A").toUpperCase();
            if (visualData.frames_analyzed !== undefined) {
                document.getElementById('mVisualFrames').innerText = `${visualData.faces_detected || 0} faces in ${visualData.frames_analyzed} frames`;
            } else {
                document.getElementById('mVisualFrames').innerText = "N/A";
            }

            const vMetrics = visualData.forensic_metrics || {};
            document.getElementById('mVisualBoundary').innerText = vMetrics.boundary_artifact_score !== undefined ? `${(vMetrics.boundary_artifact_score * 100).toFixed(1)}%` : "N/A";
            document.getElementById('mVisualFFT').innerText = vMetrics.fft_frequency_score !== undefined ? `${(vMetrics.fft_frequency_score * 100).toFixed(1)}%` : "N/A";
            document.getElementById('mVisualFlicker').innerText = vMetrics.temporal_flicker_score !== undefined ? `${(vMetrics.temporal_flicker_score * 100).toFixed(1)}%` : "N/A";

            card.style.display = 'block';
            card.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
