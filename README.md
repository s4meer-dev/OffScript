# Multimodal Audio-Visual Deepfake Detection & Localization Platform 🎙️👁️🔍

Production-grade, unified deepfake detection and temporal tampering localization integrating **Wav2Vec2** speech self-supervised representations with the **AV-Deepfake1M** computer vision benchmark.

Distinguishes between **authentic human media** and **AI-manipulated content** (synthetic speech, voice cloning, face swaps, lip-sync tampering) across standalone audio and full video streams.

---

## 🚀 Key Capabilities & Cross-Project Integration

### 1. AV-Deepfake1M Standard 4-Class Taxonomy
Predictions are mapped to the ACM Multimedia benchmark classification standard:
- **`real`**: Authentic video frames and genuine human voice.
- **`audio_modified`**: AI voice clone / synthetic speech with authentic video.
- **`visual_modified`**: Face swap / visual deepfake with genuine speech.
- **`both_modified`**: Full multimodal deepfake (both video and audio manipulated).

### 2. Multi-Level Computer Vision Forensics (from `AV-Deepfake1M` & BA-TFD)
- **Facial Localization & Tracking**: Extracts primary face bounding box across video tracks.
- **Spatial Boundary Blending Artifacts**: Evaluates Laplacian gradient variance across inner facial features and outer seam perimeters to catch compositing seams.
- **2D FFT Frequency Inspection**: Detects high-frequency periodic grid artifacts left by GAN and diffusion generators.
- **Temporal Motion Flicker Consistency**: Measures consecutive frame identity continuity and motion jitter inspired by C3D and BA-TFD.
- **Deep Feature Representation**: Torchvision deep backbone evaluating facial manifold anomalies.

### 3. Acoustic Speech Forensics (from Wav2Vec2 Engine)
- **Speech Classifier**: Sequence classification powered by `mo-thecreator/Deepfake-audio-detection` (Wav2Vec2 architecture, ~94.5M parameters).
- **Acoustic Preprocessing**: Dynamic silence trimming, peak volume normalization, and stereo-to-mono 16kHz resampling.
- **Bandwidth & Codec Cutoff**: Detects telephony / WhatsApp narrowband audio (<4kHz cutoff) to prevent acoustic false alarms.

### 4. Temporal Tampering Localization
- Pinpoints localized fake intervals `[start_s, end_s]` for both audio (`audio_fake_segments`) and visual tracks (`visual_fake_segments`), merging them into unified `fake_segments`.

### 5. Production Interfaces
- **Interactive Web Dashboard** (FastAPI): Drag-and-drop supporting audio and video, HTML5 video player, audio waveform preview, dual authenticity gauges, interactive temporal timeline, and forensic breakdown cards.
- **REST API**: Standardized JSON API at `/api/predict`, health status at `/api/health`, and model specifications at `/api/info` with Swagger UI at `/docs`.
- **Gradio Interface** (`app_gradio.py`): Streamlined UI with confidence sliders and metric inspection.
- **CLI Utility** (`predict_cli.py`): Terminal testing with structured forensic reports.

---

## 📁 Project Architecture

```
f:\DeepFake\
├── model/                     # Local Wav2Vec2 weights and configs
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── AV-Deepfake1M/             # Benchmark dataset & research models (Xception, BA-TFD)
│   ├── examples/
│   │   ├── batfd/             # Boundary-Aware Temporal Face Deepfake models (C3D, MViT)
│   │   └── xception/          # XceptionNet video frame classifier
│   └── ...
├── detector.py                # Core Wav2Vec2 audio detection engine
├── vision_detector.py         # Computer Vision facial & temporal artifact detector
├── multimodal_detector.py     # Unified Audio-Visual fusion & localization coordinator
├── server.py                  # FastAPI server & interactive Web Dashboard
├── app_gradio.py              # Multimodal Gradio interface
├── predict_cli.py             # Multimodal CLI tool
├── test_system.py             # Automated test suite (audio, video, API)
├── start_server.bat           # 1-click Windows launcher for FastAPI
├── start_gradio.bat           # 1-click Windows launcher for Gradio
├── requirements.txt           # Python package dependencies
└── README.md                  # System documentation
```

---

## 🛠️ Quick Start

### 1. Launch the FastAPI Web Dashboard & REST API
```bash
python server.py
# or double-click start_server.bat
```
Navigate to:
- **Interactive Dashboard**: [http://localhost:8000](http://localhost:8000)
- **OpenAPI / Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Launch the Gradio Interface
```bash
python app_gradio.py
# or double-click start_gradio.bat
```
Navigate to [http://localhost:7860](http://localhost:7860).

### 3. Command-Line Prediction
```bash
# Audio classification:
python predict_cli.py samples\real_human_voice.wav

# Video classification:
python predict_cli.py sample_test_video.mp4
```

### 4. Run Automated Test Suite
```bash
python test_system.py
```

---

## 📡 REST API Reference

### `POST /api/predict`
Upload an audio or video file for classification and temporal localization.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
     -F "file=@sample_test_video.mp4"
```

**Response JSON:**
```json
{
  "status": "success",
  "media_type": "video",
  "overall_verdict": "real",
  "overall_prediction": "real",
  "overall_confidence": 0.7481,
  "is_fake": false,
  "verdict_title": "Authentic Video (Silent Media)",
  "av_deepfake1m_classification": {
    "label": "real",
    "description": "Authentic Video (Silent Media)",
    "taxonomy": ["real", "audio_modified", "visual_modified", "both_modified"]
  },
  "audio_analysis": {
    "status": "no_audio_stream",
    "prediction": "silent"
  },
  "visual_analysis": {
    "status": "success",
    "prediction": "real",
    "confidence": 0.7481,
    "frames_analyzed": 10,
    "faces_detected": 10,
    "forensic_metrics": {
      "boundary_artifact_score": 0.197,
      "fft_frequency_score": 0.61,
      "temporal_flicker_score": 0.0
    }
  },
  "fake_segments": [],
  "timeline_events": [],
  "duration_seconds": 2.0,
  "inference_time_seconds": 0.187
}
```
