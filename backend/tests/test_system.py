"""
Comprehensive test suite for Unified Audio-Visual Deepfake Detection & Localization System.
Tests:
1. Synthetic audio generation & Wav2Vec2 detector inference.
2. Synthetic video generation with facial geometry & Computer Vision detector inference.
3. Unified multimodal fusion & AV-Deepfake1M taxonomy categorization.
4. FastAPI REST endpoints (/api/health, /api/info, /, /api/predict with audio and video).
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_models.audio.detector import DeepfakeAudioDetector
from ai_models.vision.vision_detector import DeepfakeVisionDetector
from ai_models.multimodal.multimodal_detector import UnifiedDeepfakeDetector
from backend.main import app

TEST_AUDIO_PATH = ROOT_DIR / "sample_test.wav"
TEST_VIDEO_PATH = ROOT_DIR / "sample_test_video.mp4"


def generate_synthetic_audio(path: Path, duration_sec: float = 2.0, sr: int = 44100):
    """
    Generate a synthetic multi-frequency audio file simulating human vocal harmonics.
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    envelope = np.sin(np.pi * t / duration_sec) ** 2
    audio = (audio * envelope).astype(np.float32)
    sf.write(str(path), audio, sr)
    print(f"Generated synthetic audio at: {path} ({duration_sec}s, {sr}Hz)")
    return path


def generate_synthetic_video(path: Path, duration_sec: float = 2.0, fps: float = 10.0):
    """
    Generate a synthetic video with talking facial geometry and motion.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    num_frames = int(duration_sec * fps)
    out = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))

    for i in range(num_frames):
        frame = np.full((240, 320, 3), 45, dtype=np.uint8)
        # Animated head / face oval
        cy = 120 + int(6 * np.sin(i * 0.4))
        cv2.ellipse(frame, (160, cy), (50, 65), 0, 0, 360, (140, 160, 210), -1)
        # Eyes
        cv2.circle(frame, (142, cy - 15), 6, (40, 40, 40), -1)
        cv2.circle(frame, (178, cy - 15), 6, (40, 40, 40), -1)
        # Mouth
        mouth_open = int(4 * abs(np.sin(i * 0.8)))
        cv2.ellipse(frame, (160, cy + 25), (14, 6 + mouth_open), 0, 0, 360, (50, 50, 180), -1)
        out.write(frame)

    out.release()
    print(f"Generated synthetic video at: {path} ({num_frames} frames, {fps}fps)")
    return path


def test_audio_detector():
    print("\n--- Test 1: Testing Audio Detector (Wav2Vec2) ---")
    detector = DeepfakeAudioDetector()
    result = detector.predict(TEST_AUDIO_PATH)
    print("Audio Inference Result:", result["prediction"], "Confidence:", result["confidence"])
    assert result["status"] == "success"
    assert result["prediction"] in ["fake", "real", "uncertain_ambient"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["duration_seconds"] > 0
    print("Audio Detector Test: PASSED")


def test_vision_detector():
    print("\n--- Test 2: Testing Computer Vision Detector (Facial & Temporal Analysis) ---")
    vision_detector = DeepfakeVisionDetector()
    result = vision_detector.predict(TEST_VIDEO_PATH)
    print("Vision Inference Result:", result["prediction"], "Confidence:", result["confidence"])
    print("Frames analyzed:", result["frames_analyzed"], "Faces detected:", result["faces_detected"])
    assert result["status"] == "success"
    assert result["prediction"] in ["fake", "real", "suspicious_visual"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["frames_analyzed"] > 0
    assert result["faces_detected"] > 0
    assert "visual_fake_segments" in result
    print("Vision Detector Test: PASSED")


def test_multimodal_detector():
    print("\n--- Test 3: Testing Unified Multimodal Detector (AV-Deepfake1M Taxonomy) ---")
    multi_detector = UnifiedDeepfakeDetector()

    # 1. Pure Audio
    audio_res = multi_detector.predict(TEST_AUDIO_PATH, filename="sample_test.wav")
    print("Pure Audio Multimodal Verdict:", audio_res["overall_verdict"])
    assert audio_res["status"] == "success"
    assert audio_res["media_type"] == "audio"
    assert audio_res["overall_verdict"] in ["real", "audio_modified"]
    assert "fake_segments" in audio_res

    # 2. Video Media
    video_res = multi_detector.predict(TEST_VIDEO_PATH, filename="sample_test_video.mp4")
    print("Video Multimodal Verdict:", video_res["overall_verdict"])
    assert video_res["status"] == "success"
    assert video_res["media_type"] == "video"
    assert video_res["overall_verdict"] in ["real", "audio_modified", "visual_modified", "both_modified"]
    assert "audio_analysis" in video_res
    assert "visual_analysis" in video_res
    assert "fake_segments" in video_res
    print("Unified Multimodal Test: PASSED")


def test_fastapi_endpoints():
    print("\n--- Test 4: Testing FastAPI Endpoints (Audio, Video & UI) ---")
    with TestClient(app) as client:
        # 1. Health Check
        res = client.get("/api/health")
        assert res.status_code == 200
        health = res.json()
        print("Health Check:", health)
        assert health["status"] == "online"
        assert health["multimodal_ready"] is True

        # 2. Info Check
        res = client.get("/api/info")
        assert res.status_code == 200
        info = res.json()
        print("Info Check: System Name =", info["system_name"])
        assert "audio_model" in info
        assert "visual_model" in info
        assert "taxonomy" in info

        # 3. HTML Frontend UI
        res = client.get("/")
        assert res.status_code == 200
        assert "Multimodal Deepfake Detector" in res.text
        print("HTML Web UI: PASSED")

        # 4. Predict Audio
        with open(TEST_AUDIO_PATH, "rb") as f:
            res = client.post("/api/predict", files={"file": ("sample_test.wav", f, "audio/wav")})
        assert res.status_code == 200
        data_audio = res.json()
        print("API Audio Predict:", data_audio["overall_verdict"])
        assert data_audio["media_type"] == "audio"
        assert "av_deepfake1m_classification" in data_audio

        # 5. Predict Video
        with open(TEST_VIDEO_PATH, "rb") as f:
            res = client.post("/api/predict", files={"file": ("sample_test_video.mp4", f, "video/mp4")})
        assert res.status_code == 200
        data_video = res.json()
        print("API Video Predict:", data_video["overall_verdict"])
        assert data_video["media_type"] == "video"
        assert "visual_analysis" in data_video
        assert "timeline_events" in data_video
        print("FastAPI Endpoints: ALL PASSED")


if __name__ == "__main__":
    print("==================================================")
    print("Running Modular Audio-Visual Deepfake Test Suite")
    print("==================================================")

    generate_synthetic_audio(TEST_AUDIO_PATH)
    generate_synthetic_video(TEST_VIDEO_PATH)

    try:
        test_audio_detector()
        test_vision_detector()
        test_multimodal_detector()
        test_fastapi_endpoints()
        print("\n==================================================")
        print("ALL MODULAR TESTS PASSED SUCCESSFULLY! (100% GREEN)")
        print("==================================================")
    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
