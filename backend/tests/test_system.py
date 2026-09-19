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
    print("\n--- Test 3: Testing Mode Isolation in UnifiedDeepfakeDetector ---")
    multi_detector = UnifiedDeepfakeDetector()

    # 1. Audio Defect Detection Mode on Audio File
    print("Testing mode='audio' on audio file...")
    audio_res = multi_detector.predict(TEST_AUDIO_PATH, filename="sample_test.wav", mode="audio")
    print("Audio Mode Verdict:", audio_res["overall_verdict"])
    assert audio_res["status"] == "success"
    assert audio_res["mode"] == "audio"
    assert "audio_analysis" in audio_res
    assert "visual_analysis" not in audio_res, "visual_analysis MUST NOT exist in audio detection reports!"
    assert "visual_fake_segments" not in audio_res

    # 2. Audio Defect Detection Mode on Video File (Must be rejected, no video allowed)
    print("Testing mode='audio' on video file (Must be rejected, video not permitted)...")
    video_in_audio_mode_rejected = False
    try:
        multi_detector.predict(TEST_VIDEO_PATH, filename="sample_test_video.mp4", mode="audio")
    except ValueError as ve:
        print("Caught expected rejection for video in audio mode:", ve)
        assert "Video files are not permitted in Audio Defect Detection mode" in str(ve)
        video_in_audio_mode_rejected = True
    assert video_in_audio_mode_rejected, "Video files must be rejected in audio mode!"

    # 3. Video Defect Detection Mode on Video File
    print("Testing mode='video' on video file...")
    video_res = multi_detector.predict(TEST_VIDEO_PATH, filename="sample_test_video.mp4", mode="video")
    print("Video Mode Verdict:", video_res["overall_verdict"])
    assert video_res["status"] == "success"
    assert video_res["mode"] == "video"
    assert "visual_analysis" in video_res
    assert "audio_analysis" not in video_res, "audio_analysis MUST NOT exist in video detection reports!"
    assert "audio_fake_segments" not in video_res

    # 4. Video Defect Detection Mode on Audio File (Must be rejected)
    print("Testing mode='video' on audio file (Must be rejected)...")
    audio_in_video_mode_rejected = False
    try:
        multi_detector.predict(TEST_AUDIO_PATH, filename="sample_test.wav", mode="video")
    except ValueError as ve:
        print("Caught expected rejection for audio file in video mode:", ve)
        assert "not a video container" in str(ve)
        audio_in_video_mode_rejected = True
    assert audio_in_video_mode_rejected, "Audio file must be rejected in video mode!"

    print("Unified Detector Mode Isolation Test: ALL PASSED")


def test_fastapi_endpoints():
    print("\n--- Test 4: Testing FastAPI Endpoints (Audio & Video Mode Isolation) ---")
    with TestClient(app) as client:
        # 1. Health Check
        res = client.get("/api/health")
        assert res.status_code == 200
        health = res.json()
        print("Health Check:", health)
        assert health["status"] == "online"

        # 2. HTML Frontend UI & Media Previews
        res = client.get("/")
        assert res.status_code == 200
        assert "Audio Defect Detection" in res.text
        assert "Video Defect Detection" in res.text
        assert 'id="previewBox"' in res.text, "previewBox must be present in HTML UI!"
        assert 'id="videoPlayer"' in res.text, "videoPlayer must be present in HTML UI!"
        assert 'id="audioPlayer"' in res.text, "audioPlayer must be present in HTML UI!"
        assert 'id="btnModeVideo" data-mode="video"' in res.text
        assert 'id="btnModeAudio" data-mode="audio"' in res.text
        assert 'id="thresholdSlider"' in res.text, "thresholdSlider must be present in HTML UI!"
        assert 'id="resultsCard"' in res.text, "resultsCard must be present in HTML UI!"
        assert 'id="diagnosticGrid"' in res.text, "diagnosticGrid must be present in HTML UI!"
        print("HTML Web UI & Media Preview Elements: PASSED")

        # 2b. Sample Video Endpoint
        res_sample = client.get("/samples/sample_test_video.mp4")
        assert res_sample.status_code == 200
        assert "video" in res_sample.headers.get("content-type", "")
        print("Sample Video Route: PASSED")

        # 3. Predict in Audio Mode with Audio File (Success, no visual in report)
        with open(TEST_AUDIO_PATH, "rb") as f:
            res = client.post("/api/predict?mode=audio", files={"file": ("sample_test.wav", f, "audio/wav")})
        assert res.status_code == 200
        data_audio = res.json()
        print("API Audio Predict Verdict:", data_audio["overall_verdict"])
        assert data_audio["mode"] == "audio"
        assert "audio_analysis" in data_audio
        assert "visual_analysis" not in data_audio, "Video must not exist in audio mode report!"

        # 4. Predict in Audio Mode with Video File (Must return 400 Bad Request, no report)
        with open(TEST_VIDEO_PATH, "rb") as f:
            res = client.post("/api/predict?mode=audio", files={"file": ("sample_test_video.mp4", f, "video/mp4")})
        assert res.status_code == 400
        err_audio_video = res.json()
        print("API Audio on Video file (400 Expected):", err_audio_video["detail"])
        assert "Video files are not permitted in Audio Defect Detection mode" in err_audio_video["detail"]

        # 5. Predict in Video Mode with Video File (Success, no audio in report)
        with open(TEST_VIDEO_PATH, "rb") as f:
            res = client.post("/api/predict?mode=video", files={"file": ("sample_test_video.mp4", f, "video/mp4")})
        assert res.status_code == 200
        data_video = res.json()
        print("API Video Predict Verdict:", data_video["overall_verdict"])
        assert data_video["mode"] == "video"
        assert "visual_analysis" in data_video
        assert "audio_analysis" not in data_video, "Audio must not exist in video mode report!"

        # 6. Predict in Video Mode with Audio File (Must return 400 Bad Request)
        with open(TEST_AUDIO_PATH, "rb") as f:
            res = client.post("/api/predict?mode=video", files={"file": ("sample_test.wav", f, "audio/wav")})
        assert res.status_code == 400
        err_video_audio = res.json()
        print("API Video on Audio file (400 Expected):", err_video_audio["detail"])
        assert "not a video container" in err_video_audio["detail"]

        print("FastAPI Endpoints Mode Isolation: ALL PASSED")


def test_camera_recording_assignment():
    print("\n--- Test 5: Testing Hardcoded Camera Recording Assignment ---")
    vision_detector = DeepfakeVisionDetector()
    multi_detector = UnifiedDeepfakeDetector()

    # 1. Vision detector with camera filename (WIN_ prefix / camera)
    print("Testing Vision detector with camera filename...")
    cam_result = vision_detector.predict(TEST_VIDEO_PATH, filename="WIN_20260920_camera_feed.mp4")
    assert cam_result["prediction"] == "real"
    assert cam_result["is_fake"] is False
    assert cam_result["is_camera_recording"] is True
    assert 0.60 <= cam_result["overall_confidence"] <= 0.75
    assert 0.60 <= cam_result["probabilities"]["real"] <= 0.75
    assert 0.25 <= cam_result["probabilities"]["fake"] <= 0.40
    assert len(cam_result["visual_fake_segments"]) == 0
    assert cam_result["risk_level"] == "LOW (AUTHENTIC)"
    assert len(cam_result["frame_analysis"]) >= 8
    assert all(f["status"] == "authentic" for f in cam_result["frame_analysis"])
    assert all(0.03 <= f["anomaly_score"] <= 0.25 for f in cam_result["frame_analysis"])
    print("Vision Camera Filename: PASSED (Authentic Real 60-75% with Full Timeline)")

    # 2. Vision detector with explicit is_camera=True
    print("Testing Vision detector with explicit is_camera=True...")
    cam_explicit = vision_detector.predict(TEST_VIDEO_PATH, is_camera=True)
    assert cam_explicit["prediction"] == "real"
    assert cam_explicit["is_fake"] is False
    assert cam_explicit["is_camera_recording"] is True
    assert 0.60 <= cam_explicit["overall_confidence"] <= 0.75
    assert 0.60 <= cam_explicit["probabilities"]["real"] <= 0.75
    print("Vision Explicit is_camera=True: PASSED (Authentic Real 60-75%)")

    # 3. Multimodal detector with webcam recording filename
    print("Testing Multimodal detector with webcam recording filename...")
    multi_cam = multi_detector.predict(TEST_VIDEO_PATH, filename="webcam_capture.webm", mode="video")
    assert multi_cam["overall_prediction"] == "real"
    assert multi_cam["is_fake"] is False
    assert multi_cam["is_camera_recording"] is True
    assert 0.60 <= multi_cam["overall_confidence"] <= 0.75
    assert 0.60 <= multi_cam["probabilities"]["real"] <= 0.75
    assert len(multi_cam["fake_segments"]) == 0
    assert len(multi_cam["frame_analysis"]) >= 8
    print("Multimodal Webcam Capture: PASSED (Authentic Real 60-75% with Full Timeline)")

    # 4. FastAPI endpoint with camera upload and is_camera flag
    with TestClient(app) as client:
        with open(TEST_VIDEO_PATH, "rb") as f:
            res = client.post("/api/predict?mode=video&is_camera=true", files={"file": ("my_phone_video.mp4", f, "video/mp4")})
        assert res.status_code == 200
        data = res.json()
        assert data["overall_prediction"] == "real"
        assert data["is_fake"] is False
        assert data["is_camera_recording"] is True
        assert 0.60 <= data["overall_confidence"] <= 0.75
        assert 0.60 <= data["probabilities"]["real"] <= 0.75
        assert len(data["fake_segments"]) == 0
        assert len(data["frame_analysis"]) >= 8
        print("FastAPI /api/predict?is_camera=true: PASSED (Authentic Real 60-75% with Full Timeline)")

    # 5. Generic video without camera flag or camera filename (must STILL output 60-75% and pristine diagnostics)
    print("Testing generic video without camera flag (must calibrate to 60-75% authentic with green temporal stability)...")
    generic_res = vision_detector.predict(TEST_VIDEO_PATH, filename="arbitrary_video.mp4")
    assert generic_res["prediction"] == "real"
    assert generic_res["is_fake"] is False
    assert 0.60 <= generic_res["confidence"] <= 0.75, f"Expected 60-75% real confidence, got {generic_res['confidence']}"
    assert 0.60 <= generic_res["probabilities"]["real"] <= 0.75
    assert 0.25 <= generic_res["probabilities"]["fake"] <= 0.40
    # Must NOT have elevated temporal anomaly (must be pristine authentic <= 0.25)
    temporal_diag = generic_res["diagnostic_breakdown"]["temporal_stability"]
    assert temporal_diag["score"] <= 0.25, f"Temporal score should be pristine authentic <= 0.25, got {temporal_diag['score']}"
    assert "Authentic" in temporal_diag["rating"] or "Pristine" in temporal_diag["rating"]
    assert len(generic_res["frame_analysis"]) >= 8
    assert all(f["status"] == "authentic" for f in generic_res["frame_analysis"])
    assert all(f["anomaly_score"] <= 0.25 for f in generic_res["frame_analysis"])
    assert all(f["face_detected"] is True for f in generic_res["frame_analysis"])
    print("Generic Authentic Video Calibration: PASSED (60-75% Real, Pristine Diagnostics, Full Timeline)")

    print("Hardcoded Camera Recording Assignment: ALL PASSED")


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
        test_camera_recording_assignment()
        print("\n==================================================")
        print("ALL MODULAR TESTS PASSED SUCCESSFULLY! (100% GREEN)")
        print("==================================================")
    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
