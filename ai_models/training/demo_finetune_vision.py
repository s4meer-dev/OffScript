"""
End-to-End Demo & Verification Script for Video Deepfake Fine-Tuning.
Generates a lightweight synthetic dataset, runs 1 fine-tuning epoch,
verifies model weight saving, and confirms DeepfakeVisionDetector integration.

Usage:
  python ai_models/training/demo_finetune_vision.py
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_models.training.finetune_vision import train_vision_model
from ai_models.vision.vision_detector import DeepfakeVisionDetector

DEMO_DIR = ROOT_DIR / "ai_models" / "training" / "demo_dataset"
WEIGHTS_DIR = ROOT_DIR / "ai_models" / "weights"


def create_synthetic_clip(path: Path, is_fake: bool, duration_sec: float = 1.0, fps: float = 10.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    num_frames = int(duration_sec * fps)
    out = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))

    for i in range(num_frames):
        frame = np.full((240, 320, 3), 50, dtype=np.uint8)
        cy = 120 + int(4 * np.sin(i * 0.5))
        # Head oval
        cv2.ellipse(frame, (160, cy), (48, 60), 0, 0, 360, (135, 155, 205), -1)
        # Eyes
        cv2.circle(frame, (144, cy - 12), 5, (30, 30, 30), -1)
        cv2.circle(frame, (176, cy - 12), 5, (30, 30, 30), -1)
        # Mouth
        mouth_h = int(4 * abs(np.sin(i * 0.7)))
        cv2.ellipse(frame, (160, cy + 22), (12, 5 + mouth_h), 0, 0, 360, (40, 40, 160), -1)

        if is_fake:
            # Inject visible synthetic boundary seam and periodic grid noise
            cv2.ellipse(frame, (160, cy), (50, 62), 0, 0, 360, (0, 255, 255), 2)
            noise = (np.random.randn(240, 320, 3) * 25).astype(np.uint8)
            frame = cv2.add(frame, noise)

        out.write(frame)

    out.release()


def run_demo():
    print("=" * 65)
    print("  Video Deepfake Fine-Tuning Demo & Verification")
    print("=" * 65)

    real_dir = DEMO_DIR / "real"
    fake_dir = DEMO_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Generating synthetic demo dataset...")
    for i in range(2):
        create_synthetic_clip(real_dir / f"demo_real_{i+1}.mp4", is_fake=False)
        create_synthetic_clip(fake_dir / f"demo_fake_{i+1}.mp4", is_fake=True)
    print(f"Created 2 real and 2 fake clips in: {DEMO_DIR}")

    print("\n[2/4] Executing Fine-Tuning Pipeline (1 Epoch, Mode: head_only)...")
    parser = argparse.Namespace(
        data_dir=str(DEMO_DIR),
        output_dir=str(WEIGHTS_DIR),
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        mode="head_only",
        device=None,
    )
    checkpoint_path = train_vision_model(parser)

    print("\n[3/4] Verifying Saved Checkpoint...")
    pt_file = Path(checkpoint_path)
    assert pt_file.exists(), f"Expected checkpoint file at {pt_file}"
    assert pt_file.stat().st_size > 10000, "Checkpoint file is smaller than expected."
    print(f"Checkpoint verified: {pt_file.name} ({pt_file.stat().st_size / 1024:.1f} KB)")

    print("\n[4/4] Verifying DeepfakeVisionDetector Loading & Inference...")
    detector = DeepfakeVisionDetector()
    assert detector.has_finetuned_model, "Detector did not load the fine-tuned model!"

    sample_test = ROOT_DIR / "sample_test_video.mp4"
    if sample_test.exists():
        res = detector.predict(sample_test)
        print(f"\nInference on Sample Video: {res['prediction'].upper()} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Verdict: {res['verdict']}")
        for f in res["findings_log"]:
            print(f"  Log: {f}")
        assert "neural_classifier" in res["diagnostic_breakdown"], "neural_classifier missing in diagnostic breakdown!"

    print("\nCleaning up synthetic demo dataset...")
    shutil.rmtree(DEMO_DIR, ignore_errors=True)

    print("\n" + "=" * 65)
    print("ALL FINE-TUNING DEMO & INTEGRATION CHECKS PASSED (100% GREEN)")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
