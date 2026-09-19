"""
Video Deepfake Detection Benchmark Evaluation Suite.
Evaluates DeepfakeVisionDetector across real and fake video test sets.
Computes Accuracy, Precision, Recall, F1-Score, False Alarm Rate, and Latency.

Usage:
  python evaluate_video_benchmark.py --test_dir ./video_dataset --threshold 0.50
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_models.vision.vision_detector import DeepfakeVisionDetector


def evaluate_video_benchmark(test_dir: Path, threshold: float = 0.50):
    print("=" * 70)
    print("  Video Deepfake Detection Forensic Benchmark Evaluation")
    print("=" * 70)

    real_dir = test_dir / "real"
    fake_dir = test_dir / "fake"
    video_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv"}

    real_videos = [p for p in real_dir.rglob("*") if p.suffix.lower() in video_exts] if real_dir.exists() else []
    fake_videos = [p for p in fake_dir.rglob("*") if p.suffix.lower() in video_exts] if fake_dir.exists() else []

    print(f"Test Directory: {test_dir}")
    print(f"  Authentic Videos Found: {len(real_videos)}")
    print(f"  Manipulated Videos Found: {len(fake_videos)}")
    print(f"  Decision Threshold: {threshold * 100:.1f}%")

    if not real_videos and not fake_videos:
        print(f"[!] Error: No video files found in {real_dir} or {fake_dir}.")
        return

    detector = DeepfakeVisionDetector(confidence_threshold=threshold)

    results: List[Dict] = []
    latencies: List[float] = []

    print("\nEvaluating Authentic (Real) Videos:")
    for v in real_videos:
        t0 = time.perf_counter()
        res = detector.predict(v, fake_threshold=threshold)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        is_correct = res["prediction"] == "real"
        results.append({
            "path": v.name,
            "ground_truth": "real",
            "prediction": res["prediction"],
            "prob_fake": res["probabilities"]["fake"],
            "prob_real": res["probabilities"]["real"],
            "correct": is_correct,
            "latency": lat,
        })
        mark = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"  [{mark}] {v.name:<25} -> Pred: {res['prediction']:<8} (Conf: {res['confidence']*100:.1f}%, Real Prob: {res['probabilities']['real']*100:.1f}%, Latency: {lat:.2f}s)")

    print("\nEvaluating Manipulated (Fake) Videos:")
    for v in fake_videos:
        t0 = time.perf_counter()
        res = detector.predict(v, fake_threshold=threshold)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        is_correct = res["prediction"] == "fake"
        results.append({
            "path": v.name,
            "ground_truth": "fake",
            "prediction": res["prediction"],
            "prob_fake": res["probabilities"]["fake"],
            "prob_real": res["probabilities"]["real"],
            "correct": is_correct,
            "latency": lat,
        })
        mark = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"  [{mark}] {v.name:<25} -> Pred: {res['prediction']:<8} (Conf: {res['confidence']*100:.1f}%, Fake Prob: {res['probabilities']['fake']*100:.1f}%, Latency: {lat:.2f}s)")

    # Compute Metrics
    tp = sum(1 for r in results if r["ground_truth"] == "fake" and r["prediction"] == "fake")
    fp = sum(1 for r in results if r["ground_truth"] == "real" and r["prediction"] == "fake")
    tn = sum(1 for r in results if r["ground_truth"] == "real" and r["prediction"] == "real")
    fn = sum(1 for r in results if r["ground_truth"] == "fake" and r["prediction"] == "real")

    total = len(results)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))
    false_alarm_rate = fp / max(1, (fp + tn))
    miss_rate = fn / max(1, (fn + tp))
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 70)
    print("  FORENSIC BENCHMARK PERFORMANCE REPORT")
    print("=" * 70)
    print(f"  Total Videos Evaluated:       {total}")
    print(f"  Accuracy:                     {accuracy * 100:.1f}%")
    print(f"  Precision:                    {precision * 100:.1f}%")
    print(f"  Recall (Sensitivity):         {recall * 100:.1f}%")
    print(f"  F1-Score:                     {f1:.3f}")
    print(f"  False Alarm Rate (Real->Fake):{false_alarm_rate * 100:.1f}%")
    print(f"  Miss Rate (Fake->Real):       {miss_rate * 100:.1f}%")
    print(f"  Mean Processing Latency:      {avg_latency:.2f}s / video")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Deepfake Detection Benchmark")
    parser.add_argument("--test_dir", type=str, default="./video_dataset", help="Directory containing real/ and fake/ video folders")
    parser.add_argument("--threshold", type=float, default=0.50, help="Forensic sensitivity threshold")
    args = parser.parse_args()

    evaluate_video_benchmark(Path(args.test_dir), threshold=args.threshold)
