"""
Command-line interface for Unified Multimodal Deepfake Detection & Localization.
Usage:
    python predict_cli.py path/to/audio.wav
    python predict_cli.py path/to/video.mp4
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows terminals if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from ai_models.multimodal.multimodal_detector import UnifiedDeepfakeDetector
except ImportError:
    from multimodal_detector import UnifiedDeepfakeDetector


def main():
    parser = argparse.ArgumentParser(
        description="Classify audio or video media as authentic or deepfake using Wav2Vec2 and AV-Deepfake1M benchmarks."
    )
    parser.add_argument("media_file", type=str, help="Path to media file (WAV, MP3, FLAC, MP4, WebM, AVI, etc.)")
    parser.add_argument("--audio-threshold", type=float, default=0.85, help="Confidence threshold for audio deepfake (default: 0.85)")
    parser.add_argument("--visual-threshold", type=float, default=0.65, help="Confidence threshold for visual deepfake (default: 0.65)")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()
    file_path = Path(args.media_file)

    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing media: {file_path.name}...")
    detector = UnifiedDeepfakeDetector(
        device=args.device,
        audio_fake_threshold=args.audio_threshold,
        visual_fake_threshold=args.visual_threshold,
    )
    result = detector.predict(file_path, filename=file_path.name)

    print("\n" + "=" * 60)
    print("      MULTIMODAL AUDIO-VISUAL DEEPFAKE DETECTION REPORT")
    print("=" * 60)

    # Status badge
    is_fake = result.get("is_fake", False)
    badge = "[ALERT] 🚨" if is_fake else "[PASS] ✅"

    print(f"Overall Decision:       {badge} {result.get('verdict_title')}")
    print(f"AV-Deepfake1M Class:    {result.get('overall_verdict', 'unknown').upper()}")
    print(f"Overall Confidence:     {result.get('overall_confidence', 0.0) * 100:.2f}%")
    print(f"Media Type:             {result.get('media_type', 'unknown').upper()}")
    print(f"Total Duration:         {result.get('duration_seconds', 0.0):.2f}s")
    print(f"Inference Latency:      {result.get('inference_time_seconds', 0.0):.3f}s")
    print("-" * 60)

    # Audio details
    audio_res = result.get("audio_analysis", {})
    is_audio_active = (
        audio_res.get("activated", True)
        and audio_res.get("probabilities") is not None
        and audio_res.get("status") not in ["not_active", "no_audio_stream"]
    )
    if is_audio_active:
        print("🎙️ AUDIO / SPEECH BREAKDOWN:")
        print(f"   Prediction:          {audio_res.get('prediction', 'N/A').upper()}")
        if audio_res.get("probabilities"):
            print(f"   Real Probability:    {audio_res['probabilities'].get('real', 0.0) * 100:.2f}%")
            print(f"   Fake Probability:    {audio_res['probabilities'].get('fake', 0.0) * 100:.2f}%")
        if audio_res.get("audio_fake_segments"):
            print(f"   Fake Audio Windows:  {audio_res['audio_fake_segments']}")
    else:
        print("🎙️ AUDIO / SPEECH BREAKDOWN: DEACTIVATED (Silent / No Audio Track)")
        print(f"   Note:                {audio_res.get('note', 'Audio subsystem was not activated.')}")

    # Visual details
    visual_res = result.get("visual_analysis", {})
    if visual_res.get("status") != "skipped":
        print("-" * 60)
        print("👁️ COMPUTER VISION BREAKDOWN:")
        print(f"   Prediction:          {visual_res.get('prediction', 'N/A').upper()}")
        if visual_res.get("probabilities"):
            print(f"   Real Probability:    {visual_res['probabilities'].get('real', 0.0) * 100:.2f}%")
            print(f"   Fake Probability:    {visual_res['probabilities'].get('fake', 0.0) * 100:.2f}%")
        print(f"   Frames / Faces:      {visual_res.get('faces_detected', 0)} faces in {visual_res.get('frames_analyzed', 0)} frames")
        v_met = visual_res.get("forensic_metrics", {})
        if v_met:
            print(f"   Boundary Artifact:   {v_met.get('boundary_artifact_score', 0.0) * 100:.1f}%")
            print(f"   2D FFT Grid Anomaly: {v_met.get('fft_frequency_score', 0.0) * 100:.1f}%")
            print(f"   Temporal Flicker:    {v_met.get('temporal_flicker_score', 0.0) * 100:.1f}%")
        if visual_res.get("visual_fake_segments"):
            print(f"   Fake Video Windows:  {visual_res['visual_fake_segments']}")

    # Unified Temporal Localization
    fake_segments = result.get("fake_segments", [])
    print("-" * 60)
    print(f"⏱️ LOCALIZED TAMPERING INTERVALS: {len(fake_segments)} detected")
    for seg in fake_segments:
        print(f"   -> [{seg[0]:.2f}s -> {seg[1]:.2f}s]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
