import os
import json
from pathlib import Path
import sys

# Ensure backend and ai_models are in the python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_models.audio.detector import DeepfakeAudioDetector

def evaluate_directory(detector: DeepfakeAudioDetector, dir_path: str, expected_label: str):
    path = Path(dir_path)
    if not path.exists():
        print(f"Directory {dir_path} does not exist.")
        return []

    results = []
    print(f"\n--- Evaluating {expected_label} Directory: {dir_path} ---")
    for file in path.glob("*.*"):
        if file.suffix.lower() not in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]:
            continue
            
        print(f"Processing: {file.name}")
        try:
            res = detector.predict(str(file))
            pred = res.get("prediction")
            conf = res.get("calibrated_confidence")
            fake_prob = res.get("fake_probability")
            real_prob = res.get("real_probability")
            
            is_correct = pred == expected_label
            
            results.append({
                "file": file.name,
                "expected": expected_label,
                "predicted": pred,
                "is_correct": is_correct,
                "calibrated_confidence": conf,
                "fake_probability": fake_prob,
                "real_probability": real_prob
            })
            
            mark = "[PASS]" if is_correct else "[FAIL]"
            print(f"  {mark} Predicted: {pred} | Conf: {conf*100:.1f}% | FakeProb: {fake_prob:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] processing {file.name}: {e}")
            
    return results

def main():
    print("Initializing DeepfakeAudioDetector...")
    detector = DeepfakeAudioDetector()
    
    root_dir = Path(__file__).resolve().parent.parent
    real_dir = root_dir / "dataset" / "real"
    fake_dir = root_dir / "dataset" / "fake"
    
    all_results = []
    
    all_results.extend(evaluate_directory(detector, real_dir, "REAL"))
    all_results.extend(evaluate_directory(detector, fake_dir, "FAKE"))
    
    if not all_results:
        print("No audio files found in dataset/real or dataset/fake.")
        return
        
    correct = sum(1 for r in all_results if r["is_correct"])
    total = len(all_results)
    accuracy = correct / total
    
    print("\n===============================")
    print(f"FINAL EVALUATION REPORT")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print("===============================\n")
    
    # Generate Markdown Table
    print("| File | Expected | Predicted | Correct | Confidence |")
    print("|------|----------|-----------|---------|------------|")
    for r in all_results:
        mark = "[PASS]" if r["is_correct"] else "[FAIL]"
        print(f"| {r['file']} | {r['expected']} | {r['predicted']} | {mark} | {r['calibrated_confidence']*100:.1f}% |")

if __name__ == "__main__":
    main()
