"""
Model downloader for mo-thecreator/Deepfake-audio-detection.
Downloads model weights and preprocessor configuration from Hugging Face Hub
to a local directory for offline inference and fast loading.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

REPO_ID = "mo-thecreator/Deepfake-audio-detection"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "model"


def download_model(repo_id: str = REPO_ID, local_dir: Path = DEFAULT_LOCAL_DIR) -> Path:
    print(f"==================================================")
    print(f"Downloading model: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"==================================================")
    
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download core files (exclude tensorboard events to save bandwidth)
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["runs/*", "*.tfevents*", "training_args.bin"],
        resume_download=True,
    )
    
    print(f"\nFiles downloaded to: {downloaded_path}")
    
    # Verify core files
    expected_files = ["config.json", "preprocessor_config.json", "model.safetensors"]
    missing = [f for f in expected_files if not (local_dir / f).exists()]
    
    if missing:
        raise FileNotFoundError(f"Missing required model files: {missing}")
        
    print("All required model files are present:")
    for f in local_dir.glob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f" - {f.name}: {size_mb:.2f} MB")
            
    print("\nVerifying model loading from local directory...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(local_dir))
    model = AutoModelForAudioClassification.from_pretrained(str(local_dir))
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model successfully verified! Total parameters: {num_params:,}")
    print(f"Labels mapping: {model.config.id2label}")
    print("Ready for inference!")
    return local_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download mo-thecreator/Deepfake-audio-detection model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_LOCAL_DIR),
        help="Local directory to store the model",
    )
    args = parser.parse_args()
    download_model(local_dir=Path(args.output_dir))
