"""
Memory-Efficient LoRA / PEFT Fine-Tuning Script for Wav2Vec2 Audio Deepfake Detection.
Optimized for NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) and CPU fallback.

Usage:
  1. Install CUDA PyTorch & PEFT:
     pip install peft accelerate datasets
  2. Organize dataset:
     dataset/
       real/   (human voice WAV/MP3 files)
       fake/   (AI cloned voice WAV/MP3 files)
  3. Run training:
     python finetune_lora.py --data_dir ./dataset --output_dir ./finetuned_model --epochs 3
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)

TARGET_SAMPLE_RATE = 16000
BASE_MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "finetuned_model"


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths: List[Path], labels: List[int], feature_extractor, max_duration: float = 3.0):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_samples = int(TARGET_SAMPLE_RATE * max_duration)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            data, sr = sf.read(str(file_path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            if sr != TARGET_SAMPLE_RATE:
                resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
                data = resampler(torch.from_numpy(data).float().unsqueeze(0)).squeeze(0).numpy()

            if len(data) > self.max_samples:
                start = np.random.randint(0, len(data) - self.max_samples)
                data = data[start : start + self.max_samples]
            elif len(data) < self.max_samples:
                data = np.pad(data, (0, self.max_samples - len(data)), mode="constant")

            inputs = self.feature_extractor(
                data,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )

            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)
            return item

        except Exception as e:
            dummy = np.zeros(self.max_samples, dtype=np.float32)
            inputs = self.feature_extractor(dummy, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)
            return item


def load_local_dataset(data_dir: Path) -> Tuple[List[Path], List[int]]:
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    paths, labels = [], []
    if real_dir.exists():
        for p in real_dir.rglob("*"):
            if p.suffix.lower() in valid_exts:
                paths.append(p)
                labels.append(1)

    if fake_dir.exists():
        for p in fake_dir.rglob("*"):
            if p.suffix.lower() in valid_exts:
                paths.append(p)
                labels.append(0)

    return paths, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}


def train(args):
    print("=" * 60)
    print("  Wav2Vec2 Deepfake Audio Detector Fine-Tuning Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware execution device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("Note: Running on CPU. For faster GPU training on your RTX 3050,")
        print("install CUDA PyTorch: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124")

    data_dir = Path(args.data_dir)
    paths, labels = load_local_dataset(data_dir)
    print(f"Dataset loaded from: {data_dir}")
    print(f"  Total samples: {len(paths)}")
    print(f"  Real samples: {labels.count(1)}")
    print(f"  Fake samples: {labels.count(0)}")

    if len(paths) == 0:
        print(f"[!] Error: No audio files found in {data_dir}/real or {data_dir}/fake.")
        print("Please place human voice files in dataset/real/ and synthetic files in dataset/fake/.")
        sys.exit(1)

    indices = np.arange(len(paths))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(paths))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx] if len(val_idx) > 0 else train_paths[:2]
    val_labels = [labels[i] for i in val_idx] if len(val_idx) > 0 else train_labels[:2]

    model_source = str(BASE_MODEL_DIR) if BASE_MODEL_DIR.exists() else "mo-thecreator/Deepfake-audio-detection"
    print(f"Loading base model from: {model_source}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
    model = AutoModelForAudioClassification.from_pretrained(
        model_source,
        num_labels=2,
        id2label={0: "fake", 1: "real"},
        label2id={"fake": 0, "real": 1},
    )

    use_peft = args.use_lora
    if use_peft:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            print("Applying LoRA for 4GB VRAM memory efficiency...")
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("PEFT package not found. Freezing feature encoder.")
            model.freeze_feature_encoder()

    train_dataset = AudioDataset(train_paths, train_labels, feature_extractor)
    val_dataset = AudioDataset(val_paths, val_labels, feature_extractor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch" if len(val_paths) > 2 else "no",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=len(val_paths) > 2,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting Training...")
    trainer.train()

    print(f"Training Complete! Saving checkpoint to: {output_dir}")
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))
    print(f"Checkpoint successfully saved in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for Deepfake Detection")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset/ containing real/ and fake/ subfolders")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model", help="Path to save fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device (keep 2 for 4GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Enable PEFT/LoRA adaptation")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    args = parser.parse_args()
    train(args)
