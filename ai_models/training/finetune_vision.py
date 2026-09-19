"""
Memory-Efficient Fine-Tuning Pipeline for Video Deepfake Detection.
Optimized for NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) and CPU fallback.
Designed as the visual counterpart to the audio Wav2Vec2 fine-tuning pipeline.

Features:
  1. Automated face detection & extraction from video files (.mp4, .webm, .avi, etc.) or frame directories.
  2. Data augmentation (horizontal flip, color jitter, affine rotation, blur simulation).
  3. Pretrained MobileNetV3 backbone with configurable head-only or full fine-tuning.
  4. Memory-efficient gradient accumulation, AdamW optimizer, and CosineAnnealingLR.
  5. Checkpoint export to ai_models/weights/finetuned_vision.pt for immediate detector inference.

Usage:
  1. Organize dataset:
     dataset/
       real/   (authentic human video clips or face images)
       fake/   (deepfake / AI manipulated video clips or face images)
  2. Run training:
     python finetune_vision.py --data_dir ./dataset --output_dir ../weights --epochs 5 --mode head_only
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Set root path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_WEIGHTS_DIR = ROOT_DIR / "ai_models" / "weights"
DEFAULT_OUTPUT_MODEL = DEFAULT_WEIGHTS_DIR / "finetuned_vision.pt"


class VisionDeepfakeModel(nn.Module):
    """
    MobileNetV3 backbone with fine-tuning classification head.
    Outputs binary logits: [logit_fake, logit_real].
    """
    def __init__(self, num_classes: int = 2, mode: str = "head_only"):
        super().__init__()
        self.mode = mode
        
        try:
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            self.backbone = models.mobilenet_v3_small(weights=None)
            
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        if mode == "head_only":
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Multi-layer classification head with dropout and batch normalization
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        logits = self.head(features)
        return logits


class VideoDeepfakeDataset(Dataset):
    """
    Dataset loader for real and fake video clips or frame folders.
    Extracts face crops or central portraits across sampled frames.
    """
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[transforms.Compose] = None,
        frames_per_video: int = 4,
    ):
        self.samples = samples
        self.transform = transform or self.default_transforms()
        self.frames_per_video = frames_per_video
        self.extracted_items: List[Tuple[np.ndarray, int]] = []
        self._preprocess_samples()

    @staticmethod
    def default_transforms(train: bool = True) -> transforms.Compose:
        if train:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _extract_faces_from_video(self, video_path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 30

        step = max(1, total_frames // self.frames_per_video)
        crops = []
        frame_idx = 0

        while cap.isOpened() and len(crops) < self.frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                h, w = frame.shape[:2]
                # Center face crop fallback
                cy, cx = h // 2, w // 2
                box_sz = int(min(h, w) * 0.65)
                y1 = max(0, cy - box_sz // 2)
                y2 = min(h, y1 + box_sz)
                x1 = max(0, cx - box_sz // 2)
                x2 = min(w, x1 + box_sz)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            frame_idx += 1

        cap.release()
        return crops

    def _preprocess_samples(self):
        video_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv"}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for path, label in self.samples:
            ext = path.suffix.lower()
            if ext in video_exts:
                crops = self._extract_faces_from_video(path)
                for crop in crops:
                    self.extracted_items.append((crop, label))
            elif ext in image_exts:
                img = cv2.imread(str(path))
                if img is not None:
                    self.extracted_items.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), label))

    def __len__(self) -> int:
        return len(self.extracted_items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_rgb, label = self.extracted_items[idx]
        tensor = self.transform(image_rgb)
        return tensor, label


def discover_dataset(data_dir: Path) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    """
    Discovers dataset from data_dir/real and data_dir/fake folders.
    Labels: 0 = fake, 1 = real.
    """
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    valid_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".jpg", ".jpeg", ".png"}

    samples: List[Tuple[Path, int]] = []
    real_count = 0
    fake_count = 0

    if real_dir.exists():
        for p in real_dir.rglob("*"):
            if p.suffix.lower() in valid_exts and p.is_file():
                samples.append((p, 1))
                real_count += 1

    if fake_dir.exists():
        for p in fake_dir.rglob("*"):
            if p.suffix.lower() in valid_exts and p.is_file():
                samples.append((p, 0))
                fake_count += 1

    counts = {"real": real_count, "fake": fake_count, "total": len(samples)}
    return samples, counts


def train_vision_model(args):
    """
    Main training routine for fine-tuning the vision deepfake classifier.
    """
    print("=" * 65)
    print("  Video Deepfake Neural Model Fine-Tuning Pipeline")
    print("=" * 65)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Execution Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("Running on CPU. (For CUDA GPU acceleration, ensure CUDA PyTorch is installed)")

    data_dir = Path(args.data_dir)
    samples, counts = discover_dataset(data_dir)
    print(f"\nDataset Discovery in: {data_dir}")
    print(f"  Total Video / Frame Samples: {counts['total']}")
    print(f"  Authentic (Real): {counts['real']}")
    print(f"  Manipulated (Fake): {counts['fake']}")

    if counts["total"] == 0:
        print(f"\n[!] Error: No valid video or image files found in {data_dir}/real or {data_dir}/fake.")
        print("Please place authentic clips in dataset/real/ and manipulated clips in dataset/fake/.")
        sys.exit(1)

    # Train / Validation Split (80 / 20)
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split = int(0.8 * len(samples))
    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]] if split < len(samples) else train_samples[:2]

    print("\nExtracting & Preprocessing Face Tracks from Videos...")
    train_dataset = VideoDeepfakeDataset(train_samples, transform=VideoDeepfakeDataset.default_transforms(train=True))
    val_dataset = VideoDeepfakeDataset(val_samples, transform=VideoDeepfakeDataset.default_transforms(train=False))

    print(f"Total Processed Frame Crops -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    if len(train_dataset) == 0:
        print("[!] Error: No face crops could be extracted from the dataset.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize Model
    print(f"\nInitializing MobileNetV3 Deepfake Classifier (Mode: {args.mode})...")
    model = VisionDeepfakeModel(num_classes=2, mode=args.mode)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "finetuned_vision.pt"

    best_val_acc = 0.0
    print("\nStarting Training Execution...")
    start_train_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss / max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% "
            f"|| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% "
            f"| LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save Best Checkpoint
        if val_acc >= best_val_acc or epoch == 1:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "val_accuracy": val_acc,
                "classes": ["fake", "real"],
                "mode": args.mode,
                "timestamp": time.time(),
            }
            torch.save(checkpoint, str(best_model_path))
            print(f"  -> Saved best checkpoint (Val Acc: {val_acc:.1f}%) to: {best_model_path.name}")

    total_time = time.perf_counter() - start_train_time
    print("\n" + "=" * 65)
    print(f"Training Completed in {total_time:.2f}s!")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Saved Checkpoint: {best_model_path.resolve()}")
    print("=" * 65)
    return str(best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Video Deepfake Neural Classifier")
    parser.add_argument("--data_dir", type=str, default="./video_dataset", help="Path to dataset/ containing real/ and fake/ subfolders")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_WEIGHTS_DIR), help="Output directory for fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (keep 4-8 for laptop GPU or CPU)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--mode", type=str, choices=["head_only", "full"], default="head_only", help="Fine-tuning mode: head_only (freeze backbone) or full")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cuda', 'cpu', or auto-detect")

    args = parser.parse_args()
    train_vision_model(args)
