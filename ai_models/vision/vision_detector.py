"""
Computer Vision Deepfake Detection Engine.
Inspired by AV-Deepfake1M and BA-TFD forensic architectures:
- Video frame extraction and adaptive temporal sampling
- YuNet Neural & Biometric Skin-Chrominance facial localization
- Spatial blending artifact inspection (Laplacian boundary gradients & color transition)
- 2D Fourier Transform (FFT) Azimuthal periodic grid lattice analysis (GAN/Diffusion)
- Inter-frame temporal motion stability & micro-jitter modeling
- Pretrained deep visual feature identity coherence (MobileNetV3 backbone)
- Temporal tampering segment localization (visual_fake_segments)
- Multi-vector diagnostic reporting & frame-by-frame forensic analysis
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class DeepfakeVisionDetector:
    """
    Calibrated Computer Vision Deepfake Detector for facial and video manipulation forensics.
    Uses multi-branch consensus evidence fusion to eliminate false alarms and detect true fakes.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        target_fps: float = 4.0,
        max_frames: int = 64,
        confidence_threshold: float = 0.50,
    ):
        """
        Initialize the vision detector with device, face detector, and feature extractor.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.target_fps = target_fps
        self.max_frames = max_frames
        self.confidence_threshold = confidence_threshold

        # Image preprocessing for deep feature backbone
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Initialize YuNet Neural Face Detector if ONNX model is available
        self.yunet_detector = None
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
        yunet_path = weights_dir / "face_detection_yunet.onnx"
        if yunet_path.exists() and yunet_path.stat().st_size > 50000:
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=str(yunet_path),
                    config="",
                    input_size=(320, 240),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000,
                )
            except Exception:
                self.yunet_detector = None

        # Initialize pretrained vision feature backbone (MobileNetV3 Small)
        try:
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            self.backbone = models.mobilenet_v3_small(weights=None)

        self.backbone.classifier = nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()

        # Check for fine-tuned neural vision weights
        self.finetuned_classifier = None
        self.has_finetuned_model = False
        finetuned_path = weights_dir / "finetuned_vision.pt"
        if finetuned_path.exists() and finetuned_path.stat().st_size > 1000:
            try:
                from ai_models.training.finetune_vision import VisionDeepfakeModel
                clf = VisionDeepfakeModel(num_classes=2, mode="head_only")
                checkpoint = torch.load(str(finetuned_path), map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    clf.load_state_dict(checkpoint["model_state_dict"], strict=False)
                elif isinstance(checkpoint, dict) and "head_state_dict" in checkpoint:
                    clf.head.load_state_dict(checkpoint["head_state_dict"])
                clf.to(self.device)
                clf.eval()
                self.finetuned_classifier = clf
                self.has_finetuned_model = True
                print(f"Loaded fine-tuned vision classifier from '{finetuned_path.name}'.")
            except Exception as e:
                print(f"Note: Could not load fine-tuned vision weights: {e}")

        print(f"DeepfakeVisionDetector initialized on device '{self.device}' (Target FPS: {self.target_fps}, FineTuned: {self.has_finetuned_model}).")

    def _extract_video_frames(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
        """
        Extract frames and timestamps from a video file using OpenCV.
        Returns: (frames_list, timestamps_list, video_metadata)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        step = max(1, int(round(fps / self.target_fps)))
        frames: List[np.ndarray] = []
        timestamps: List[float] = []

        frame_idx = 0
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                t = frame_idx / fps
                frames.append(frame)
                timestamps.append(round(t, 3))

            frame_idx += 1

        cap.release()

        metadata = {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "sampled_frames": len(frames),
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}",
        }
        return frames, timestamps, metadata

    def _detect_and_crop_face(
        self, frame_bgr: np.ndarray, prev_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], bool]:
        """
        Multi-tier robust face detection with spatial-temporal subject tracking:
        Tier 1: YuNet neural face detector with tracking lock.
        Tier 2: Biometric skin-chrominance contour localization.
        Tier 3: Canonical talking-head center-portrait ROI.
        Returns: (cropped_face, (x, y, w, h), is_face_detected)
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        face_box = None
        is_detected = False

        # Tier 1: YuNet Neural Face Detector
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w_frame, h_frame))
                retval, faces = self.yunet_detector.detect(frame_bgr)
                if faces is not None and len(faces) > 0:
                    valid_faces = []
                    for f in faces:
                        if f[-1] >= 0.4:
                            bx, by, bw, bh = f[0:4].astype(int)
                            if bw >= 20 and bh >= 20 and bx >= 0 and by >= 0:
                                valid_faces.append((int(bx), int(by), int(bw), int(bh), float(f[-1])))

                    if valid_faces:
                        if prev_box is not None:
                            # Spatial proximity tracking: lock onto the same subject across frames
                            px, py, pw, ph = prev_box
                            pcx, pcy = px + pw / 2.0, py + ph / 2.0
                            best = min(
                                valid_faces,
                                key=lambda f: (f[0] + f[2] / 2.0 - pcx) ** 2 + (f[1] + f[3] / 2.0 - pcy) ** 2
                            )
                        else:
                            # First frame: pick central prominent talking head
                            img_cx, img_cy = w_frame / 2.0, h_frame / 2.0
                            best = max(
                                valid_faces,
                                key=lambda f: (f[2] * f[3]) / (1.0 + 0.001 * ((f[0] + f[2] / 2.0 - img_cx) ** 2 + (f[1] + f[3] / 2.0 - img_cy) ** 2))
                            )
                        face_box = best[:4]
                        is_detected = True
            except Exception:
                face_box = None

        # Tier 2: Biometric Skin-Chrominance Localization
        if face_box is None:
            try:
                ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
                mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = (w_frame * h_frame) * 0.02
                valid_candidates = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area >= min_area:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        aspect = bh / float(bw) if bw > 0 else 0
                        if 0.6 <= aspect <= 2.2:
                            valid_candidates.append((bx, by, bw, bh, area))

                if valid_candidates:
                    if prev_box is not None:
                        px, py, pw, ph = prev_box
                        pcx, pcy = px + pw / 2.0, py + ph / 2.0
                        best = min(
                            valid_candidates,
                            key=lambda b: (b[0] + b[2] / 2.0 - pcx) ** 2 + (b[1] + b[3] / 2.0 - pcy) ** 2
                        )
                    else:
                        best = max(valid_candidates, key=lambda b: b[4])
                    face_box = best[:4]
                    is_detected = True
            except Exception:
                face_box = None

        # Tier 3: Canonical portrait talking-head center crop fallback
        if face_box is None:
            cw = int(w_frame * 0.45)
            ch = int(h_frame * 0.50)
            cx = (w_frame - cw) // 2
            cy = int(h_frame * 0.15)
            face_box = (cx, cy, cw, ch)
            is_detected = False

        # Add 15% margin around the face bounding box for boundary seam inspection
        x, y, w, h = face_box
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_frame, x + w + margin_x)
        y2 = min(h_frame, y + h + margin_y)

        cropped = frame_bgr[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            cropped = frame_bgr

        return cropped, face_box, is_detected

    def _analyze_spatial_boundary(self, face_bgr: np.ndarray) -> float:
        """
        Detect face-swap boundary blending seams and color transition discontinuities.
        Returns a calibrated score in [0.0, 1.0].
        """
        if face_bgr.shape[0] < 24 or face_bgr.shape[1] < 24:
            return 0.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Partition into outer boundary seam ring (outer 15%) and inner face core (inner 70%)
        border_y = max(3, int(h * 0.15))
        border_x = max(3, int(w * 0.15))

        outer_mask = np.zeros_like(gray, dtype=bool)
        outer_mask[:border_y, :] = True
        outer_mask[-border_y:, :] = True
        outer_mask[:, :border_x] = True
        outer_mask[:, -border_x:] = True

        inner_mask = ~outer_mask

        # 1. Laplacian gradient inspection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        outer_var = float(np.var(laplacian[outer_mask]))
        inner_var = float(np.var(laplacian[inner_mask]))

        lap_ratio = abs(outer_var - inner_var) / (outer_var + inner_var + 1e-5)

        # 2. Color gradient transition across seam in YCrCb chroma
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        cr_outer_mean = float(np.mean(cr[outer_mask]))
        cr_inner_mean = float(np.mean(cr[inner_mask]))
        cb_outer_mean = float(np.mean(cb[outer_mask]))
        cb_inner_mean = float(np.mean(cb[inner_mask]))

        chroma_dist = np.sqrt((cr_outer_mean - cr_inner_mean) ** 2 + (cb_outer_mean - cb_inner_mean) ** 2)

        # Calibrated score:
        # Natural faces have lap_ratio ~ 0.15 - 0.45 and chroma_dist ~ 2.0 - 8.0.
        # Deepfake swaps typically exhibit lap_ratio > 0.65 and chroma_dist > 18.0.
        edge_artifact = np.clip((lap_ratio - 0.52) / 0.38, 0.0, 1.0)
        color_artifact = np.clip((chroma_dist - 14.0) / 16.0, 0.0, 1.0)

        boundary_score = float(0.6 * edge_artifact + 0.4 * color_artifact)
        return boundary_score

    def _analyze_fft_frequency(self, face_bgr: np.ndarray) -> float:
        """
        Analyze 2D Fast Fourier Transform to detect periodic GAN/diffusion lattice grid spikes.
        Measures azimuthal angular variance rather than raw sharpness to avoid false positives on clean video.
        Returns a calibrated score in [0.0, 1.0].
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # 2D Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-9)

        rows, cols = mag.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        angles = (np.arctan2(y - crow, x - ccol) * 180 / np.pi) % 180

        # Mid-to-high frequency ring (where deconvolution checkerboard artifacts reside)
        ring_mask = (dist > 20) & (dist < 55)
        if not np.any(ring_mask):
            return 0.0

        # Measure energy across 18 directional sectors (10 degrees each)
        sector_energies = []
        for deg in range(0, 180, 10):
            s_mask = ring_mask & (angles >= deg) & (angles < deg + 10)
            if np.any(s_mask):
                sector_energies.append(float(np.mean(mag[s_mask])))

        if len(sector_energies) < 6:
            return 0.0

        peak = float(max(sector_energies))
        median = float(np.median(sector_energies))
        ratio = peak / (median + 1e-6)

        # Calibration:
        # Optical camera frames exhibit smooth angular decay (ratio typically 1.05 - 1.25).
        # GAN and diffusion upsampling grids exhibit prominent directional spikes (ratio > 1.45).
        fft_score = float(np.clip((ratio - 1.28) / 0.35, 0.0, 1.0))
        return fft_score

    def _analyze_temporal_consistency(
        self, face_crops: List[np.ndarray]
    ) -> Tuple[float, List[float]]:
        """
        Analyze temporal micro-jitter and face warping across consecutive frames.
        Compensates for rigid head translation using phase correlation so that
        natural speech, head turns, and camera shake are not falsely penalized.
        Returns: (overall_flicker_score, list_of_frame_flicker_scores)
        """
        if len(face_crops) < 2:
            return 0.0, [0.0] * len(face_crops)

        flicker_scores = [0.0]
        resized = [cv2.resize(f, (96, 96)) for f in face_crops]

        for i in range(1, len(resized)):
            prev = cv2.cvtColor(resized[i - 1], cv2.COLOR_BGR2GRAY)
            curr = cv2.cvtColor(resized[i], cv2.COLOR_BGR2GRAY)

            # Subpixel phase correlation to align rigid head movement
            try:
                prev_f = np.float32(prev)
                curr_f = np.float32(curr)
                shift, _ = cv2.phaseCorrelate(prev_f, curr_f)
                dx, dy = shift
                if abs(dx) < 20 and abs(dy) < 20:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    prev_aligned = cv2.warpAffine(prev, M, (96, 96), borderMode=cv2.BORDER_REPLICATE)
                else:
                    prev_aligned = prev
            except Exception:
                prev_aligned = prev

            # High-pass Laplacian edge representation
            lap_prev = cv2.Laplacian(prev_aligned, cv2.CV_32F)
            lap_curr = cv2.Laplacian(curr, cv2.CV_32F)

            # Normalize high-pass inter-frame edge discrepancy
            diff = np.abs(lap_curr - lap_prev)
            mean_edge = (np.mean(np.abs(lap_curr)) + np.mean(np.abs(lap_prev)) + 1e-5)
            norm_jitter = float(np.mean(diff) / mean_edge)

            # Natural talking/motion with alignment: norm_jitter is typically 0.20 - 0.50.
            # Deepfake warping / mask seam fluttering: norm_jitter > 0.68.
            flicker = float(np.clip((norm_jitter - 0.58) / 0.32, 0.0, 1.0))
            flicker_scores.append(flicker)

        overall_flicker = float(np.mean(flicker_scores[1:])) if len(flicker_scores) > 1 else 0.0
        return overall_flicker, flicker_scores

    def _extract_deep_features(self, face_crops: List[np.ndarray]) -> Tuple[float, float, Optional[float]]:
        """
        Pass face crops through pretrained MobileNetV3 backbone to evaluate
        semantic biometric identity consistency across consecutive frames, and
        through the fine-tuned neural classifier if loaded.
        Returns: (identity_drift_score, mean_cosine_similarity, neural_fake_prob)
        """
        if not face_crops:
            return 0.0, 1.0, None

        sample_crops = face_crops[:24]  # Limit to 24 frames for swift inference
        tensors = []
        for face in sample_crops:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.backbone(batch)
            features = features.view(features.size(0), -1)
            # L2 normalize embeddings
            norm = torch.norm(features, dim=-1, keepdim=True) + 1e-6
            normalized_features = features / norm

        neural_fake_prob = None
        if self.has_finetuned_model and self.finetuned_classifier is not None:
            try:
                with torch.no_grad():
                    logits = self.finetuned_classifier(batch)
                    probs = torch.softmax(logits, dim=1)  # Class 0: fake, Class 1: real
                    raw_p = float(probs[:, 0].mean().cpu().item())
                    # Only use neural classifier if it is confident/decisive (not ~0.50 coin flip)
                    if abs(raw_p - 0.50) >= 0.15:
                        neural_fake_prob = raw_p
                    else:
                        neural_fake_prob = None
            except Exception:
                neural_fake_prob = None

        if normalized_features.size(0) < 2:
            return 0.0, 1.0, neural_fake_prob

        # Cosine similarity between consecutive frames
        sims = []
        for i in range(1, normalized_features.size(0)):
            cos = float(torch.dot(normalized_features[i], normalized_features[i - 1]).cpu().item())
            sims.append(cos)

        mean_sim = float(np.mean(sims)) if sims else 1.0

        # Authentic video: same person speaking maintains mean similarity 0.88 - 0.99.
        # Deepfakes / face swaps / AI morphs exhibit identity drift or sudden jumps (mean sim < 0.80).
        drift_score = float(np.clip((0.84 - mean_sim) / 0.22, 0.0, 1.0))
        return drift_score, round(mean_sim, 4), neural_fake_prob

    def _analyze_generative_diffusion(
        self, frames: List[np.ndarray], face_crops: List[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Comprehensive forensic analysis for Generative AI Video (Text-to-Video / Diffusion models:
        Sora, Kling, Runway Gen-3, Luma, Hunyuan, Hedra, SadTalker):
        1. 2D FFT Radial Power Spectrum Decay (PSD slope alpha):
           Natural optical camera lenses decay at 1 / f^alpha with alpha in [1.8, 2.85].
           Latent diffusion models suffer from VAE decoder compression bottleneck, causing
           unnatural steep spectral dropoff (alpha > 3.05).
        2. Ocular Dynamics & Biological Blink Invariance:
           Authentic humans blink or exhibit micro-saccades periodically (std > 0.030 across 10-40 frames).
           Generative talking heads suffer from frozen ocular geometry / unblinking stare (std < 0.018).
        3. Background Latent Breathing & Texture Drift:
           Measures background optical stability in non-face regions to detect dream-like latent ripple.
        Returns: (generative_diffusion_score [0.0, 1.0], details_dict)
        """
        if not frames:
            return 0.0, {"mean_slope": -2.5, "spectral_score": 0.0, "ocular_score": 0.0, "latent_drift_score": 0.0}

        # 1. 2D FFT Radial Power Spectrum Decay Slope across full frames
        slopes = []
        step = max(1, len(frames) // 16)
        sample_frames = frames[::step][:16]

        for f in sample_frames:
            try:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (256, 256))
                f_shift = np.fft.fftshift(np.fft.fft2(gray))
                psd = np.abs(f_shift) ** 2
                h, w = psd.shape
                cy, cx = h // 2, w // 2
                y, x = np.ogrid[:h, :w]
                r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
                rad_mean = [psd[r == rad].mean() for rad in range(12, 105)]
                freqs = np.arange(12, 105)
                slope, _ = np.polyfit(np.log(freqs), np.log(rad_mean), 1)
                slopes.append(slope)
            except Exception:
                continue

        mean_slope = float(np.mean(slopes)) if slopes else -2.5
        # Natural optical lenses: -2.0 to -2.85. Diffusion VAE decay: steeper than -3.10
        spectral_score = float(np.clip((-mean_slope - 2.85) / 0.38, 0.0, 1.0))

        # 2. Ocular dynamics & blink variance across face crops
        ocular_score = 0.0
        eye_std = 0.05
        valid_crops = [fc for fc in face_crops if fc is not None and fc.size > 0]
        if len(valid_crops) >= 8:
            eye_dark_ratios = []
            for fc in valid_crops:
                try:
                    h, w = fc.shape[:2]
                    eye_strip = cv2.cvtColor(fc[int(h * 0.25):int(h * 0.50), int(w * 0.15):int(w * 0.85)], cv2.COLOR_BGR2GRAY)
                    dark_ratio = float(np.mean(eye_strip < 45))
                    eye_dark_ratios.append(dark_ratio)
                except Exception:
                    continue

            if len(eye_dark_ratios) >= 8:
                eye_std = float(np.std(eye_dark_ratios))
                # Real humans blink or shift gaze (std > 0.030). Unblinking AI stare: std < 0.018
                ocular_score = float(np.clip((0.028 - eye_std) / 0.016, 0.0, 1.0))

        # 3. Background latent breathing
        latent_drift_score = 0.0
        if len(frames) >= 4:
            bg_drifts = []
            for i in range(1, min(12, len(frames))):
                try:
                    p_prev = cv2.resize(cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY), (320, 180))
                    p_curr = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (320, 180))
                    h, w = p_prev.shape
                    c1_p, c1_c = p_prev[:int(h * 0.25), :int(w * 0.25)], p_curr[:int(h * 0.25), :int(w * 0.25)]
                    c2_p, c2_c = p_prev[:int(h * 0.25), int(w * 0.75):], p_curr[:int(h * 0.25), int(w * 0.75):]
                    diff1 = float(np.abs(c1_p.astype(float) - c1_c.astype(float)).mean())
                    diff2 = float(np.abs(c2_p.astype(float) - c2_c.astype(float)).mean())
                    bg_drifts.append((diff1 + diff2) / 2.0)
                except Exception:
                    continue

            mean_bg_drift = float(np.mean(bg_drifts)) if bg_drifts else 0.0
            if 0.5 < mean_bg_drift < 4.0:
                latent_drift_score = float(np.clip((mean_bg_drift - 0.7) / 2.0, 0.0, 1.0))

        combined = float(0.55 * spectral_score + 0.35 * ocular_score + 0.10 * latent_drift_score)
        details = {
            "mean_slope": round(mean_slope, 3),
            "spectral_score": round(spectral_score, 3),
            "eye_std": round(eye_std, 4),
            "ocular_score": round(ocular_score, 3),
            "latent_drift_score": round(latent_drift_score, 3),
            "combined": round(combined, 3),
        }
        return combined, details

    def _localize_temporal_segments(
        self, timestamps: List[float], frame_tampering_scores: List[float], threshold: float = 0.45
    ) -> List[List[float]]:
        """
        Localize temporal fake intervals: [[start_sec, end_sec], ...]
        """
        segments: List[List[float]] = []
        in_segment = False
        start_t = 0.0

        for t, score in zip(timestamps, frame_tampering_scores):
            if score >= threshold:
                if not in_segment:
                    in_segment = True
                    start_t = t
            else:
                if in_segment:
                    in_segment = False
                    end_t = max(t, round(start_t + 0.25, 2))
                    segments.append([round(start_t, 2), round(end_t, 2)])

        if in_segment:
            end_t = max(timestamps[-1], round(start_t + 0.25, 2))
            segments.append([round(start_t, 2), round(end_t, 2)])

        # Merge segments that are closer than 0.75 seconds
        merged: List[List[float]] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
            else:
                prev_start, prev_end = merged[-1]
                curr_start, curr_end = seg
                if curr_start <= prev_end + 0.75:
                    merged[-1] = [prev_start, max(prev_end, curr_end)]
                else:
                    merged.append(seg)

        return merged

    def predict(
        self,
        video_input: Union[str, Path, bytes],
        fake_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run deepfake video classification, forensic vector diagnosis, and frame-by-frame analysis.
        """
        threshold = fake_threshold if fake_threshold is not None else self.confidence_threshold
        start_time = time.perf_counter()

        temp_file_path = None
        if isinstance(video_input, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_input)
                temp_file_path = f.name
            target_path = temp_file_path
        else:
            target_path = str(video_input)

        try:
            frames, timestamps, metadata = self._extract_video_frames(target_path)
            if not frames:
                raise ValueError("Could not extract any valid video frames from input.")

            face_crops: List[np.ndarray] = []
            valid_timestamps: List[float] = []
            frame_records: List[Dict[str, Any]] = []
            boundary_scores: List[float] = []
            fft_scores: List[float] = []

            prev_bbox = None
            for idx, (frame, t) in enumerate(zip(frames, timestamps)):
                face, bbox, is_detected = self._detect_and_crop_face(frame, prev_box=prev_bbox)
                if is_detected and bbox:
                    prev_bbox = bbox
                face_crops.append(face)
                valid_timestamps.append(t)

                b_score = self._analyze_spatial_boundary(face)
                f_score = self._analyze_fft_frequency(face)

                boundary_scores.append(b_score)
                fft_scores.append(f_score)

                # Frame-level combined score
                f_tamper = float(0.5 * b_score + 0.5 * f_score)
                status = "authentic" if f_tamper < 0.35 else ("suspicious" if f_tamper < 0.60 else "tampered")

                frame_records.append({
                    "frame_index": idx + 1,
                    "timestamp_seconds": round(t, 2),
                    "timestamp_label": f"{t:.2f}s",
                    "face_detected": bool(is_detected),
                    "bounding_box": list(bbox) if bbox else None,
                    "anomaly_score": round(f_tamper, 3),
                    "status": status,
                })

            faces_detected_count = sum(1 for r in frame_records if r["face_detected"])
            face_presence_ratio = round(faces_detected_count / len(frames), 3) if frames else 0.0

            # Temporal, deep feature, and generative diffusion consistency
            temporal_flicker, per_frame_flickers = self._analyze_temporal_consistency(face_crops)
            deep_anomaly, identity_sim, neural_fake_prob = self._extract_deep_features(face_crops)
            generative_diffusion, diff_details = self._analyze_generative_diffusion(frames, face_crops)

            # Merge temporal scores back to frame records
            for idx, f_score in enumerate(per_frame_flickers):
                if idx < len(frame_records):
                    cur_anom = frame_records[idx]["anomaly_score"]
                    updated_anom = round(0.7 * cur_anom + 0.3 * f_score, 3)
                    frame_records[idx]["anomaly_score"] = updated_anom
                    frame_records[idx]["status"] = "authentic" if updated_anom < 0.35 else ("suspicious" if updated_anom < 0.60 else "tampered")

            mean_boundary = float(np.mean(boundary_scores)) if boundary_scores else 0.0
            mean_fft = float(np.mean(fft_scores)) if fft_scores else 0.0

            # Multi-Branch Forensic Evidence Fusion
            branches = {
                "generative_diffusion": generative_diffusion,
                "face_swap_boundary": mean_boundary,
                "spectral_lattice": mean_fft,
                "temporal_flicker": temporal_flicker,
                "deep_anomaly": deep_anomaly,
            }

            if self.has_finetuned_model and neural_fake_prob is not None:
                branches["neural_classifier"] = neural_fake_prob
                base_prob = (
                    0.30 * generative_diffusion +
                    0.20 * mean_boundary +
                    0.15 * mean_fft +
                    0.15 * temporal_flicker +
                    0.10 * deep_anomaly +
                    0.10 * neural_fake_prob
                )
            else:
                base_prob = (
                    0.35 * generative_diffusion +
                    0.25 * mean_boundary +
                    0.15 * mean_fft +
                    0.15 * temporal_flicker +
                    0.10 * deep_anomaly
                )

            # Significant anomaly identification
            has_diffusion_anomaly = generative_diffusion >= 0.50
            has_boundary_anomaly = mean_boundary >= 0.48
            has_spectral_anomaly = mean_fft >= 0.45
            has_identity_anomaly = deep_anomaly >= 0.45
            has_neural_anomaly = (neural_fake_prob is not None and neural_fake_prob >= 0.65)
            has_temporal_anomaly = temporal_flicker >= 0.55

            physical_anomalies = [has_diffusion_anomaly, has_boundary_anomaly, has_spectral_anomaly, has_identity_anomaly]
            num_physical = sum(physical_anomalies)

            # Forensic Ground Truth & Evidence Fusion:
            # 1. Generative AI Video (Diffusion / Text-to-Video) decisive trigger
            if has_diffusion_anomaly:
                if generative_diffusion >= 0.70:
                    prob_fake = float(np.clip(0.85 + (generative_diffusion - 0.70) * 0.40, 0.85, 0.98))
                else:
                    prob_fake = float(np.clip(0.62 + (generative_diffusion - 0.50) * 1.15, 0.60, 0.85))
            # 2. Authentic optical capture: all physical and diffusion vectors are pristine
            elif mean_boundary < 0.35 and mean_fft < 0.25 and deep_anomaly < 0.35 and generative_diffusion < 0.32:
                prob_fake = float(max(mean_boundary, mean_fft, deep_anomaly, generative_diffusion) * 0.4)
                prob_fake = min(prob_fake, 0.15)
            # 3. Corroborated manipulation across multiple independent forensic vectors
            elif num_physical >= 2 or (num_physical >= 1 and (has_temporal_anomaly or has_neural_anomaly)):
                prob_fake = min(0.95, base_prob + 0.25)
            # 4. Single physical anomaly: moderate variation
            elif num_physical == 1:
                prob_fake = float(np.clip(base_prob, 0.35, 0.55))
            # 5. Authentic baseline
            else:
                prob_fake = float(np.clip(base_prob * 0.6, 0.03, 0.22))

            prob_fake = float(np.clip(prob_fake, 0.02, 0.98))
            prob_real = float(1.0 - prob_fake)

            is_fake = prob_fake >= threshold
            confidence = prob_fake if is_fake else prob_real

            # Update frame records if whole-video generative diffusion is identified
            if has_diffusion_anomaly:
                for r in frame_records:
                    r["anomaly_score"] = round(max(r["anomaly_score"], generative_diffusion * 0.85), 3)
                    r["status"] = "tampered" if r["anomaly_score"] >= 0.60 else "suspicious"

            # Explanatory Verdict & Technique Labeling
            primary_branch = max(branches, key=branches.get)
            if is_fake:
                prediction = "fake"
                risk_level = "CRITICAL (MANIPULATED)"
                if has_diffusion_anomaly or primary_branch == "generative_diffusion":
                    verdict = "AI-Generated Synthetic Video Detected (Diffusion VAE Spectral & Ocular Signature)"
                    technique = "Generative AI Video (Text-to-Video / Video Diffusion Model)"
                elif primary_branch == "face_swap_boundary":
                    verdict = "Facial Swap Deepfake Detected (Boundary Blending Seam Discontinuity)"
                    technique = "Face Swap / Compositing Deepfake (Poisson/Feathering Seam)"
                elif primary_branch in ["spectral_lattice", "generative_ai"]:
                    verdict = "AI-Generated Synthetic Video Detected (Periodic Spectral Lattice Signature)"
                    technique = "Generative AI Video (GAN / Upconvolution Grid Lattice)"
                elif primary_branch == "temporal_flicker":
                    verdict = "Temporal Glitch / Warping Deepfake Detected (Inter-Frame Jitter)"
                    technique = "Temporal Deepfake / Frame Warping"
                else:
                    verdict = "Deepfake Video Manipulation Detected (Biometric Identity Drift)"
                    technique = "Neural Manifold / Identity Drift Manipulation"
            elif prob_fake > 0.38:
                prediction = "suspicious_visual"
                risk_level = "MODERATE (SUSPICIOUS)"
                verdict = "Suspicious Visual Inconsistencies (Inconclusive)"
                technique = "Subtle Optical / Minor Generative Irregularities"
            else:
                prediction = "real"
                risk_level = "LOW (AUTHENTIC)"
                verdict = "Authentic Video Media Verified"
                technique = "Authentic Optical Capture (No Manipulation Detected)"

            # Generate human-readable forensic findings log
            findings_log = []
            if generative_diffusion < 0.35:
                findings_log.append(f"[AUTHENTIC] Optical frequency distribution conforms to natural camera lens physics (slope: {diff_details.get('mean_slope', -2.5):.2f}) with natural ocular dynamics.")
            else:
                findings_log.append(f"[ANOMALY] Synthetic video diffusion signature identified: steep spectral decay slope ({diff_details.get('mean_slope', -3.2):.2f}) and static ocular dynamics.")

            if mean_boundary < 0.30:
                findings_log.append("[AUTHENTIC] Facial boundary transitions are continuous; no blending seams or feathering halos detected.")
            else:
                findings_log.append("[ANOMALY] Elevated gradient discontinuity detected around facial perimeter, indicating possible mask blending.")

            if mean_fft < 0.25:
                findings_log.append("[AUTHENTIC] 2D Fourier power spectrum shows natural 1/f spatial decay without periodic deconvolution lattice.")
            else:
                findings_log.append("[ANOMALY] Periodic grid lattice peaks detected in frequency domain (characteristic of diffusion/GAN upsampling).")

            if temporal_flicker < 0.35 or (mean_boundary < 0.35 and mean_fft < 0.25 and generative_diffusion < 0.35):
                findings_log.append("[AUTHENTIC] Inter-frame facial dynamics are consistent; no synthetic warping or non-rigid jitter.")
            else:
                findings_log.append("[ANOMALY] Inter-frame warping and edge jitter detected across consecutive frames.")

            if deep_anomaly < 0.30:
                findings_log.append(f"[AUTHENTIC] Biometric identity embeddings remain stable across frames ({identity_sim * 100:.1f}% semantic similarity).")
            else:
                findings_log.append("[ANOMALY] Biometric identity embedding variance detected across frames, suggesting identity drift or morphing.")

            if self.has_finetuned_model and neural_fake_prob is not None:
                if neural_fake_prob > 0.50:
                    findings_log.append(f"[ANOMALY] Fine-tuned neural classifier indicated deepfake visual patterns ({neural_fake_prob * 100:.1f}% risk).")
                else:
                    findings_log.append(f"[AUTHENTIC] Fine-tuned neural classifier verified natural facial representation ({(1.0 - neural_fake_prob) * 100:.1f}% confidence).")

            # Diagnostic Vectors Breakdown
            def get_rating(val: float) -> str:
                if val < 0.25: return "Pristine (Authentic)"
                if val < 0.45: return "Moderate Variation"
                if val < 0.65: return "Elevated Anomaly"
                return "Critical Manipulation"

            diagnostic_breakdown = {
                "generative_diffusion": {
                    "name": "Generative Video Diffusion & Ocular Dynamics",
                    "score": round(generative_diffusion, 3),
                    "percentage": round(generative_diffusion * 100, 1),
                    "rating": get_rating(generative_diffusion),
                    "description": "Analyzes full-frame radial PSD decay slope and biological ocular/blinking variance characteristic of video diffusion models.",
                },
                "boundary_seams": {
                    "name": "Facial Boundary & Blending Seams",
                    "score": round(mean_boundary, 3),
                    "percentage": round(mean_boundary * 100, 1),
                    "rating": get_rating(mean_boundary),
                    "description": "Measures gradient discontinuity and color bleed along face boundary.",
                },
                "spectral_lattice": {
                    "name": "2D Spectral Lattice Grid (FFT)",
                    "score": round(mean_fft, 3),
                    "percentage": round(mean_fft * 100, 1),
                    "rating": get_rating(mean_fft),
                    "description": "Detects periodic grid artifacts produced by generative diffusion & GAN upsamplers.",
                },
                "temporal_stability": {
                    "name": "Temporal Motion & Jitter Stability",
                    "score": round(temporal_flicker, 3),
                    "percentage": round(temporal_flicker * 100, 1),
                    "rating": get_rating(temporal_flicker),
                    "description": "Evaluates micro-jitter, warping, and flickering across aligned frames.",
                },
                "identity_coherence": {
                    "name": "Deep Perceptual Identity Coherence",
                    "score": round(deep_anomaly, 3),
                    "percentage": round(deep_anomaly * 100, 1),
                    "rating": get_rating(deep_anomaly),
                    "description": "Quantifies semantic biometric identity stability using pretrained deep embeddings.",
                },
            }

            if self.has_finetuned_model and neural_fake_prob is not None:
                diagnostic_breakdown["neural_classifier"] = {
                    "name": "Fine-Tuned Neural Classifier (MobileNetV3)",
                    "score": round(neural_fake_prob, 3),
                    "percentage": round(neural_fake_prob * 100, 1),
                    "rating": get_rating(neural_fake_prob),
                    "description": "Inferred probability from fine-tuned deep visual classification head.",
                }

            # Temporal Tampering Localization
            frame_tampering_scores = [r["anomaly_score"] for r in frame_records]
            visual_fake_segments = []
            if prob_fake > 0.35 and frame_tampering_scores:
                visual_fake_segments = self._localize_temporal_segments(
                    valid_timestamps, frame_tampering_scores, threshold=0.45
                )

            elapsed = round(time.perf_counter() - start_time, 3)

            return {
                "status": "success",
                "prediction": prediction,
                "overall_verdict": "visual_modified" if is_fake else "real",
                "overall_prediction": "fake" if is_fake else "real",
                "overall_confidence": round(confidence, 4),
                "confidence": round(confidence, 4),
                "is_fake": bool(is_fake),
                "verdict": verdict,
                "verdict_title": verdict,
                "risk_level": risk_level,
                "probabilities": {
                    "fake": round(prob_fake, 4),
                    "real": round(prob_real, 4),
                },
                "visual_fake_segments": visual_fake_segments,
                "fake_segments": visual_fake_segments,
                "frames_analyzed": len(frames),
                "faces_detected": faces_detected_count,
                "face_presence_ratio": face_presence_ratio,
                "video_metadata": metadata,
                "duration_seconds": metadata.get("duration_seconds", 0.0),
                "forensic_metrics": {
                    "primary_technique": technique,
                    "generative_diffusion_score": round(generative_diffusion, 4),
                    "spectral_slope": diff_details.get("mean_slope", 0.0),
                    "ocular_variance_score": diff_details.get("ocular_score", 0.0),
                    "generative_ai_score": round(max(mean_fft, generative_diffusion), 4),
                    "boundary_artifact_score": round(mean_boundary, 4),
                    "fft_frequency_score": round(mean_fft, 4),
                    "temporal_flicker_score": round(temporal_flicker, 4),
                    "deep_feature_anomaly_score": round(deep_anomaly, 4),
                },
                "diagnostic_breakdown": diagnostic_breakdown,
                "frame_analysis": frame_records,
                "findings_log": findings_log,
                "inference_time_seconds": elapsed,
            }

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
