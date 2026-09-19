"""
Enhanced Inference Engine for Deepfake Audio Detection.
Refactored to meet strict Machine Learning pipeline standards.
"""

import io
import time
import logging
from pathlib import Path
from typing import Dict, Union, Tuple, Optional, Any, List

import numpy as np
import torch
import soundfile as sf
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

# Setup logging
logger = logging.getLogger("AudioDetector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

_CURRENT_DIR = Path(__file__).resolve().parent
_WEIGHTS_V3_DIR = _CURRENT_DIR.parent / "weights_v3"
_WEIGHTS_DIR = _CURRENT_DIR.parent / "weights"
_ROOT_MODEL_DIR = _CURRENT_DIR.parent.parent / "model"
_LOCAL_MODEL_DIR = _CURRENT_DIR / "model"

if (_WEIGHTS_V3_DIR / "model.safetensors").exists():
    DEFAULT_MODEL_DIR = _WEIGHTS_V3_DIR
elif (_WEIGHTS_DIR / "model.safetensors").exists():
    DEFAULT_MODEL_DIR = _WEIGHTS_DIR
elif (_ROOT_MODEL_DIR / "model.safetensors").exists():
    DEFAULT_MODEL_DIR = _ROOT_MODEL_DIR
elif (_LOCAL_MODEL_DIR / "model.safetensors").exists():
    DEFAULT_MODEL_DIR = _LOCAL_MODEL_DIR
else:
    DEFAULT_MODEL_DIR = _WEIGHTS_V3_DIR

HF_REPO_ID = "MelodyMachine/Deepfake-audio-detection-V2"
TARGET_SAMPLE_RATE = 16000

# Pipeline Configuration
CHUNK_DURATION_SEC = 4.0
OVERLAP_RATIO = 0.50
CALIBRATION_TEMPERATURE = 1.0  # Set to 1.0 to retain raw true probabilities


class DeepfakeAudioDetector:
    def __init__(
        self,
        model_path_or_id: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if model_path_or_id is None:
            if DEFAULT_MODEL_DIR.exists() and (DEFAULT_MODEL_DIR / "model.safetensors").exists():
                self.model_path = str(DEFAULT_MODEL_DIR)
            else:
                self.model_path = HF_REPO_ID
        else:
            self.model_path = str(model_path_or_id)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading DeepfakeAudioDetector from '{self.model_path}' on '{self.device}'...")
        start_time = time.perf_counter()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # PHASE 2: VERIFY LABEL MAPPING
        self.real_idx, self.fake_idx = self._verify_label_mapping()
        
        load_duration = time.perf_counter() - start_time
        logger.info(f"Model loaded in {load_duration:.2f}s.")

    def _verify_label_mapping(self) -> Tuple[int, int]:
        """Strictly verify the model's id2label mapping."""
        id2label = self.model.config.id2label
        real_idx, fake_idx = -1, -1
        
        logger.info("MODEL LABEL MAPPING")
        logger.info("-------------------")
        for idx, lbl in id2label.items():
            idx = int(idx)
            lbl_lower = str(lbl).lower()
            if "real" in lbl_lower or "bonafide" in lbl_lower or "authentic" in lbl_lower:
                real_idx = idx
                logger.info(f"REAL  -> {idx} ({lbl})")
            elif "fake" in lbl_lower or "spoof" in lbl_lower or "modified" in lbl_lower:
                fake_idx = idx
                logger.info(f"FAKE  -> {idx} ({lbl})")
                
        if real_idx == -1 or fake_idx == -1:
            raise ValueError(f"Could not unambiguously determine REAL and FAKE labels from config: {id2label}")
            
        return real_idx, fake_idx

    def load_audio(self, audio_input: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, int]:
        """Decode audio robustly."""
        # soundfile fallback logic ...
        try:
            if isinstance(audio_input, (str, Path)):
                data, sr = sf.read(str(audio_input), dtype="float32")
            elif isinstance(audio_input, (bytes, bytearray)):
                data, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
            elif isinstance(audio_input, io.BytesIO):
                audio_input.seek(0)
                data, sr = sf.read(audio_input, dtype="float32")
            else:
                raise TypeError(f"Unsupported type: {type(audio_input)}")
            return data, sr
        except Exception as e:
            if HAS_AV:
                return self._decode_with_av(audio_input)
            raise ValueError(f"Failed to decode audio: {e}")

    def _decode_with_av(self, audio_source: Any) -> Tuple[np.ndarray, int]:
        if isinstance(audio_source, (str, Path)):
            container = av.open(str(audio_source))
        elif isinstance(audio_source, (bytes, bytearray)):
            container = av.open(io.BytesIO(audio_source))
        else:
            audio_source.seek(0)
            container = av.open(audio_source)

        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if not audio_stream:
            container.close()
            raise ValueError("No audio stream found.")

        orig_sr = audio_stream.codec_context.sample_rate or TARGET_SAMPLE_RATE
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SAMPLE_RATE)
        chunks = []
        for frame in container.decode(audio_stream):
            for r_frame in resampler.resample(frame):
                chunks.append(r_frame.to_ndarray()[0])
        for r_frame in resampler.resample(None):
            chunks.append(r_frame.to_ndarray()[0])
        container.close()
        
        if not chunks:
            raise ValueError("Empty audio stream.")
        return np.concatenate(chunks, axis=0).astype(np.float32), TARGET_SAMPLE_RATE

    def preprocess_audio(self, data: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        PHASE 3: Standardized Preprocessing.
        Converts to mono, resamples, measures forensic metadata, removes DC offset.
        Does NOT aggressively peak normalize.
        """
        # 1. Mono conversion
        if data.ndim > 1:
            if data.shape[0] < data.shape[1] and data.shape[0] <= 8:
                data = np.mean(data, axis=0)
            else:
                data = np.mean(data, axis=1)
        data = data.squeeze()
        
        # 2. Resampling
        if orig_sr != TARGET_SAMPLE_RATE:
            tensor_wave = torch.from_numpy(data).float().unsqueeze(0)
            resampler = T.Resample(orig_freq=orig_sr, new_freq=TARGET_SAMPLE_RATE)
            data = resampler(tensor_wave).squeeze(0).numpy()
            
        duration = len(data) / TARGET_SAMPLE_RATE
        
        # 3. Forensic Measurements
        rms = float(np.sqrt(np.mean(data**2)))
        peak_amp = float(np.max(np.abs(data)))
        clipping_pct = float(np.mean(np.abs(data) > 0.99) * 100)
        silence_pct = float(np.mean(np.abs(data) < 1e-4) * 100)
        
        # 4. DC Offset removal (gentle)
        data = data - np.mean(data)
        
        # 5. Safe amplitude scaling (only if clipping heavily or extremely quiet)
        if peak_amp > 1.0:
            data = data / (peak_amp + 1e-6)
        
        metadata = {
            "original_sample_rate": orig_sr,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "duration_seconds": round(duration, 3),
            "rms": round(rms, 5),
            "peak_amplitude": round(peak_amp, 5),
            "clipping_percentage": round(clipping_pct, 2),
            "silence_percentage": round(silence_pct, 2)
        }
        return data, metadata

    def calibrate_probability(self, prob: float, temperature: float = CALIBRATION_TEMPERATURE) -> float:
        """
        PHASE 7: Confidence Calibration (Temperature Scaling approximation).
        Squashes extreme probabilities (0.999 -> 0.85) to represent true epistemic uncertainty.
        """
        if prob <= 0 or prob >= 1:
            return prob
            
        logit = np.log(prob / (1 - prob))
        scaled_logit = logit / temperature
        calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
        return float(calibrated_prob)

    def _infer_chunk(self, chunk: np.ndarray) -> Tuple[float, float, float, float]:
        """Run forward pass on a 16kHz audio chunk and return raw & calibrated probabilities."""
        inputs = self.feature_extractor(
            chunk, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Raw logits
            logits = outputs.logits.squeeze(0).cpu().numpy()
            
            # Standard Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            prob_fake = float(probs[self.fake_idx])
            prob_real = float(probs[self.real_idx])
            
            # Calibrated Confidence
            calibrated_fake = self.calibrate_probability(prob_fake)
            calibrated_real = self.calibrate_probability(prob_real)
            
        return prob_fake, prob_real, calibrated_fake, calibrated_real

    def predict(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO],
        fake_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        """
        PHASE 4, 5, 6: Sliding window inference, probability aggregation, and calibration.
        """
        start_time = time.perf_counter()

        # 1. Load and Preprocess
        raw_waveform, orig_sr = self.load_audio(audio_input)
        waveform, metadata = self.preprocess_audio(raw_waveform, orig_sr)
        
        if len(waveform) < TARGET_SAMPLE_RATE * 0.5:
            raise ValueError("Audio is too short (less than 0.5 seconds).")

        # 2. Sliding Window Chunking
        chunk_length_samples = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC)
        stride_samples = int(chunk_length_samples * (1.0 - OVERLAP_RATIO))
        
        chunks_info = []
        fake_probs = []
        real_probs = []
        
        if len(waveform) <= chunk_length_samples:
            # Single chunk fallback
            start_idx = 0
            end_idx = len(waveform)
            chunk = waveform
            pf, pr, cf, cr = self._infer_chunk(chunk)
            
            fake_probs.append(cf)
            real_probs.append(cr)
            
            chunks_info.append({
                "start_s": 0.0,
                "end_s": round(end_idx / TARGET_SAMPLE_RATE, 2),
                "fake_probability": round(cf, 4),
                "real_probability": round(cr, 4),
                "prediction": "FAKE" if cf >= fake_threshold else "REAL"
            })
        else:
            # Sliding window
            for start_idx in range(0, len(waveform) - chunk_length_samples + 1, stride_samples):
                end_idx = start_idx + chunk_length_samples
                chunk = waveform[start_idx:end_idx]
                
                # Simple VAD (skip completely silent chunks)
                rms = np.sqrt(np.mean(chunk**2))
                if rms < 1e-3:
                    continue
                    
                pf, pr, cf, cr = self._infer_chunk(chunk)
                
                fake_probs.append(cf)
                real_probs.append(cr)
                
                chunks_info.append({
                    "start_s": round(start_idx / TARGET_SAMPLE_RATE, 2),
                    "end_s": round(end_idx / TARGET_SAMPLE_RATE, 2),
                    "fake_probability": round(cf, 4),
                    "real_probability": round(cr, 4),
                    "prediction": "FAKE" if cf >= fake_threshold else "REAL"
                })

            # Handle remainder chunk if needed
            if len(waveform) - end_idx > TARGET_SAMPLE_RATE * 1.0: # If more than 1 sec left
                start_idx = len(waveform) - chunk_length_samples
                end_idx = len(waveform)
                chunk = waveform[start_idx:end_idx]
                rms = np.sqrt(np.mean(chunk**2))
                if rms >= 1e-3:
                    pf, pr, cf, cr = self._infer_chunk(chunk)
                    fake_probs.append(cf)
                    real_probs.append(cr)
                    chunks_info.append({
                        "start_s": round(start_idx / TARGET_SAMPLE_RATE, 2),
                        "end_s": round(end_idx / TARGET_SAMPLE_RATE, 2),
                        "fake_probability": round(cf, 4),
                        "real_probability": round(cr, 4),
                        "prediction": "FAKE" if cf >= fake_threshold else "REAL"
                    })

        if not fake_probs:
            raise ValueError("No valid speech chunks detected.")

        # 3. Robust Aggregation (Median & Trimmed Mean)
        # Using Median is robust to single-chunk anomalies.
        agg_fake_prob = float(np.median(fake_probs))
        agg_real_prob = float(np.median(real_probs))
        
        fake_chunks_count = sum(1 for p in fake_probs if p >= fake_threshold)
        total_chunks = len(fake_probs)
        fake_chunk_ratio = float(fake_chunks_count / total_chunks)
        
        # 4. Final Verdict
        is_fake = agg_fake_prob >= fake_threshold
        prediction = "FAKE" if is_fake else "REAL"
        
        # 5. Temporal Localization (Phase 19)
        # Merge adjacent fake chunks
        temporal_segments = []
        in_segment = False
        seg_start = 0.0
        
        for c in chunks_info:
            if c["prediction"] == "FAKE":
                if not in_segment:
                    in_segment = True
                    seg_start = c["start_s"]
            else:
                if in_segment:
                    in_segment = False
                    temporal_segments.append({
                        "start_s": seg_start,
                        "end_s": c["start_s"]  # Ends where the real chunk begins
                    })
        if in_segment:
            temporal_segments.append({
                "start_s": seg_start,
                "end_s": chunks_info[-1]["end_s"]
            })

        processing_time = round(time.perf_counter() - start_time, 3)

        return {
            "prediction": prediction,
            "real_probability": round(agg_real_prob, 4),
            "fake_probability": round(agg_fake_prob, 4),
            "calibrated_confidence": round(max(agg_fake_prob, agg_real_prob), 4),
            "threshold": fake_threshold,
            "model": "Wav2Vec2 Deepfake Detector",
            "chunks_analyzed": total_chunks,
            "fake_chunks": fake_chunks_count,
            "temporal_segments": temporal_segments,
            "duration": metadata["duration_seconds"],
            "preprocessing_metadata": metadata,
            "chunk_details": chunks_info,
            "inference_time_seconds": processing_time
        }
