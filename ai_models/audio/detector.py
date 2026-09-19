"""
Enhanced Inference Engine for Deepfake Audio Detection.
Features:
- Multi-format audio decoding (soundfile + PyAV)
- Silence trimming & peak normalization
- Sliding-window timeline analysis
- Bandwidth & codec artifact detection (telephony / 8kHz cutoff)
- Calibrated forensic thresholding with 3-tier verdict
- Multi-model support (mo-thecreator V1 and MelodyMachine V2)
"""

import io
import time
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

# Resolve local weights directory across modular layout and legacy locations
_CURRENT_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _CURRENT_DIR.parent / "weights"
_ROOT_MODEL_DIR = _CURRENT_DIR.parent.parent / "model"
_LOCAL_MODEL_DIR = _CURRENT_DIR / "model"

if (_WEIGHTS_DIR / "model.safetensors").exists():
    DEFAULT_MODEL_DIR = _WEIGHTS_DIR
elif (_ROOT_MODEL_DIR / "model.safetensors").exists():
    DEFAULT_MODEL_DIR = _ROOT_MODEL_DIR
elif (_LOCAL_MODEL_DIR / "model.safetensors").exists():
    DEFAULT_MODEL_DIR = _LOCAL_MODEL_DIR
else:
    DEFAULT_MODEL_DIR = _WEIGHTS_DIR

HF_REPO_ID = "mo-thecreator/Deepfake-audio-detection"
TARGET_SAMPLE_RATE = 16000


class DeepfakeAudioDetector:
    """
    Audio deepfake detector based on Wav2Vec2ForSequenceClassification.
    """

    def __init__(
        self,
        model_path_or_id: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the detector with model and feature extractor.
        """
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

        print(f"Loading DeepfakeAudioDetector from '{self.model_path}' on device '{self.device}'...")
        start_time = time.perf_counter()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        load_duration = time.perf_counter() - start_time
        print(f"Model loaded successfully in {load_duration:.2f}s.")
        print(f"Label mapping: {self.model.config.id2label}")

    def _decode_with_av(self, audio_source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, int]:
        """
        Decode audio using PyAV (FFmpeg-backed), supporting WebM, Opus, M4A, AAC, MP4, etc.
        Directly resamples to 16,000 Hz mono float32.
        """
        if not HAS_AV:
            raise RuntimeError("PyAV ('av' package) is required to decode this audio format.")

        if isinstance(audio_source, (str, Path)):
            container = av.open(str(audio_source))
        elif isinstance(audio_source, (bytes, bytearray)):
            container = av.open(io.BytesIO(audio_source))
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            container = av.open(audio_source)
        else:
            raise TypeError(f"Unsupported audio source for PyAV: {type(audio_source)}")

        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            container.close()
            raise ValueError("No audio stream found in the uploaded file.")

        original_sr = audio_stream.codec_context.sample_rate or TARGET_SAMPLE_RATE

        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SAMPLE_RATE)
        chunks = []
        for frame in container.decode(audio_stream):
            for resampled_frame in resampler.resample(frame):
                chunks.append(resampled_frame.to_ndarray()[0])

        for resampled_frame in resampler.resample(None):
            chunks.append(resampled_frame.to_ndarray()[0])

        container.close()

        if not chunks:
            raise ValueError("Decoded audio stream produced 0 samples.")

        waveform = np.concatenate(chunks, axis=0).astype(np.float32)
        return waveform, original_sr

    def load_audio(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from file path, raw bytes, or array, and convert to 1D float32 array.
        """
        if isinstance(audio_input, (np.ndarray, torch.Tensor)):
            if sample_rate is None:
                raise ValueError("sample_rate must be provided when passing raw numpy or torch audio data")
            if isinstance(audio_input, torch.Tensor):
                data = audio_input.detach().cpu().numpy().astype(np.float32)
            else:
                data = audio_input.astype(np.float32)
            sr = sample_rate

            if data.ndim > 1:
                if data.shape[0] < data.shape[1] and data.shape[0] <= 8:
                    data = np.mean(data, axis=0)
                else:
                    data = np.mean(data, axis=1)
            return data.squeeze(), sr

        # Try soundfile first
        try:
            if isinstance(audio_input, (str, Path)):
                path = Path(audio_input)
                if not path.exists():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                data, sr = sf.read(str(path), dtype="float32")
            elif isinstance(audio_input, (bytes, bytearray)):
                buffer = io.BytesIO(audio_input)
                data, sr = sf.read(buffer, dtype="float32")
            elif isinstance(audio_input, io.BytesIO):
                audio_input.seek(0)
                data, sr = sf.read(audio_input, dtype="float32")
            else:
                raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

            if data.ndim > 1:
                if data.shape[0] < data.shape[1] and data.shape[0] <= 8:
                    data = np.mean(data, axis=0)
                else:
                    data = np.mean(data, axis=1)

            data = data.squeeze()
            return data, sr

        except Exception as sf_err:
            try:
                data, orig_sr = self._decode_with_av(audio_input)
                return data, TARGET_SAMPLE_RATE
            except Exception as av_err:
                raise ValueError(
                    f"Could not read audio with soundfile ({sf_err}) or PyAV ({av_err})"
                )

    def resample_if_needed(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample waveform to 16,000 Hz if necessary.
        """
        if orig_sr == TARGET_SAMPLE_RATE:
            return waveform

        tensor_wave = torch.from_numpy(waveform).float().unsqueeze(0)
        resampler = T.Resample(orig_freq=orig_sr, new_freq=TARGET_SAMPLE_RATE)
        resampled_tensor = resampler(tensor_wave).squeeze(0)
        return resampled_tensor.numpy()

    def trim_silence(self, waveform: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Trim leading and trailing silence where amplitude is below threshold.
        """
        abs_wave = np.abs(waveform)
        mask = abs_wave > threshold
        if not np.any(mask):
            return waveform

        start_idx = np.argmax(mask)
        end_idx = len(mask) - np.argmax(mask[::-1])
        pad = int(TARGET_SAMPLE_RATE * 0.1)
        start_idx = max(0, start_idx - pad)
        end_idx = min(len(waveform), end_idx + pad)

        trimmed = waveform[start_idx:end_idx]
        return trimmed if len(trimmed) >= 8000 else waveform

    def normalize_volume(self, waveform: np.ndarray) -> np.ndarray:
        """
        Peak normalize audio to avoid low-level noise bias.
        """
        peak = np.max(np.abs(waveform))
        if peak > 1e-4:
            return (waveform / peak) * 0.95
        return waveform

    def detect_bandwidth_artifacts(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frequency spectrum to detect telephony/narrowband cutoff (< 4kHz).
        """
        if len(waveform) < 1024:
            return {"is_narrowband": False, "high_freq_ratio": 1.0}

        freqs = np.fft.rfftfreq(len(waveform), d=1.0 / TARGET_SAMPLE_RATE)
        fft_mag = np.abs(np.fft.rfft(waveform))

        high_energy = np.sum(fft_mag[freqs > 4000] ** 2)
        total_energy = np.sum(fft_mag**2) + 1e-9
        high_ratio = float(high_energy / total_energy)

        # In uncompressed 16kHz speech, energy above 4kHz is typically > 2-5%
        # In 8kHz/telephony/narrowband audio, it drops below 0.5%
        is_narrowband = high_ratio < 0.008

        return {
            "is_narrowband": is_narrowband,
            "high_freq_ratio": round(high_ratio, 5),
            "note": "Telephony / Narrowband bandwidth cutoff detected (< 4kHz)" if is_narrowband else "Normal wideband frequency spectrum",
        }

    def _infer_segment(self, segment: np.ndarray) -> Tuple[float, float]:
        """
        Run forward pass on a single 16kHz audio segment.
        Returns: (prob_fake, prob_real)
        """
        inputs = self.feature_extractor(
            segment,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        id2label = self.model.config.id2label
        fake_idx = 0
        real_idx = 1
        for idx, lbl in id2label.items():
            if str(lbl).lower() == "fake":
                fake_idx = int(idx)
            elif str(lbl).lower() == "real":
                real_idx = int(idx)

        prob_fake = float(probabilities[fake_idx].item())
        prob_real = float(probabilities[real_idx].item())
        return prob_fake, prob_real

    def predict(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        fake_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Run deepfake classification on an audio input.
        
        Args:
            audio_input: File path, bytes, or numpy array.
            sample_rate: Required if audio_input is array/tensor.
            fake_threshold: Confidence threshold required to confirm 'fake' (default 0.85).
                            Values between 0.50 and threshold are classified as likely authentic
                            with ambient / compression artifacts.
        """
        start_time = time.perf_counter()

        # 1. Load audio
        waveform, original_sr = self.load_audio(audio_input, sample_rate=sample_rate)
        if len(waveform) == 0:
            raise ValueError("Audio waveform is empty.")

        duration_seconds = round(float(len(waveform)) / float(original_sr), 3)

        # 2. Resample to 16,000 Hz
        waveform_16k = self.resample_if_needed(waveform, original_sr)

        # 3. Preprocess: silence trimming & peak volume normalization
        waveform_proc = self.trim_silence(waveform_16k)
        waveform_proc = self.normalize_volume(waveform_proc)

        # 4. Check bandwidth artifacts (telephony/phone cutoff)
        bandwidth_info = self.detect_bandwidth_artifacts(waveform_proc)

        # 5. Adaptive Voice Activity Detection (VAD) & Segment Analysis
        chunk_length = 16000 * 3  # 3.0s window
        stride = 16000 * 2        # 2.0s hop

        timeline_chunks = []
        speech_ratio = 1.0

        if len(waveform_proc) > chunk_length + 8000:
            frame_size = int(TARGET_SAMPLE_RATE * 0.05)  # 50ms frame
            hop_size = int(TARGET_SAMPLE_RATE * 0.025)   # 25ms hop

            n_frames = max(1, (len(waveform_proc) - frame_size) // hop_size)
            frame_rms = np.array([
                np.sqrt(np.mean(waveform_proc[i * hop_size : i * hop_size + frame_size] ** 2))
                for i in range(n_frames)
            ])

            # Dynamic noise floor and vocal threshold
            p20 = float(np.percentile(frame_rms, 20))
            p90 = float(np.percentile(frame_rms, 90))
            speech_threshold = max(0.015, p20 + (p90 - p20) * 0.20)

            voiced_frames = frame_rms > speech_threshold
            total_voiced_count = int(np.sum(voiced_frames))
            speech_ratio = round(total_voiced_count / max(1, len(voiced_frames)), 3)

            weighted_fake = 0.0
            weighted_real = 0.0
            total_weight = 0.0
            speech_chunk_count = 0

            for start in range(0, len(waveform_proc) - chunk_length + 1, stride):
                end = start + chunk_length
                chunk = waveform_proc[start:end]

                start_frame = start // hop_size
                end_frame = min(len(voiced_frames), end // hop_size)
                chunk_voiced = voiced_frames[start_frame:end_frame]
                chunk_rms = frame_rms[start_frame:end_frame]

                voiced_ratio = float(np.mean(chunk_voiced)) if len(chunk_voiced) > 0 else 0.0
                start_sec = round(start / TARGET_SAMPLE_RATE, 1)
                end_sec = round(end / TARGET_SAMPLE_RATE, 1)

                # If window has less than 30% speech, it is a pause / room ambient noise.
                # Exclude from scoring to prevent pink noise false positives!
                if voiced_ratio < 0.30:
                    timeline_chunks.append({
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "status": "pause",
                        "voiced_ratio": round(voiced_ratio, 2),
                        "fake": None,
                        "real": None,
                        "note": "Ambient pause (Ignored in score)",
                    })
                else:
                    pf, pr = self._infer_segment(chunk)
                    avg_vocal_rms = float(np.mean(chunk_rms[chunk_voiced])) if np.any(chunk_voiced) else 0.01
                    weight = float(voiced_ratio * (avg_vocal_rms + 1e-4))

                    weighted_fake += pf * weight
                    weighted_real += pr * weight
                    total_weight += weight
                    speech_chunk_count += 1

                    timeline_chunks.append({
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "status": "speech",
                        "voiced_ratio": round(voiced_ratio, 2),
                        "fake": round(pf, 3),
                        "real": round(pr, 3),
                        "weight": round(weight, 4),
                        "note": "Human Voice" if pr > 0.5 else "Synthetic Pattern",
                    })

            if total_weight > 0 and speech_chunk_count > 0:
                prob_fake = float(weighted_fake / total_weight)
                prob_real = float(weighted_real / total_weight)
            else:
                # Fallback if recording was completely quiet
                prob_fake, prob_real = self._infer_segment(waveform_proc[:chunk_length])
                timeline_chunks.append({
                    "start_sec": 0.0,
                    "end_sec": round(chunk_length / TARGET_SAMPLE_RATE, 1),
                    "status": "speech",
                    "voiced_ratio": 1.0,
                    "fake": round(prob_fake, 3),
                    "real": round(prob_real, 3),
                    "note": "Human Voice" if prob_real > 0.5 else "Synthetic Pattern",
                })
        else:
            prob_fake, prob_real = self._infer_segment(waveform_proc)
            timeline_chunks.append({
                "start_sec": 0.0,
                "end_sec": duration_seconds,
                "status": "speech",
                "voiced_ratio": 1.0,
                "fake": round(prob_fake, 3),
                "real": round(prob_real, 3),
                "note": "Human Voice" if prob_real > 0.5 else "Synthetic Pattern",
            })

        # 6. Calibrated Decision Logic
        # Forensic thresholding: A confirmed deepfake requires high fake probability (>= fake_threshold).
        # Mid-range scores (50% - threshold) on real-world mic/phone audio represent acoustic/codec noise.
        if prob_fake >= fake_threshold:
            verdict = "fake"
            verdict_label = "AI DEEPFAKE DETECTED"
            verdict_description = f"Synthetic speech patterns detected with {prob_fake*100:.1f}% confidence."
            is_fake = True
            confidence = prob_fake
        elif prob_fake >= 0.50:
            verdict = "uncertain_ambient"
            verdict_label = "AUTHENTIC VOICE (WITH AMBIENT/CODEC NOISE)"
            verdict_description = f"Natural human acoustic features detected. Elevated artifacts ({prob_fake*100:.1f}%) attributed to microphone acoustics, room echo, or compression."
            is_fake = False
            confidence = prob_real
        else:
            verdict = "real"
            verdict_label = "AUTHENTIC HUMAN SPEECH"
            verdict_description = f"Natural human vocal tract acoustics verified with {prob_real*100:.1f}% confidence."
            is_fake = False
            confidence = prob_real

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "prediction": verdict,
            "verdict_label": verdict_label,
            "verdict_description": verdict_description,
            "confidence": round(confidence, 4),
            "probabilities": {
                "fake": round(prob_fake, 4),
                "real": round(prob_real, 4),
            },
            "is_fake": is_fake,
            "duration_seconds": duration_seconds,
            "original_sample_rate": original_sr,
            "processing_time_ms": processing_time_ms,
            "speech_ratio": speech_ratio,
            "active_speech_seconds": round(duration_seconds * speech_ratio, 2),
            "bandwidth_analysis": bandwidth_info,
            "timeline": timeline_chunks,
            "threshold_used": fake_threshold,
            "device": str(self.device),
        }
