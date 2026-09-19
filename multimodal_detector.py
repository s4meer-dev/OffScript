"""
Unified Multimodal Audio-Visual Deepfake Detection & Localization Engine.
Integrates the Wav2Vec2 audio detection architecture with the Computer Vision
and temporal localization benchmarks from AV-Deepfake1M.

Supports:
- Pure Audio Files (WAV, MP3, FLAC, OGG, M4A, etc.)
- Video Containers (MP4, WebM, AVI, MOV, MKV, etc.)
- AV-Deepfake1M 4-Class Taxonomy (real, audio_modified, visual_modified, both_modified)
- Cross-modal temporal tampering localization (fake_segments, audio_fake_segments, visual_fake_segments)
"""

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from detector import DeepfakeAudioDetector
from vision_detector import DeepfakeVisionDetector

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


class UnifiedDeepfakeDetector:
    """
    Multimodal Deepfake Detector coordinating Speech (Wav2Vec2) and Computer Vision (CV / Face / Temporal)
    analysis with AV-Deepfake1M benchmark compliance.
    """

    def __init__(
        self,
        audio_model_path: Optional[str] = None,
        device: Optional[str] = None,
        audio_fake_threshold: float = 0.85,
        visual_fake_threshold: float = 0.65,
    ):
        print("Initializing Unified Multimodal Deepfake Detector...")
        start_time = time.perf_counter()

        self.audio_detector = DeepfakeAudioDetector(model_path_or_id=audio_model_path, device=device)
        self.vision_detector = DeepfakeVisionDetector(device=device, confidence_threshold=visual_fake_threshold)

        self.audio_threshold = audio_fake_threshold
        self.visual_threshold = visual_fake_threshold
        self.device = self.audio_detector.device

        elapsed = time.perf_counter() - start_time
        print(f"UnifiedDeepfakeDetector initialized in {elapsed:.2f}s on device '{self.device}'.")

    @staticmethod
    def is_video_file(file_or_path: Union[str, Path, bytes], filename: Optional[str] = None) -> bool:
        """
        Check whether the input file is a video container.
        """
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                return True
            if ext in AUDIO_EXTENSIONS:
                return False

        if isinstance(file_or_path, (str, Path)):
            ext = Path(file_or_path).suffix.lower()
            return ext in VIDEO_EXTENSIONS

        if isinstance(file_or_path, (bytes, bytearray)):
            # Inspect first 16 bytes for common video magic headers (ftyp, matroska/webm, RIFF/AVI)
            header = file_or_path[:64]
            if b"ftyp" in header or b"moov" in header or b"\x1a\x45\xdf\xa3" in header or b"AVI " in header:
                return True

        return False

    def _merge_temporal_segments(
        self, audio_segments: List[List[float]], visual_segments: List[List[float]]
    ) -> List[List[float]]:
        """
        Merge overlapping or adjacent intervals across both modalities.
        """
        all_intervals = sorted(audio_segments + visual_segments, key=lambda s: s[0])
        if not all_intervals:
            return []

        merged: List[List[float]] = []
        for start, end in all_intervals:
            if not merged:
                merged.append([start, end])
            else:
                prev_start, prev_end = merged[-1]
                if start <= prev_end + 0.5:
                    merged[-1] = [prev_start, max(prev_end, end)]
                else:
                    merged.append([start, end])

        return [[round(s[0], 2), round(s[1], 2)] for s in merged]

    def _extract_audio_segments_from_windows(
        self, audio_result: Dict[str, Any]
    ) -> List[List[float]]:
        """
        Derive audio fake segment timestamps from window analysis if present.
        """
        if not audio_result.get("is_fake", False):
            return []

        windows = audio_result.get("window_analysis", {}).get("windows", [])
        if not windows:
            # Entire duration is treated as affected if audio is fake
            duration = audio_result.get("duration_seconds", 0.0)
            return [[0.0, round(duration, 2)]] if duration > 0 else []

        segments: List[List[float]] = []
        in_segment = False
        start_t = 0.0

        for w in windows:
            is_w_fake = w.get("prob_fake", 0.0) >= self.audio_threshold
            t_start = w.get("start", 0.0)
            t_end = w.get("end", 0.0)

            if is_w_fake:
                if not in_segment:
                    in_segment = True
                    start_t = t_start
            else:
                if in_segment:
                    in_segment = False
                    segments.append([round(start_t, 2), round(t_end, 2)])

        if in_segment and windows:
            segments.append([round(start_t, 2), round(windows[-1].get("end", 0.0), 2)])

        return segments if segments else [[0.0, round(audio_result.get("duration_seconds", 0.0), 2)]]

    def predict(
        self,
        media_input: Union[str, Path, bytes],
        filename: Optional[str] = None,
        audio_threshold: Optional[float] = None,
        visual_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Unified classification and temporal localization for audio or video media.
        """
        a_thresh = audio_threshold if audio_threshold is not None else self.audio_threshold
        v_thresh = visual_threshold if visual_threshold is not None else self.visual_threshold
        start_time = time.perf_counter()

        is_video = self.is_video_file(media_input, filename=filename)

        if not is_video:
            # ==========================================
            # PURE AUDIO MEDIA PIPELINE
            # ==========================================
            audio_res = self.audio_detector.predict(media_input, fake_threshold=a_thresh)
            audio_fake = audio_res.get("is_fake", False)
            audio_conf = audio_res.get("confidence", 0.5)

            audio_fake_segs = self._extract_audio_segments_from_windows(audio_res)

            # Map to AV-Deepfake1M taxonomy
            av_label = "audio_modified" if audio_fake else "real"
            overall_prediction = "fake" if audio_fake else ("uncertain" if audio_res.get("prediction") == "uncertain_ambient" else "real")
            
            verdict_text = (
                "AI Synthetic / Voice Clone Detected (Audio)"
                if audio_fake
                else "Authentic Human Speech (Audio)"
            )

            elapsed = round(time.perf_counter() - start_time, 3)

            return {
                "status": "success",
                "media_type": "audio",
                "overall_verdict": av_label,
                "overall_prediction": overall_prediction,
                "overall_confidence": audio_conf,
                "is_fake": bool(audio_fake),
                "verdict_title": verdict_text,
                "av_deepfake1m_classification": {
                    "label": av_label,
                    "description": "Synthetic audio clone" if audio_fake else "Authentic audio",
                    "taxonomy": ["real", "audio_modified", "visual_modified", "both_modified"],
                },
                "audio_analysis": {
                    **audio_res,
                    "audio_fake_segments": audio_fake_segs,
                },
                "visual_analysis": {
                    "status": "skipped",
                    "note": "Visual analysis not applicable for pure audio media.",
                },
                "fake_segments": audio_fake_segs,
                "audio_fake_segments": audio_fake_segs,
                "visual_fake_segments": [],
                "duration_seconds": audio_res.get("duration_seconds", 0.0),
                "inference_time_seconds": elapsed,
            }

        # ==============================================
        # MULTIMODAL VIDEO PIPELINE (AUDIO + VISUAL)
        # ==============================================
        # Save temp file if raw bytes to share across OpenCV and PyAV
        temp_video_path = None
        if isinstance(media_input, (bytes, bytearray)):
            suffix = Path(filename).suffix if filename else ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(media_input)
                temp_video_path = f.name
            target_source = temp_video_path
        else:
            target_source = str(media_input)

        try:
            # 1. Computer Vision Analysis
            vision_res = self.vision_detector.predict(target_source, fake_threshold=v_thresh)
            visual_fake = vision_res.get("is_fake", False)
            visual_conf = vision_res.get("confidence", 0.5)
            visual_fake_segs = vision_res.get("visual_fake_segments", [])

            # 2. Audio Stream Extraction & Analysis
            audio_fake = False
            audio_conf = 0.5
            audio_res = None
            audio_fake_segs = []
            has_audio = True

            try:
                audio_res = self.audio_detector.predict(target_source, fake_threshold=a_thresh)
                audio_fake = audio_res.get("is_fake", False)
                audio_conf = audio_res.get("confidence", 0.5)
                audio_fake_segs = self._extract_audio_segments_from_windows(audio_res)
            except Exception as e:
                # Video file has no audio track or decoding failed
                has_audio = False
                audio_res = {
                    "status": "no_audio_stream",
                    "note": f"Video does not contain an extractable audio track ({e})",
                    "is_fake": False,
                    "confidence": 1.0,
                    "probabilities": {"fake": 0.0, "real": 1.0},
                    "prediction": "silent",
                }

            # 3. Multimodal Fusion (AV-Deepfake1M Standard)
            if has_audio:
                if audio_fake and visual_fake:
                    av_label = "both_modified"
                    verdict_title = "Full Audio-Visual Deepfake (Both Video & Voice Manipulated)"
                    overall_prediction = "fake"
                    overall_conf = round(max(audio_conf, visual_conf), 4)
                elif visual_fake and not audio_fake:
                    av_label = "visual_modified"
                    verdict_title = "Face / Visual Deepfake (Manipulated Video, Authentic Speech)"
                    overall_prediction = "fake"
                    overall_conf = round(visual_conf, 4)
                elif audio_fake and not visual_fake:
                    av_label = "audio_modified"
                    verdict_title = "Synthetic Voice Clone (Manipulated Audio, Authentic Video)"
                    overall_prediction = "fake"
                    overall_conf = round(audio_conf, 4)
                else:
                    av_label = "real"
                    verdict_title = "Authentic Media (Authentic Video & Authentic Voice)"
                    overall_prediction = "real"
                    overall_conf = round(min(audio_conf, visual_conf), 4)
            else:
                if visual_fake:
                    av_label = "visual_modified"
                    verdict_title = "Face / Visual Deepfake (Silent Video)"
                    overall_prediction = "fake"
                    overall_conf = round(visual_conf, 4)
                else:
                    av_label = "real"
                    verdict_title = "Authentic Video (Silent Media)"
                    overall_prediction = "real"
                    overall_conf = round(visual_conf, 4)

            # Combined temporal segments
            fake_segments = self._merge_temporal_segments(audio_fake_segs, visual_fake_segs)

            # Cross-modal timeline events for UI
            timeline_events = []
            for seg in audio_fake_segs:
                timeline_events.append({
                    "start": seg[0],
                    "end": seg[1],
                    "modality": "audio",
                    "description": "Synthetic Speech / Audio Artifact",
                })
            for seg in visual_fake_segs:
                timeline_events.append({
                    "start": seg[0],
                    "end": seg[1],
                    "modality": "visual",
                    "description": "Facial Manipulation / Blending Discontinuity",
                })

            timeline_events.sort(key=lambda x: x["start"])

            elapsed = round(time.perf_counter() - start_time, 3)

            return {
                "status": "success",
                "media_type": "video",
                "overall_verdict": av_label,
                "overall_prediction": overall_prediction,
                "overall_confidence": overall_conf,
                "is_fake": bool(overall_prediction == "fake"),
                "verdict_title": verdict_title,
                "av_deepfake1m_classification": {
                    "label": av_label,
                    "description": verdict_title,
                    "taxonomy": ["real", "audio_modified", "visual_modified", "both_modified"],
                },
                "audio_analysis": {
                    **audio_res,
                    "audio_fake_segments": audio_fake_segs,
                },
                "visual_analysis": {
                    **vision_res,
                    "visual_fake_segments": visual_fake_segs,
                },
                "fake_segments": fake_segments,
                "audio_fake_segments": audio_fake_segs,
                "visual_fake_segments": visual_fake_segs,
                "timeline_events": timeline_events,
                "duration_seconds": vision_res.get("video_metadata", {}).get("duration_seconds", 0.0),
                "inference_time_seconds": elapsed,
            }

        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass
