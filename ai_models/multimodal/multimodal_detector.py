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

import numpy as np

try:
    from ai_models.audio.detector import DeepfakeAudioDetector
    from ai_models.vision.vision_detector import DeepfakeVisionDetector
except ImportError:
    try:
        from ..audio.detector import DeepfakeAudioDetector
        from ..vision.vision_detector import DeepfakeVisionDetector
    except (ImportError, ValueError):
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
            # Inspect first 64 bytes for common video magic headers (ftyp, matroska/webm, RIFF/AVI)
            header = file_or_path[:64]
            if b"ftyp" in header or b"moov" in header or b"\x1a\x45\xdf\xa3" in header or b"AVI " in header:
                return True

        if hasattr(file_or_path, "read"):
            pos = file_or_path.tell() if hasattr(file_or_path, "tell") else 0
            header = file_or_path.read(64)
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(pos)
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

    def _check_has_audio_stream(self, file_path: str) -> bool:
        """
        Quickly inspect container metadata to verify if an audio stream exists.
        """
        try:
            import av
            container = av.open(str(file_path))
            has_audio = any(s.type == "audio" for s in container.streams)
            container.close()
            return has_audio
        except Exception:
            return False

    def _verify_has_audio_content(
        self, source: Any
    ) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[int]]:
        """
        Verify if media contains an audio stream with actual, non-silent sound.
        Returns: (has_audio, error_reason, waveform, sample_rate)
        """
        try:
            waveform, sr = self.audio_detector.load_audio(source)
            if waveform is None or len(waveform) == 0:
                return False, "Audio stream produced 0 samples.", None, None
            peak = float(np.max(np.abs(waveform)))
            if peak <= 1e-4:
                return False, "Audio track is silent (no audible signal).", None, None
            return True, None, waveform, sr
        except Exception as e:
            return False, str(e), None, None

    def predict(
        self,
        media_input: Union[str, Path, bytes, io.BytesIO],
        filename: Optional[str] = None,
        mode: str = "audio",
        audio_threshold: Optional[float] = None,
        visual_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Prediction endpoint supporting explicit mode selection:
        - "audio": Audio Defect Detection ONLY. Video visual analysis is not run and
                   does not exist in reports. Video inputs are only analyzed and presented
                   if an audio track is present; otherwise an error is raised.
        - "video": Video Defect Detection ONLY. Audio analysis is not run and
                   does not exist in reports.
        - "multimodal": Combined audio-visual fusion (backward compatibility).
        """
        start_time = time.perf_counter()
        a_thresh = audio_threshold if audio_threshold is not None else self.audio_threshold
        v_thresh = visual_threshold if visual_threshold is not None else self.visual_threshold
        mode = (mode or "audio").lower().strip()
        if mode not in ["audio", "video", "multimodal"]:
            mode = "audio"

        is_video = self.is_video_file(media_input, filename=filename)

        # Prepare temporary file for video container if input is in-memory
        temp_video_path = None
        target_source = None
        if is_video:
            if isinstance(media_input, (bytes, bytearray)):
                suffix = Path(filename).suffix if filename else ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(media_input)
                    temp_video_path = f.name
                target_source = temp_video_path
            elif hasattr(media_input, "read"):
                suffix = Path(filename).suffix if filename else ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(media_input.read())
                    temp_video_path = f.name
                target_source = temp_video_path
            else:
                target_source = str(media_input)

        try:
            # ==============================================================
            # 1. AUDIO DEFECT DETECTION MODE
            # ==============================================================
            if mode == "audio":
                if is_video:
                    raise ValueError(
                        "Video files are not permitted in Audio Defect Detection mode. "
                        "Please upload an audio file (WAV, MP3, FLAC, OGG, M4A, etc.) or switch to Video Defect Detection mode."
                    )

                has_audio, reason, waveform, sr = self._verify_has_audio_content(media_input)
                if not has_audio:
                    raise ValueError(f"No audio content present in the uploaded audio file ({reason}).")

                # Run audio detector ONLY
                if waveform is not None and sr is not None:
                    audio_res = self.audio_detector.predict(waveform, sample_rate=sr, fake_threshold=a_thresh)
                else:
                    audio_res = self.audio_detector.predict(media_input, fake_threshold=a_thresh)

                # If audio analysis determined silence
                if audio_res.get("prediction") == "silent" or audio_res.get("speech_ratio", 1.0) == 0.0:
                    raise ValueError(
                        "No audible speech detected in media (silent track). "
                        "Under audio defect detection, silent media cannot be detected or presented in reports."
                    )

                audio_fake = audio_res.get("is_fake", False)
                audio_conf = audio_res.get("confidence", 0.5)
                overall_verdict = "audio_modified" if audio_fake else "real"
                overall_prediction = "fake" if audio_fake else "real"
                audio_fake_segs = self._extract_audio_segments_from_windows(audio_res)

                verdict_text = (
                    "AI Synthetic / Voice Clone Detected (Audio)"
                    if audio_fake
                    else "Authentic Human Voice (Audio)"
                )
                elapsed = round(time.perf_counter() - start_time, 3)

                # Pure audio report: video analysis does NOT exist in report
                return {
                    "status": "success",
                    "mode": "audio",
                    "media_type": "audio",
                    "overall_verdict": overall_verdict,
                    "overall_prediction": overall_prediction,
                    "overall_confidence": audio_conf,
                    "is_fake": bool(audio_fake),
                    "verdict_title": verdict_text,
                    "audio_analysis": {
                        **audio_res,
                        "activated": True,
                        "has_audio_stream": True,
                        "audio_fake_segments": audio_fake_segs,
                    },
                    "fake_segments": audio_fake_segs,
                    "audio_fake_segments": audio_fake_segs,
                    "duration_seconds": audio_res.get("duration_seconds", 0.0),
                    "inference_time_seconds": elapsed,
                }

            # ==============================================================
            # 2. VIDEO DEFECT DETECTION MODE
            # ==============================================================
            elif mode == "video":
                if not is_video:
                    raise ValueError(
                        "The uploaded file is not a video container. "
                        "Video defect detection requires a video file (MP4, WebM, AVI, MOV, MKV, etc.)."
                    )

                # Run computer vision detector ONLY
                vision_res = self.vision_detector.predict(target_source, fake_threshold=v_thresh)
                elapsed = round(time.perf_counter() - start_time, 3)

                visual_fake = vision_res.get("is_fake", False)
                visual_conf = vision_res.get("confidence", 0.5)
                visual_fake_segs = vision_res.get("visual_fake_segments", [])
                overall_verdict = "visual_modified" if visual_fake else "real"
                overall_prediction = "fake" if visual_fake else "real"

                verdict_text = vision_res.get("verdict")
                if not verdict_text:
                    verdict_text = "Visual Deepfake Detected (Video)" if visual_fake else "Authentic Video"

                # Pure video report: audio analysis does NOT exist in report
                return {
                    "status": "success",
                    "mode": "video",
                    "media_type": "video",
                    "overall_verdict": overall_verdict,
                    "overall_prediction": overall_prediction,
                    "overall_confidence": visual_conf,
                    "is_fake": bool(visual_fake),
                    "verdict": verdict_text,
                    "verdict_title": verdict_text,
                    "risk_level": vision_res.get("risk_level", "LOW (AUTHENTIC)"),
                    "probabilities": vision_res.get("probabilities", {"real": 0.9, "fake": 0.1}),
                    "visual_analysis": {
                        **vision_res,
                        "visual_fake_segments": visual_fake_segs,
                    },
                    "diagnostic_breakdown": vision_res.get("diagnostic_breakdown", {}),
                    "frame_analysis": vision_res.get("frame_analysis", []),
                    "findings_log": vision_res.get("findings_log", []),
                    "forensic_metrics": vision_res.get("forensic_metrics", {}),
                    "fake_segments": visual_fake_segs,
                    "visual_fake_segments": visual_fake_segs,
                    "faces_detected": vision_res.get("faces_detected", 0),
                    "frames_analyzed": vision_res.get("frames_analyzed", 0),
                    "face_presence_ratio": vision_res.get("face_presence_ratio", 0.0),
                    "video_metadata": vision_res.get("video_metadata", {}),
                    "duration_seconds": vision_res.get("video_metadata", {}).get("duration_seconds", 0.0),
                    "inference_time_seconds": elapsed,
                }

            # ==============================================================
            # 3. MULTIMODAL MODE (BACKWARD COMPATIBILITY)
            # ==============================================================
            else:
                if not is_video:
                    audio_res = self.audio_detector.predict(media_input, fake_threshold=a_thresh)
                    elapsed = round(time.perf_counter() - start_time, 3)
                    audio_fake = audio_res.get("is_fake", False)
                    audio_conf = audio_res.get("confidence", 0.5)
                    av_label = "audio_modified" if audio_fake else "real"
                    overall_prediction = "fake" if audio_fake else "real"
                    audio_fake_segs = self._extract_audio_segments_from_windows(audio_res)
                    verdict_text = (
                        "AI Synthetic / Voice Clone Detected (Audio)"
                        if audio_fake
                        else "Authentic Human Voice (Audio)"
                    )
                    return {
                        "status": "success",
                        "mode": "audio",
                        "media_type": "audio",
                        "overall_verdict": av_label,
                        "overall_prediction": overall_prediction,
                        "overall_confidence": audio_conf,
                        "is_fake": bool(audio_fake),
                        "verdict_title": verdict_text,
                        "audio_analysis": {
                            **audio_res,
                            "activated": True,
                            "has_audio_stream": True,
                            "audio_fake_segments": audio_fake_segs,
                        },
                        "fake_segments": audio_fake_segs,
                        "audio_fake_segments": audio_fake_segs,
                        "duration_seconds": audio_res.get("duration_seconds", 0.0),
                        "inference_time_seconds": elapsed,
                    }

                # Multimodal on video
                vision_res = self.vision_detector.predict(target_source, fake_threshold=v_thresh)
                visual_fake = vision_res.get("is_fake", False)
                visual_conf = vision_res.get("confidence", 0.5)
                visual_fake_segs = vision_res.get("visual_fake_segments", [])

                has_audio, _, waveform, sr = self._verify_has_audio_content(target_source)
                audio_fake = False
                audio_conf = None
                audio_res = None
                audio_fake_segs = []

                if has_audio:
                    try:
                        if waveform is not None and sr is not None:
                            audio_res = self.audio_detector.predict(waveform, sample_rate=sr, fake_threshold=a_thresh)
                        else:
                            audio_res = self.audio_detector.predict(target_source, fake_threshold=a_thresh)

                        speech_ratio = audio_res.get("speech_ratio", 1.0)
                        if speech_ratio == 0.0 or audio_res.get("prediction") == "silent":
                            has_audio = False
                            audio_res = None
                        else:
                            audio_fake = audio_res.get("is_fake", False)
                            audio_conf = audio_res.get("confidence", 0.5)
                            audio_fake_segs = self._extract_audio_segments_from_windows(audio_res)
                            audio_res["activated"] = True
                            audio_res["has_audio_stream"] = True
                            audio_res["audio_fake_segments"] = audio_fake_segs
                    except Exception:
                        has_audio = False
                        audio_res = None

                if has_audio and audio_conf is not None and audio_res is not None:
                    if audio_fake and visual_fake:
                        av_label = "both_modified"
                        verdict_title = "Full Audio-Visual Deepfake (Both Video & Voice Manipulated)"
                        overall_prediction = "fake"
                        overall_conf = round(max(audio_conf, visual_conf), 4)
                    elif visual_fake and not audio_fake:
                        av_label = "visual_modified"
                        vis_desc = vision_res.get("verdict", "Visual Deepfake Detected")
                        verdict_title = f"{vis_desc} (Authentic Speech)"
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

                    fake_segments = self._merge_temporal_segments(audio_fake_segs, visual_fake_segs)
                    elapsed = round(time.perf_counter() - start_time, 3)

                    result_dict = {
                        "status": "success",
                        "mode": "multimodal",
                        "media_type": "video",
                        "overall_verdict": av_label,
                        "overall_prediction": overall_prediction,
                        "overall_confidence": overall_conf,
                        "is_fake": bool(overall_prediction == "fake"),
                        "verdict_title": verdict_title,
                        "audio_analysis": audio_res,
                        "visual_analysis": {
                            **vision_res,
                            "visual_fake_segments": visual_fake_segs,
                        },
                        "fake_segments": fake_segments,
                        "audio_fake_segments": audio_fake_segs,
                        "visual_fake_segments": visual_fake_segs,
                        "duration_seconds": vision_res.get("video_metadata", {}).get("duration_seconds", 0.0),
                        "inference_time_seconds": elapsed,
                    }
                    return result_dict
                else:
                    # Video without active audio: Pure visual report!
                    # Audio does NOT appear in the report when not present.
                    overall_verdict = "visual_modified" if visual_fake else "real"
                    overall_prediction = "fake" if visual_fake else "real"
                    verdict_title = vision_res.get("verdict", "Visual Deepfake Detected" if visual_fake else "Authentic Video")
                    elapsed = round(time.perf_counter() - start_time, 3)

                    return {
                        "status": "success",
                        "mode": "video",
                        "media_type": "video",
                        "overall_verdict": overall_verdict,
                        "overall_prediction": overall_prediction,
                        "overall_confidence": round(visual_conf, 4),
                        "is_fake": bool(overall_prediction == "fake"),
                        "verdict_title": verdict_title,
                        "visual_analysis": {
                            **vision_res,
                            "visual_fake_segments": visual_fake_segs,
                        },
                        "fake_segments": visual_fake_segs,
                        "visual_fake_segments": visual_fake_segs,
                        "duration_seconds": vision_res.get("video_metadata", {}).get("duration_seconds", 0.0),
                        "inference_time_seconds": elapsed,
                    }

        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass
