from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api", tags=["System & Health"])


@router.get("/health", summary="Health Check")
async def health_check(request: Request):
    detector = getattr(request.app.state, "detector", None)
    return {
        "status": "online",
        "model_ready": detector is not None,
        "device": str(detector.device) if detector else "uninitialized",
        "multimodal_ready": True,
        "modalities": ["audio", "video", "audio_visual"],
        "frameworks": ["Wav2Vec2", "AV-Deepfake1M (Computer Vision & Temporal Localization)"],
    }


@router.get("/info", summary="Model & Architecture Information")
async def model_info(request: Request):
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")
    return {
        "system_name": "Unified Multimodal Audio-Visual Deepfake Detection & Localization Platform",
        "audio_model": {
            "name": "mo-thecreator/Deepfake-audio-detection",
            "architecture": detector.audio_detector.model.config.architectures,
            "target_sample_rate": 16000,
            "labels": detector.audio_detector.model.config.id2label,
        },
        "visual_model": {
            "name": "DeepfakeVisionDetector",
            "architecture": "Biometric Facial ROI, Spatial Laplacian Boundary Analysis, 2D FFT Frequency, Temporal Motion Flicker, MobileNet",
            "target_fps": detector.vision_detector.target_fps,
            "max_frames": detector.vision_detector.max_frames,
        },
        "taxonomy": {
            "standard": "AV-Deepfake1M 4-Class Taxonomy",
            "classes": ["real", "audio_modified", "visual_modified", "both_modified"],
        },
        "temporal_localization": {
            "supported": True,
            "outputs": ["fake_segments", "audio_fake_segments", "visual_fake_segments"],
        },
        "device": str(detector.device),
    }
