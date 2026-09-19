from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api", tags=["Inference"])


@router.post("/predict", summary="Classify Audio or Video File")
async def predict_media(
    request: Request,
    file: UploadFile = File(..., description="Media file (WAV, MP3, FLAC, OGG, MP4, WebM, AVI, MOV, etc.)"),
    mode: str = Query("video", description="Detection mode: 'video' or 'audio' (or 'multimodal')"),
    audio_threshold: Optional[float] = Query(None, ge=0.50, le=0.99, description="Audio deepfake threshold"),
    visual_threshold: Optional[float] = Query(None, ge=0.40, le=0.95, description="Visual deepfake threshold"),
    threshold: Optional[float] = Query(None, ge=0.50, le=0.99, description="Legacy threshold alias"),
):
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Resolve threshold parameters
        a_thresh = audio_threshold if audio_threshold is not None else (threshold if threshold is not None else 0.85)
        v_thresh = visual_threshold if visual_threshold is not None else 0.65

        result = detector.predict(
            content,
            filename=file.filename,
            mode=mode,
            audio_threshold=a_thresh,
            visual_threshold=v_thresh,
        )
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process media: {str(e)}")
