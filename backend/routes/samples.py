from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/samples", tags=["Sample Benchmarks"])

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


@router.get("/{filename}", summary="Download Demo Sample")
async def get_sample_file(filename: str):
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    media_type = "video/mp4" if filename.endswith((".mp4", ".webm")) else "audio/wav"
    return FileResponse(file_path, media_type=media_type)
