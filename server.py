"""
FastAPI Server Entrypoint for Multimodal Audio-Visual Deepfake Detection & Localization.
Re-exports the application from backend.main for backward compatibility.
"""

from backend.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
