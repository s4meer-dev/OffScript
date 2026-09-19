"""
FastAPI Server for Multimodal Audio-Visual Deepfake Detection & Localization.
Connects the backend API endpoints with the modular AI models and serves the frontend UI.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_models.multimodal.multimodal_detector import UnifiedDeepfakeDetector
from backend.routes.predict import router as predict_router
from backend.routes.health import router as health_router
from backend.routes.samples import router as samples_router
from backend.routes.train import router as train_router

FRONTEND_DIR = ROOT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Unified Deepfake Detector on startup...")
    app.state.detector = UnifiedDeepfakeDetector()
    print("Unified Detector ready for incoming requests.")
    yield
    print("Shutting down Deepfake Detector service.")


app = FastAPI(
    title="Unified Audio-Visual Deepfake Detection API",
    description="Multimodal deepfake detection and temporal localization powered by Wav2Vec2 and AV-Deepfake1M benchmarks",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(samples_router)
app.include_router(train_router)

from backend.routes.health import health_check
app.add_api_route("/health", health_check, methods=["GET"], include_in_schema=False)



@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# Mount static assets and serve Frontend UI
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", summary="Frontend Web Interface", include_in_schema=False)
    async def index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return Response(content="Frontend index.html not found.", media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
