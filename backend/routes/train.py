import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/train", tags=["training"])

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"
TRAIN_SCRIPT = ROOT_DIR / "ai_models" / "training" / "finetune_lora.py"

class TrainStatus(BaseModel):
    status: str
    message: str

def run_training_and_reload(app):
    """Background task to run finetuning and then reload the detector."""
    print("Starting fine-tuning in background...")
    try:
        cmd = [
            "python", str(TRAIN_SCRIPT),
            "--data_dir", str(DATASET_DIR),
            "--output_dir", str(ROOT_DIR / "finetuned_model"),
            "--epochs", "5",  # We increase epochs since dataset is very small
            "--batch_size", "1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return
            
        print("Training completed successfully. Reloading model...")
        
        # We assume the finetune script outputs to ROOT_DIR / "finetuned_model"
        # We should copy the finetuned model weights to ai_models/weights/
        
        finetuned_dir = ROOT_DIR / "finetuned_model"
        weights_dir = ROOT_DIR / "ai_models" / "weights"
        
        if finetuned_dir.exists():
            for f in finetuned_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, weights_dir)
                elif f.is_dir():
                    shutil.copytree(f, weights_dir / f.name, dirs_exist_ok=True)
                
        # Re-initialize the global detector to use the newly trained model weights
        from ai_models.multimodal.multimodal_detector import UnifiedDeepfakeDetector
        app.state.detector = UnifiedDeepfakeDetector()
        print("Model reloaded successfully!")
        
    except Exception as e:
        print(f"Background training task error: {e}")

@router.post("", response_model=TrainStatus)
async def train_model(
    request: Request,
    background_tasks: BackgroundTasks,
    real_files: List[UploadFile] = File(...),
    fake_files: List[UploadFile] = File(...)
):
    """
    Accepts real and fake audio files, saves them to the dataset directory,
    and kicks off a background training job to calibrate the model.
    """
    if REAL_DIR.exists():
        shutil.rmtree(REAL_DIR)
    if FAKE_DIR.exists():
        shutil.rmtree(FAKE_DIR)
        
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)
    
    for idx, rf in enumerate(real_files):
        ext = Path(rf.filename).suffix or ".wav"
        save_path = REAL_DIR / f"real_{idx}{ext}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(rf.file, f)
            
    for idx, ff in enumerate(fake_files):
        ext = Path(ff.filename).suffix or ".wav"
        save_path = FAKE_DIR / f"fake_{idx}{ext}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(ff.file, f)
            
    background_tasks.add_task(run_training_and_reload, request.app)
    
    return TrainStatus(
        status="training_started",
        message="Your voices have been saved. The AI is now calibrating in the background. Please wait 1-2 minutes for the model to update."
    )
