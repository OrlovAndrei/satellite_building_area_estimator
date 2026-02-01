import torch
from pathlib import Path
from .models import load_models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_DIR = Path(__file__).resolve().parent.parent

SEG_MODEL_PATH  = PROJECT_DIR / "models" / "segmentation.pth"
SCALE_MODEL_PATH = PROJECT_DIR / "models" / "scale_estimator.pth"

SEG, SCALE = load_models(DEVICE, SEG_MODEL_PATH, SCALE_MODEL_PATH)