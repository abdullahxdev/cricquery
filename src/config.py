"""Central config — single source of truth for model names, paths, class labels."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
EXAMPLES_DIR = ROOT / "examples"

VIT_WEIGHTS = MODELS_DIR / "vit_cricket.pt"

VIT_BACKBONE = "google/vit-base-patch16-224"
CLIP_MODEL = "openai/clip-vit-base-patch32"
BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
YOLO_WEIGHTS = "yolov8n.pt"

SHOT_CLASSES = ["drive", "pullshot", "legglance-flick", "sweep", "defensive"]
NUM_CLASSES = len(SHOT_CLASSES)

ATTACKING_SHOTS = {"drive", "pullshot", "legglance-flick", "sweep"}
DEFENSIVE_SHOTS = {"defensive"}

IMAGE_SIZE = 224
DEVICE = "cuda"  # overridden at runtime if cuda not available
