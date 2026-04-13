import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


DATA_DIR = _env_path("SMARTATTEND_DATA_DIR", BASE_DIR / "data")
FACES_DIR = DATA_DIR / "faces"
LIVENESS_DIR = DATA_DIR / "liveness"
MODELS_DIR = _env_path("SMARTATTEND_MODELS_DIR", BASE_DIR / "models")
ARTIFACTS_DIR = _env_path("SMARTATTEND_ARTIFACTS_DIR", BASE_DIR / "artifacts")
DATABASE_PATH = _env_path("SMARTATTEND_DATABASE_PATH", DATA_DIR / "smartattend.db")

FACE_MODEL_PATH = MODELS_DIR / "face_recognition_model.keras"
FACE_LABELS_PATH = MODELS_DIR / "face_labels.json"
LIVENESS_MODEL_PATH = MODELS_DIR / "liveness_model.keras"
LIVENESS_METADATA_PATH = MODELS_DIR / "liveness_metadata.json"

IMAGE_SIZE = (128, 128)
RECOGNITION_CONFIDENCE_THRESHOLD = _env_float("SMARTATTEND_RECOGNITION_THRESHOLD", 0.75)
LIVENESS_CONFIDENCE_THRESHOLD = _env_float("SMARTATTEND_LIVENESS_THRESHOLD", 0.50)
UNKNOWN_LABEL = "Unknown"
FACE_DETECTOR_BACKEND = os.getenv("SMARTATTEND_FACE_DETECTOR_BACKEND", "auto")
FACE_MATCHER_THRESHOLD = _env_float("SMARTATTEND_FACE_MATCHER_THRESHOLD", 0.90)
APP_TITLE = os.getenv("SMARTATTEND_APP_TITLE", "SmartAttend")
