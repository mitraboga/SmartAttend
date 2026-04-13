from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
LIVENESS_DIR = DATA_DIR / "liveness"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATABASE_PATH = DATA_DIR / "smartattend.db"

FACE_MODEL_PATH = MODELS_DIR / "face_recognition_model.keras"
FACE_LABELS_PATH = MODELS_DIR / "face_labels.json"
LIVENESS_MODEL_PATH = MODELS_DIR / "liveness_model.keras"
LIVENESS_METADATA_PATH = MODELS_DIR / "liveness_metadata.json"

IMAGE_SIZE = (128, 128)
RECOGNITION_CONFIDENCE_THRESHOLD = 0.75
LIVENESS_CONFIDENCE_THRESHOLD = 0.50
UNKNOWN_LABEL = "Unknown"
FACE_DETECTOR_BACKEND = "auto"
FACE_MATCHER_THRESHOLD = 0.90
APP_TITLE = "SmartAttend"
