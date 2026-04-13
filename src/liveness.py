from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import IMAGE_SIZE, LIVENESS_CONFIDENCE_THRESHOLD, LIVENESS_METADATA_PATH, LIVENESS_MODEL_PATH
from src.utils import load_json, resize_and_normalize


@dataclass
class LivenessResult:
    is_live: bool
    confidence: float
    reason: str


class LivenessDetector:
    def __init__(self, model_path: Path = LIVENESS_MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = None
        self.available = False
        self.threshold = LIVENESS_CONFIDENCE_THRESHOLD
        self._load_if_possible()

    def _load_if_possible(self) -> None:
        if not self.model_path.exists():
            return

        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.model_path)
        self.available = True
        metadata = load_json(LIVENESS_METADATA_PATH, default={})
        self.threshold = float(metadata.get("threshold", LIVENESS_CONFIDENCE_THRESHOLD))

    def predict(self, face_bgr) -> LivenessResult:
        if not self.available or self.model is None:
            return LivenessResult(is_live=False, confidence=0.0, reason="model_unavailable")

        image = resize_and_normalize(face_bgr, IMAGE_SIZE)
        batch = np.expand_dims(image, axis=0)
        live_score = float(self.model.predict(batch, verbose=0)[0][0])
        return LivenessResult(
            is_live=live_score >= self.threshold,
            confidence=live_score,
            reason="ok",
        )
