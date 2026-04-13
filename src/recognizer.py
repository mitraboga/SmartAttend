from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import FACE_LABELS_PATH, FACE_MATCHER_THRESHOLD, FACE_MODEL_PATH, IMAGE_SIZE, RECOGNITION_CONFIDENCE_THRESHOLD, UNKNOWN_LABEL
from src.database import list_face_samples
from src.utils import list_image_files, load_json, resize_and_normalize


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    reason: str


class FaceRecognizer:
    def __init__(self, model_path: Path = FACE_MODEL_PATH, labels_path: Path = FACE_LABELS_PATH) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_names: list[str] = []
        self.available = False
        self.fallback_prototypes: dict[str, np.ndarray] = {}
        self._load_if_possible()
        self._load_fallback_prototypes()

    def _load_if_possible(self) -> None:
        if not self.model_path.exists() or not self.labels_path.exists():
            return

        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = load_json(self.labels_path, default=[])
        self.available = bool(self.class_names)

    def _load_fallback_prototypes(self) -> None:
        from src.config import FACES_DIR

        self.fallback_prototypes.clear()
        face_samples = list_face_samples()
        grouped_paths: dict[str, list[str]] = {}

        for sample in face_samples:
            grouped_paths.setdefault(sample["face_label"], []).append(sample["image_path"])

        if grouped_paths:
            items = sorted(grouped_paths.items())
        else:
            if not FACES_DIR.exists():
                return
            items = [(label_dir.name, [str(path) for path in list_image_files(label_dir)]) for label_dir in sorted(path for path in FACES_DIR.iterdir() if path.is_dir())]

        for label_name, image_paths in items:
            vectors: list[np.ndarray] = []
            for image_path in image_paths:
                import cv2

                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                normalized = resize_and_normalize(image, IMAGE_SIZE).flatten()
                vectors.append(normalized)

            if vectors:
                prototype = np.mean(np.stack(vectors, axis=0), axis=0)
                norm = np.linalg.norm(prototype)
                if norm > 0:
                    self.fallback_prototypes[label_name] = prototype / norm

    def predict(self, face_bgr) -> RecognitionResult:
        if self.available and self.model is not None:
            image = resize_and_normalize(face_bgr, IMAGE_SIZE)
            batch = np.expand_dims(image, axis=0)
            probabilities = self.model.predict(batch, verbose=0)[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            label = self.class_names[best_index] if best_index < len(self.class_names) else UNKNOWN_LABEL

            if confidence < RECOGNITION_CONFIDENCE_THRESHOLD:
                label = UNKNOWN_LABEL

            return RecognitionResult(label=label, confidence=confidence, reason="cnn")

        if not self.fallback_prototypes:
            return RecognitionResult(label=UNKNOWN_LABEL, confidence=0.0, reason="model_unavailable")

        query = resize_and_normalize(face_bgr, IMAGE_SIZE).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return RecognitionResult(label=UNKNOWN_LABEL, confidence=0.0, reason="invalid_face")
        query = query / query_norm

        best_label = UNKNOWN_LABEL
        best_score = -1.0
        for label, prototype in self.fallback_prototypes.items():
            score = float(np.dot(query, prototype))
            if score > best_score:
                best_label = label
                best_score = score

        if best_score < FACE_MATCHER_THRESHOLD:
            best_label = UNKNOWN_LABEL

        return RecognitionResult(label=best_label, confidence=max(best_score, 0.0), reason="fallback")
