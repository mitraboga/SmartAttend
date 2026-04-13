from dataclasses import dataclass
from pathlib import Path

from src.config import LIVENESS_DIR
from src.face_detector import FaceDetector
from src.utils import crop_face, list_image_files, safe_label, save_bgr_image, timestamp_slug


@dataclass
class LivenessSampleResult:
    success: bool
    message: str
    saved_paths: list[str]


def liveness_counts() -> dict[str, int]:
    real_dir = LIVENESS_DIR / "real"
    fake_dir = LIVENESS_DIR / "fake"
    return {
        "real": len(list_image_files(real_dir)),
        "fake": len(list_image_files(fake_dir)),
    }


def save_liveness_sample(image_bgr, label: str, detector: FaceDetector | None = None, source_prefix: str = "sample") -> LivenessSampleResult:
    detector = detector or FaceDetector()
    label = safe_label(label.lower())
    if label not in {"real", "fake"}:
        return LivenessSampleResult(False, "Invalid liveness label.", [])

    boxes = detector.detect(image_bgr)
    if not boxes:
        return LivenessSampleResult(False, "No face detected in the image.", [])
    if len(boxes) > 1:
        return LivenessSampleResult(False, "Multiple faces detected. Capture exactly one face.", [])

    face = crop_face(image_bgr, boxes[0], padding=0.25)
    target_dir = LIVENESS_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / f"{source_prefix}_{timestamp_slug()}.jpg"
    save_bgr_image(image_path, face)
    return LivenessSampleResult(True, f"Saved {label} liveness sample.", [str(image_path)])


def dataset_ready_for_training(min_per_class: int = 5) -> tuple[bool, dict[str, int]]:
    counts = liveness_counts()
    ready = counts["real"] >= min_per_class and counts["fake"] >= min_per_class
    return ready, counts
