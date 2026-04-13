import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import IMAGE_SIZE


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS)


def resize_and_normalize(face_bgr: np.ndarray, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, image_size)
    return resized.astype("float32") / 255.0


def clamp_box(x: int, y: int, w: int, h: int, shape: tuple[int, int, int], padding: float = 0.2) -> tuple[int, int, int, int]:
    frame_h, frame_w = shape[:2]
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_w, x + w + pad_w)
    y2 = min(frame_h, y + h + pad_h)
    return x1, y1, x2, y2


def crop_face(frame: np.ndarray, box: tuple[int, int, int, int], padding: float = 0.2) -> np.ndarray:
    x, y, w, h = box
    x1, y1, x2, y2 = clamp_box(x, y, w, h, frame.shape, padding=padding)
    return frame[y1:y2, x1:x2]


def timestamp_strings() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


def iso_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "student"


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def save_bgr_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
