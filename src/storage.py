from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import S3_BUCKET, S3_PREFIX, S3_REGION, STORAGE_BACKEND, STORAGE_ROOT
from src.utils import ensure_directories


@dataclass
class StoredObject:
    backend: str
    uri: str
    local_path: str | None = None


class StorageBackend:
    backend_name = "base"

    def save_image(self, relative_path: str | Path, image_bgr: np.ndarray) -> StoredObject:
        raise NotImplementedError

    def load_image(self, uri: str) -> np.ndarray | None:
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    backend_name = "local"

    def save_image(self, relative_path: str | Path, image_bgr: np.ndarray) -> StoredObject:
        target = STORAGE_ROOT / Path(relative_path)
        ensure_directories(target.parent)
        cv2.imwrite(str(target), image_bgr)
        return StoredObject(backend=self.backend_name, uri=str(target), local_path=str(target))

    def load_image(self, uri: str) -> np.ndarray | None:
        return cv2.imread(uri)


class S3StorageBackend(StorageBackend):
    backend_name = "s3"

    def __init__(self) -> None:
        if not S3_BUCKET:
            raise RuntimeError("S3 storage backend is configured without SMARTATTEND_S3_BUCKET.")
        self.bucket = S3_BUCKET
        self.prefix = S3_PREFIX.strip("/")
        self.region = S3_REGION

    def _client(self):
        import boto3

        kwargs = {"region_name": self.region} if self.region else {}
        return boto3.client("s3", **kwargs)

    def _key(self, relative_path: str | Path) -> str:
        key = Path(relative_path).as_posix().lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def save_image(self, relative_path: str | Path, image_bgr: np.ndarray) -> StoredObject:
        success, encoded = cv2.imencode(".jpg", image_bgr)
        if not success:
            raise RuntimeError("Failed to encode image for object storage upload.")
        client = self._client()
        key = self._key(relative_path)
        client.put_object(Bucket=self.bucket, Key=key, Body=encoded.tobytes(), ContentType="image/jpeg")
        return StoredObject(backend=self.backend_name, uri=f"s3://{self.bucket}/{key}")

    def load_image(self, uri: str) -> np.ndarray | None:
        bucket, key = _parse_s3_uri(uri)
        client = self._client()
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        array = np.frombuffer(body, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    trimmed = uri.removeprefix("s3://")
    bucket, _, key = trimmed.partition("/")
    return bucket, key


def get_storage_backend() -> StorageBackend:
    if STORAGE_BACKEND == "s3":
        return S3StorageBackend()
    return LocalStorageBackend()
