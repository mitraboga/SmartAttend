import cv2

from src.config import FACE_DETECTOR_BACKEND


class FaceDetector:
    def __init__(
        self,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (64, 64),
        preferred_backend: str = FACE_DETECTOR_BACKEND,
    ) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.backend = "haar"
        self.detector = None
        self.mtcnn = None

        if preferred_backend in {"auto", "mtcnn"}:
            try:
                from mtcnn import MTCNN

                self.mtcnn = MTCNN()
                self.backend = "mtcnn"
            except Exception:
                self.mtcnn = None

        if self.backend == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        if self.backend == "mtcnn" and self.mtcnn is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mtcnn.detect_faces(frame_rgb)
            boxes: list[tuple[int, int, int, int]] = []
            for result in results:
                x, y, w, h = result.get("box", (0, 0, 0, 0))
                if w <= 0 or h <= 0:
                    continue
                boxes.append((max(0, int(x)), max(0, int(y)), int(w), int(h)))
            return boxes

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [tuple(int(value) for value in box) for box in boxes]
