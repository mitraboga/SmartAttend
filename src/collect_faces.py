import argparse

import cv2

from src.config import FACES_DIR
from src.face_detector import FaceDetector
from src.utils import crop_face, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect face samples for a single student.")
    parser.add_argument("--name", required=True, help="Student name / class label")
    parser.add_argument("--count", type=int, default=50, help="Number of face images to capture")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--interval", type=int, default=5, help="Capture every N frames when a face is visible")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = FACES_DIR / args.name.strip()
    ensure_directories(output_dir)

    detector = FaceDetector()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam.")

    frame_index = 0
    saved = 0

    try:
        while saved < args.count:
            ok, frame = capture.read()
            if not ok:
                continue

            boxes = detector.detect(frame)
            frame_index += 1

            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 180, 80), 2)

            if len(boxes) == 1 and frame_index % args.interval == 0:
                face = crop_face(frame, boxes[0], padding=0.25)
                image_path = output_dir / f"{args.name}_{saved + 1:03d}.jpg"
                cv2.imwrite(str(image_path), face)
                saved += 1

            cv2.putText(frame, f"Student: {args.name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Saved: {saved}/{args.count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press q to quit", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            cv2.imshow("SmartAttend - Face Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
