import argparse

import cv2

from src.attendance_service import verify_attendance_attempt
from src.database import get_student_by_roll_no, init_database
from src.face_detector import FaceDetector
from src.liveness import LivenessDetector
from src.recognizer import FaceRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time smart attendance from webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--roll-no", required=True, help="Claimed roll number to verify during attendance")
    return parser.parse_args()


def format_label(name: str, confidence: float, live_score: float, note: str, status: str) -> list[str]:
    return [
        f"Name: {name} ({confidence:.2f})",
        f"Liveness Score: {live_score:.2f}",
        f"Status: {status} - {note}",
    ]


def run_attendance_session(camera_index: int = 0, claimed_roll_no: str = "") -> None:
    init_database()
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    liveness = LivenessDetector()

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            boxes = detector.detect(frame)
            claimed_student = get_student_by_roll_no(claimed_roll_no)
            for box in boxes[:1]:
                x, y, w, h = box
                decision = verify_attendance_attempt(
                    claimed_roll_no=claimed_roll_no,
                    capture_bgr=frame,
                    detector=detector,
                    recognizer=recognizer,
                    liveness_detector=liveness,
                )

                color = (0, 200, 0) if decision.status == "Present" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                lines = format_label(
                    f"{claimed_student['first_name']} {claimed_student['last_name']}" if claimed_student else claimed_roll_no,
                    decision.confidence,
                    decision.liveness_score,
                    decision.message,
                    decision.status,
                )

                for idx, line in enumerate(lines):
                    cv2.putText(
                        frame,
                        line,
                        (x, max(20, y - 10 - idx * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            cv2.putText(frame, f"Claimed Roll No: {claimed_roll_no}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.putText(frame, "Press q to quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.imshow("SmartAttend - Live Attendance", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_attendance_session(camera_index=args.camera, claimed_roll_no=args.roll_no)


if __name__ == "__main__":
    main()
