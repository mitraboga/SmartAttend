from dataclasses import dataclass

from src.database import get_student_by_label, get_student_by_roll_no, log_attendance_attempt, upsert_attendance
from src.face_detector import FaceDetector
from src.liveness import LivenessDetector
from src.recognizer import FaceRecognizer
from src.utils import crop_face


@dataclass
class AttendanceDecision:
    success: bool
    status: str
    message: str
    student: dict | None = None
    predicted_student: dict | None = None
    confidence: float = 0.0
    liveness_score: float = 0.0
    action: str = ""
    attempt_outcome: str = ""


def verify_attendance_attempt(
    *,
    claimed_roll_no: str,
    capture_bgr,
    detector: FaceDetector | None = None,
    recognizer: FaceRecognizer | None = None,
    liveness_detector: LivenessDetector | None = None,
) -> AttendanceDecision:
    detector = detector or FaceDetector()
    recognizer = recognizer or FaceRecognizer()
    liveness_detector = liveness_detector or LivenessDetector()

    student = get_student_by_roll_no(claimed_roll_no)
    if student is None:
        log_attendance_attempt(
            student_id=None,
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="unknown_roll_no",
            confidence=0.0,
            liveness_score=0.0,
            note="roll number not found",
            raw_label="Unknown",
            predicted_student_id=None,
        )
        return AttendanceDecision(success=False, status="Unknown", message="Roll number not found in the enrolled student list.", attempt_outcome="unknown_roll_no")

    if not liveness_detector.available:
        log_attendance_attempt(
            student_id=student["id"],
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="setup_required",
            confidence=0.0,
            liveness_score=0.0,
            note="liveness model missing",
            raw_label="Unavailable",
            predicted_student_id=None,
        )
        return AttendanceDecision(
            success=False,
            status="Setup Required",
            message="Liveness model is not trained yet. Open the Liveness Setup page, collect real and fake samples, train the model, then try attendance again.",
            student=student,
            action="setup_required",
            attempt_outcome="setup_required",
        )

    boxes = detector.detect(capture_bgr)
    if not boxes:
        log_attendance_attempt(
            student_id=student["id"],
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="no_face_detected",
            confidence=0.0,
            liveness_score=0.0,
            note="no face detected",
            raw_label="Unknown",
            predicted_student_id=None,
        )
        return AttendanceDecision(success=False, status="Retry", message="No face detected. Please retake the attendance scan.", student=student, attempt_outcome="no_face_detected")
    if len(boxes) > 1:
        log_attendance_attempt(
            student_id=student["id"],
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="multiple_faces_detected",
            confidence=0.0,
            liveness_score=0.0,
            note="multiple faces detected",
            raw_label="Unknown",
            predicted_student_id=None,
        )
        return AttendanceDecision(success=False, status="Retry", message="Multiple faces detected. Only one student should be in frame.", student=student, attempt_outcome="multiple_faces_detected")

    face = crop_face(capture_bgr, boxes[0], padding=0.25)
    recognition = recognizer.predict(face)
    liveness = liveness_detector.predict(face)
    predicted_student = get_student_by_label(recognition.label) if recognition.label and recognition.label != "Unknown" else None

    status = "Absent"
    note = "identity mismatch"
    success = False

    if recognition.label == student["face_label"] and (not liveness_detector.available or liveness.is_live):
        status = "Present"
        note = "verified"
        success = True
    elif recognition.label == "Unknown":
        note = "face not recognized"
    elif recognition.label != student["face_label"]:
        note = "face does not match claimed student"
    elif liveness_detector.available and not liveness.is_live:
        note = "spoof attempt detected"

    attempt_outcome = "verified_present" if success else "verification_failed"
    if note == "spoof attempt detected":
        attempt_outcome = "spoof_attempt"
    elif note == "face not recognized":
        attempt_outcome = "unrecognized_face"
    elif note == "face does not match claimed student":
        attempt_outcome = "identity_mismatch"

    action, _record = upsert_attendance(
        student_id=student["id"],
        status=status,
        confidence=recognition.confidence,
        liveness_score=liveness.confidence,
        note=note,
        raw_label=recognition.label,
        claimed_roll_no=claimed_roll_no,
    )

    log_attendance_attempt(
        student_id=student["id"],
        claimed_roll_no=claimed_roll_no,
        official_status="Present" if success or action in {"duplicate", "preserved_present"} else status,
        attempt_outcome=attempt_outcome,
        confidence=recognition.confidence,
        liveness_score=liveness.confidence,
        note=note,
        raw_label=recognition.label,
        predicted_student_id=predicted_student["id"] if predicted_student else None,
    )

    if action == "duplicate" and status == "Present":
        return AttendanceDecision(
            success=True,
            status="Present",
            message="Attendance already marked for today.",
            student=student,
            predicted_student=predicted_student,
            confidence=recognition.confidence,
            liveness_score=liveness.confidence,
            action=action,
            attempt_outcome=attempt_outcome,
        )

    if action == "preserved_present":
        return AttendanceDecision(
            success=False,
            status="Present",
            message="This scan failed verification, but the student was already marked present earlier today, so attendance was not downgraded.",
            student=student,
            predicted_student=predicted_student,
            confidence=recognition.confidence,
            liveness_score=liveness.confidence,
            action=action,
            attempt_outcome=attempt_outcome,
        )

    message = "Attendance marked present." if success else "Attendance marked absent."
    if not liveness_detector.available:
        message += " Liveness model is not trained yet, so this decision used face verification only."

    return AttendanceDecision(
        success=success,
        status=status,
        message=message,
        student=student,
        predicted_student=predicted_student,
        confidence=recognition.confidence,
        liveness_score=liveness.confidence,
        action=action,
        attempt_outcome=attempt_outcome,
    )
