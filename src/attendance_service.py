from dataclasses import dataclass

from src.database import (
    audit_action,
    create_exception_from_attempt,
    get_class_session,
    get_student_by_label,
    get_student_by_roll_no,
    is_attendance_rate_limited,
    is_student_in_session,
    log_attendance_attempt,
    upsert_session_attendance,
)
from src.face_detector import FaceDetector
from src.liveness import LivenessDetector
from src.recognizer import FaceRecognizer
from src.utils import crop_face, iso_timestamp


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
    exception_id: int | None = None


def _window_is_open(session: dict) -> bool:
    now = iso_timestamp()
    return session["attendance_open_at"] <= now <= session["attendance_close_at"]


def verify_attendance_attempt(
    *,
    session_id: int,
    claimed_roll_no: str,
    capture_bgr,
    actor_user_id: int | None = None,
    actor_role: str | None = None,
    detector: FaceDetector | None = None,
    recognizer: FaceRecognizer | None = None,
    liveness_detector: LivenessDetector | None = None,
) -> AttendanceDecision:
    detector = detector or FaceDetector()
    recognizer = recognizer or FaceRecognizer()
    liveness_detector = liveness_detector or LivenessDetector()

    session = get_class_session(session_id)
    if session is None:
        return AttendanceDecision(success=False, status="Invalid Session", message="The selected class session does not exist.", attempt_outcome="invalid_session")
    if session["status"] != "open":
        return AttendanceDecision(success=False, status="Session Closed", message="The selected class session is not open for attendance.", attempt_outcome="session_closed")
    if not _window_is_open(session):
        return AttendanceDecision(success=False, status="Outside Window", message="Attendance is outside the configured session window.", attempt_outcome="outside_window")

    student = get_student_by_roll_no(claimed_roll_no)
    if student is None:
        attempt = log_attendance_attempt(
            student_id=None,
            session_id=session_id,
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="unknown_roll_no",
            confidence=0.0,
            liveness_score=0.0,
            note="roll number not found",
            raw_label="Unknown",
            predicted_student_id=None,
            needs_review=False,
        )
        return AttendanceDecision(success=False, status="Unknown", message="Roll number not found in the enrolled student list.", attempt_outcome=attempt["attempt_outcome"])

    if not is_student_in_session(student["id"], session_id):
        attempt = log_attendance_attempt(
            student_id=student["id"],
            session_id=session_id,
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="not_registered_for_session",
            confidence=0.0,
            liveness_score=0.0,
            note="student is not rostered for this section/session",
            raw_label="Unavailable",
            predicted_student_id=None,
            needs_review=False,
        )
        return AttendanceDecision(success=False, status="Not Registered", message="This student is not rostered for the selected class session.", student=student, attempt_outcome=attempt["attempt_outcome"])

    if is_attendance_rate_limited(session_id, claimed_roll_no):
        return AttendanceDecision(success=False, status="Rate Limited", message="Too many failed attempts for this roll number in the current session. Ask faculty to review the case.", student=student, attempt_outcome="rate_limited")

    if not liveness_detector.available:
        attempt = log_attendance_attempt(
            student_id=student["id"],
            session_id=session_id,
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="setup_required",
            confidence=0.0,
            liveness_score=0.0,
            note="liveness model missing",
            raw_label="Unavailable",
            predicted_student_id=None,
            needs_review=False,
        )
        return AttendanceDecision(
            success=False,
            status="Setup Required",
            message="Liveness model is not trained yet. Complete the liveness setup before running production attendance.",
            student=student,
            action="setup_required",
            attempt_outcome=attempt["attempt_outcome"],
        )

    boxes = detector.detect(capture_bgr)
    if not boxes:
        attempt = log_attendance_attempt(
            student_id=student["id"],
            session_id=session_id,
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="no_face_detected",
            confidence=0.0,
            liveness_score=0.0,
            note="no face detected",
            raw_label="Unknown",
            predicted_student_id=None,
            needs_review=False,
        )
        return AttendanceDecision(success=False, status="Retry", message="No face detected. Retake the attendance scan.", student=student, attempt_outcome=attempt["attempt_outcome"])
    if len(boxes) > 1:
        attempt = log_attendance_attempt(
            student_id=student["id"],
            session_id=session_id,
            claimed_roll_no=claimed_roll_no,
            official_status=None,
            attempt_outcome="multiple_faces_detected",
            confidence=0.0,
            liveness_score=0.0,
            note="multiple faces detected",
            raw_label="Unknown",
            predicted_student_id=None,
            needs_review=False,
        )
        return AttendanceDecision(success=False, status="Retry", message="Multiple faces detected. Only one student should be in frame.", student=student, attempt_outcome=attempt["attempt_outcome"])

    face = crop_face(capture_bgr, boxes[0], padding=0.25)
    recognition = recognizer.predict(face)
    liveness = liveness_detector.predict(face)
    predicted_student = get_student_by_label(recognition.label) if recognition.label and recognition.label != "Unknown" else None

    success = recognition.label == student["face_label"] and liveness.is_live
    attempt_outcome = "verified_present" if success else "verification_failed"
    note = "verified" if success else "verification failed"
    needs_review = False

    if not liveness.is_live:
        attempt_outcome = "spoof_attempt"
        note = "spoof attempt detected"
        needs_review = True
    elif recognition.label == "Unknown":
        attempt_outcome = "unrecognized_face"
        note = "face not recognized"
        needs_review = True
    elif recognition.label != student["face_label"]:
        attempt_outcome = "identity_mismatch"
        note = "face does not match claimed student"
        needs_review = True

    if success:
        action, _record = upsert_session_attendance(
            session_id=session_id,
            student_id=student["id"],
            status="Present",
            confidence=recognition.confidence,
            liveness_score=liveness.confidence,
            note=note,
            raw_label=recognition.label,
            claimed_roll_no=claimed_roll_no,
            source="live_verification",
            verified_by_user_id=actor_user_id,
        )
        log_attendance_attempt(
            student_id=student["id"],
            session_id=session_id,
            claimed_roll_no=claimed_roll_no,
            official_status="Present",
            attempt_outcome=attempt_outcome,
            confidence=recognition.confidence,
            liveness_score=liveness.confidence,
            note=note,
            raw_label=recognition.label,
            predicted_student_id=predicted_student["id"] if predicted_student else None,
            needs_review=False,
        )
        audit_action(
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            action="attendance_verified",
            entity_type="attendance_record",
            entity_id=f"{session_id}:{student['id']}",
            payload={"action": action, "roll_no": claimed_roll_no, "session_id": session_id},
        )
        return AttendanceDecision(
            success=True,
            status="Present",
            message="Attendance accepted and recorded for the open class session.",
            student=student,
            predicted_student=predicted_student,
            confidence=recognition.confidence,
            liveness_score=liveness.confidence,
            action=action,
            attempt_outcome=attempt_outcome,
        )

    attempt = log_attendance_attempt(
        student_id=student["id"],
        session_id=session_id,
        claimed_roll_no=claimed_roll_no,
        official_status=None,
        attempt_outcome=attempt_outcome,
        confidence=recognition.confidence,
        liveness_score=liveness.confidence,
        note=note,
        raw_label=recognition.label,
        predicted_student_id=predicted_student["id"] if predicted_student else None,
        needs_review=needs_review,
    )
    exception_id = None
    if needs_review:
        exception = create_exception_from_attempt(
            attempt_id=attempt["id"],
            session_id=session_id,
            student_id=student["id"],
            reason=note,
        )
        exception_id = exception.get("id")

    audit_action(
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        action="attendance_verification_failed",
        entity_type="attendance_attempt",
        entity_id=attempt.get("id"),
        payload={"session_id": session_id, "roll_no": claimed_roll_no, "attempt_outcome": attempt_outcome},
    )
    return AttendanceDecision(
        success=False,
        status="Rejected",
        message="Attendance was not accepted. The attempt has been logged and queued for faculty review." if needs_review else "Attendance was not accepted.",
        student=student,
        predicted_student=predicted_student,
        confidence=recognition.confidence,
        liveness_score=liveness.confidence,
        action="queued_for_review" if needs_review else "rejected",
        attempt_outcome=attempt_outcome,
        exception_id=exception_id,
    )
