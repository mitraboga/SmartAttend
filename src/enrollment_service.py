from dataclasses import dataclass

from src.config import FACES_DIR, LIVENESS_DIR
from src.database import audit_action, create_student, student_exists
from src.face_detector import FaceDetector
from src.liveness import LivenessDetector
from src.storage import StorageBackend, get_storage_backend
from src.utils import crop_face, safe_label, save_bgr_image, timestamp_slug


@dataclass
class EnrollmentResult:
    success: bool
    message: str
    student: dict | None = None
    liveness_checked: bool = False
    liveness_score: float = 0.0


def enroll_student(
    *,
    first_name: str,
    last_name: str,
    roll_no: str,
    email: str,
    year: str,
    program: str,
    course: str,
    section_id: int | None,
    capture_bgr,
    detector: FaceDetector | None = None,
    liveness_detector: LivenessDetector | None = None,
    created_by_user_id: int | None = None,
    storage_backend: StorageBackend | None = None,
) -> EnrollmentResult:
    detector = detector or FaceDetector()
    liveness_detector = liveness_detector or LivenessDetector()
    storage_backend = storage_backend or get_storage_backend()

    if student_exists(roll_no, email):
        return EnrollmentResult(success=False, message="A student with this roll number or email already exists.")

    boxes = detector.detect(capture_bgr)
    if not boxes:
        return EnrollmentResult(success=False, message="No face detected. Please retake the scan.")
    if len(boxes) > 1:
        return EnrollmentResult(success=False, message="Multiple faces detected. Capture only one student.")

    face = crop_face(capture_bgr, boxes[0], padding=0.25)
    liveness_result = liveness_detector.predict(face)

    if liveness_detector.available and not liveness_result.is_live:
        return EnrollmentResult(
            success=False,
            message="Liveness verification failed. Use a live camera capture, not a photo or screen.",
            liveness_checked=True,
            liveness_score=liveness_result.confidence,
        )

    face_label = safe_label(roll_no)
    face_dir = FACES_DIR / face_label
    capture_slug = timestamp_slug()
    image_path = face_dir / f"{face_label}_{capture_slug}.jpg"
    save_bgr_image(image_path, face)
    stored_object = storage_backend.save_image(f"student-faces/{face_label}/{image_path.name}", face)

    real_liveness_dir = LIVENESS_DIR / "real"
    real_liveness_path = real_liveness_dir / f"enroll_{face_label}_{capture_slug}.jpg"
    save_bgr_image(real_liveness_path, face)

    student = create_student(
        first_name=first_name,
        last_name=last_name,
        roll_no=roll_no,
        email=email,
        year=year,
        program=program,
        course=course,
        face_label=face_label,
        face_dir=str(face_dir),
        primary_face_path=str(image_path),
        section_id=section_id,
        created_by_user_id=created_by_user_id,
        primary_face_storage_uri=stored_object.uri,
    )
    if student:
        audit_action(
            actor_user_id=created_by_user_id,
            actor_role="admin",
            action="student_enrolled",
            entity_type="student",
            entity_id=student["id"],
            payload={
                "roll_no": student["roll_no"],
                "section_id": section_id,
                "storage_backend": stored_object.backend,
            },
        )

    return EnrollmentResult(
        success=True,
        message="Student enrolled successfully.",
        student=student,
        liveness_checked=liveness_detector.available,
        liveness_score=liveness_result.confidence,
    )
