import os
import unittest
from pathlib import Path

from src import database


class ProductionFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch_root = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.db_path = scratch_root / "smartattend_test.db"
        if self.db_path.exists():
            self.db_path.unlink()
        self.original_database_url = database.DATABASE_URL
        self.original_database_url_env = os.environ.get("SMARTATTEND_DATABASE_URL")
        database.DATABASE_PATH = self.db_path
        database.DATABASE_URL = ""
        os.environ.pop("SMARTATTEND_DATABASE_URL", None)
        database.init_database()

    def tearDown(self) -> None:
        database.DATABASE_URL = self.original_database_url
        if self.original_database_url_env is None:
            os.environ.pop("SMARTATTEND_DATABASE_URL", None)
        else:
            os.environ["SMARTATTEND_DATABASE_URL"] = self.original_database_url_env
        if self.db_path.exists():
            self.db_path.unlink()

    def test_authentication_flow_supports_faculty_users(self) -> None:
        faculty = database.create_user(
            username="faculty_test",
            full_name="Faculty Test",
            email="faculty@example.com",
            password="StrongPass!123",
            role="faculty",
        )
        self.assertEqual(faculty["role"], "faculty")

        authenticated, message = database.authenticate_user("faculty_test", "StrongPass!123")
        self.assertIsNotNone(authenticated)
        self.assertEqual(message, "Login successful.")
        self.assertEqual(authenticated["username"], "faculty_test")

    def test_session_attendance_lifecycle_supports_finalize_and_exception_resolution(self) -> None:
        department = database.create_department("Computer Science", "CSE")
        program = database.create_program(department_id=department["id"], name="B. Tech CSE", code="BTECH-CSE")
        section = database.create_section(program_id=program["id"], name="A", year_label="3rd Year", semester_label="Semester 6")
        faculty = database.create_user(
            username="faculty_scope",
            full_name="Faculty Scope",
            email="faculty.scope@example.com",
            password="Faculty!1234",
            role="faculty",
        )
        database.create_faculty_profile(user_id=faculty["id"], department_id=department["id"], title="Assistant Professor")
        course = database.create_course(department_id=department["id"], course_code="CSE301", title="Operating Systems", credit_hours=3)
        offering = database.create_course_offering(
            course_id=course["id"],
            section_id=section["id"],
            faculty_user_id=faculty["id"],
            term_name="Monsoon",
            academic_year="2026-2027",
        )
        session = database.create_class_session(
            offering_id=offering["id"],
            session_title="Week 1 Lecture",
            session_date="2026-05-02",
            start_time="09:00:00",
            end_time="10:00:00",
            attendance_open_at="2026-05-02T08:55:00",
            attendance_close_at="2026-05-02T09:20:00",
            location="C-204",
            notes="Intro session",
            created_by_user_id=faculty["id"],
        )
        database.update_class_session_status(session["id"], "open", faculty["id"])

        student_one = database.create_student(
            first_name="Mitra",
            last_name="Boga",
            roll_no="2023005779",
            email="mitra@example.com",
            year="3rd Year",
            program="B. Tech",
            course="CSE",
            face_label="2023005779",
            face_dir="data/faces/2023005779",
            primary_face_path="data/faces/2023005779/sample.jpg",
            section_id=section["id"],
            created_by_user_id=faculty["id"],
            primary_face_storage_uri="local://sample",
        )
        student_two = database.create_student(
            first_name="Aisha",
            last_name="Khan",
            roll_no="2023005780",
            email="aisha@example.com",
            year="3rd Year",
            program="B. Tech",
            course="CSE",
            face_label="2023005780",
            face_dir="data/faces/2023005780",
            primary_face_path="data/faces/2023005780/sample.jpg",
            section_id=section["id"],
            created_by_user_id=faculty["id"],
            primary_face_storage_uri="local://sample2",
        )

        attempt = database.log_attendance_attempt(
            student_id=student_one["id"],
            session_id=session["id"],
            claimed_roll_no=student_one["roll_no"],
            official_status=None,
            attempt_outcome="identity_mismatch",
            confidence=0.42,
            liveness_score=0.91,
            note="face does not match claimed student",
            raw_label="Unknown",
            predicted_student_id=None,
            needs_review=True,
        )
        exception = database.create_exception_from_attempt(
            attempt_id=attempt["id"],
            session_id=session["id"],
            student_id=student_one["id"],
            reason="identity mismatch",
        )
        resolved = database.resolve_exception(
            exception_id=exception["id"],
            reviewer_user_id=faculty["id"],
            resolution="approved_present",
            resolution_note="Manual ID verification completed",
            resolved_attendance_status="Present",
        )
        self.assertEqual(resolved["status"], "resolved")

        created_absences = database.finalize_session_absences(session["id"], faculty["id"])
        self.assertEqual(created_absences, 1)

        records = database.list_attendance_records(user=faculty, session_id=session["id"])
        statuses_by_roll = {record["roll_no"]: record["status"] for record in records}
        self.assertEqual(statuses_by_roll[student_one["roll_no"]], "Present")
        self.assertEqual(statuses_by_roll[student_two["roll_no"]], "Absent")


if __name__ == "__main__":
    unittest.main()
