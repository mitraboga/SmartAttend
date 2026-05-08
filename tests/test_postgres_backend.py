import os
import time
import unittest

from src import database


@unittest.skipUnless(os.getenv("SMARTATTEND_DATABASE_URL"), "requires SMARTATTEND_DATABASE_URL")
class PostgresBackendSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_database_url = database.DATABASE_URL
        database.DATABASE_URL = os.environ["SMARTATTEND_DATABASE_URL"]
        database.init_database()

    def tearDown(self) -> None:
        database.DATABASE_URL = self.original_database_url

    def test_postgres_bootstrap_and_core_entities(self) -> None:
        suffix = str(time.time_ns())
        department = database.create_department(f"Computer Science {suffix}", f"C{suffix[-4:]}")
        program = database.create_program(
            department_id=department["id"],
            name=f"B. Tech CSE {suffix}",
            code=f"BTECH-{suffix[-4:]}",
        )
        section = database.create_section(
            program_id=program["id"],
            name=f"A-{suffix[-3:]}",
            year_label="3rd Year",
            semester_label="Semester 6",
        )
        faculty = database.create_user(
            username=f"faculty_pg_{suffix[-8:]}",
            full_name="Faculty Postgres",
            email=f"faculty.pg.{suffix}@example.com",
            password="Faculty!1234",
            role="faculty",
        )
        database.create_faculty_profile(user_id=faculty["id"], department_id=department["id"], title="Assistant Professor")
        course = database.create_course(
            department_id=department["id"],
            course_code=f"CSE{suffix[-3:]}",
            title="Distributed Systems",
            credit_hours=3,
        )
        offering = database.create_course_offering(
            course_id=course["id"],
            section_id=section["id"],
            faculty_user_id=faculty["id"],
            term_name="Monsoon",
            academic_year="2026-2027",
        )
        session = database.create_class_session(
            offering_id=offering["id"],
            session_title="Postgres Smoke Session",
            session_date="2026-05-02",
            start_time="09:00:00",
            end_time="10:00:00",
            attendance_open_at="2026-05-02T08:55:00",
            attendance_close_at="2026-05-02T09:20:00",
            location="Cloud Lab",
            notes="postgres smoke",
            created_by_user_id=faculty["id"],
        )

        self.assertEqual(database.database_backend(), "postgres")
        self.assertIsNotNone(database.current_schema_version())
        self.assertEqual(session["offering_id"], offering["id"])


if __name__ == "__main__":
    unittest.main()
