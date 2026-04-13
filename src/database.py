import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.config import DATABASE_PATH
from src.utils import ensure_directories, iso_timestamp, today_string


DEFAULT_ADMIN_USERNAME = os.getenv("SMARTATTEND_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("SMARTATTEND_ADMIN_PASSWORD", "admin123")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@contextmanager
def get_connection(database_path: Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    ensure_directories(database_path.parent)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                roll_no TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                year TEXT NOT NULL,
                program TEXT NOT NULL,
                course TEXT NOT NULL,
                face_label TEXT NOT NULL UNIQUE,
                face_dir TEXT NOT NULL,
                primary_face_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'enrollment',
                created_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                liveness_score REAL,
                note TEXT,
                raw_label TEXT,
                claimed_roll_no TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(student_id, attendance_date),
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                claimed_roll_no TEXT,
                attempt_date TEXT NOT NULL,
                attempt_time TEXT NOT NULL,
                official_status TEXT,
                attempt_outcome TEXT NOT NULL,
                confidence REAL,
                liveness_score REAL,
                note TEXT,
                raw_label TEXT,
                predicted_student_id INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE SET NULL,
                FOREIGN KEY(predicted_student_id) REFERENCES students(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_key TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        seed_default_admin(connection)


def seed_default_admin(connection: sqlite3.Connection) -> None:
    created_at = iso_timestamp()
    connection.execute(
        """
        INSERT INTO admin_users (username, password_hash, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            password_hash = excluded.password_hash
        """,
        (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD), created_at),
    )


def verify_admin(username: str, password: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT password_hash FROM admin_users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return False
    return hash_password(password) == row["password_hash"]


def student_exists(roll_no: str, email: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM students
            WHERE roll_no = ? OR email = ?
            """,
            (roll_no.strip(), email.strip().lower()),
        ).fetchone()
    return row is not None


def create_student(
    *,
    first_name: str,
    last_name: str,
    roll_no: str,
    email: str,
    year: str,
    program: str,
    course: str,
    face_label: str,
    face_dir: str,
    primary_face_path: str,
) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO students (
                first_name,
                last_name,
                roll_no,
                email,
                year,
                program,
                course,
                face_label,
                face_dir,
                primary_face_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_name.strip(),
                last_name.strip(),
                roll_no.strip(),
                email.strip().lower(),
                year.strip(),
                program.strip(),
                course.strip(),
                face_label,
                face_dir,
                primary_face_path,
                created_at,
            ),
        )
        student_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO face_samples (student_id, image_path, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, primary_face_path, "enrollment", created_at),
        )
        row = connection.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    return dict(row) if row else {}


def add_face_sample(student_id: int, image_path: str, source: str = "manual") -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO face_samples (student_id, image_path, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, image_path, source, iso_timestamp()),
        )


def get_student_by_roll_no(roll_no: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM students WHERE roll_no = ?",
            (roll_no.strip(),),
        ).fetchone()
    return dict(row) if row else None


def get_student_by_label(face_label: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM students WHERE face_label = ?",
            (face_label,),
        ).fetchone()
    return dict(row) if row else None


def list_students() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                s.*,
                COUNT(a.id) AS attendance_events,
                COALESCE(SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END), 0) AS present_count,
                COALESCE(SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END), 0) AS absent_count
            FROM students s
            LEFT JOIN attendance a ON a.student_id = s.id
            GROUP BY s.id
            ORDER BY s.first_name, s.last_name
            """
        ).fetchall()

    students: list[dict] = []
    for row in rows:
        record = dict(row)
        total = int(record["attendance_events"])
        present = int(record["present_count"])
        record["attendance_percentage"] = round((present / total) * 100.0, 2) if total else 0.0
        students.append(record)
    return students


def list_face_samples() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT s.face_label, s.roll_no, fs.image_path
            FROM face_samples fs
            INNER JOIN students s ON s.id = fs.student_id
            ORDER BY s.face_label, fs.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_attendance(
    *,
    student_id: int,
    status: str,
    confidence: float,
    liveness_score: float,
    note: str,
    raw_label: str,
    claimed_roll_no: str,
) -> tuple[str, dict]:
    attendance_date = today_string()
    attendance_time = iso_timestamp().split("T", 1)[1]
    updated_at = iso_timestamp()

    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT *
            FROM attendance
            WHERE student_id = ? AND attendance_date = ?
            """,
            (student_id, attendance_date),
        ).fetchone()

        if existing is not None:
            existing_record = dict(existing)
            if existing_record["status"] == "Present" and status == "Present":
                return "duplicate", existing_record
            if existing_record["status"] == "Present" and status != "Present":
                return "preserved_present", existing_record

            connection.execute(
                """
                UPDATE attendance
                SET attendance_time = ?,
                    status = ?,
                    confidence = ?,
                    liveness_score = ?,
                    note = ?,
                    raw_label = ?,
                    claimed_roll_no = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    attendance_time,
                    status,
                    confidence,
                    liveness_score,
                    note,
                    raw_label,
                    claimed_roll_no,
                    updated_at,
                    existing_record["id"],
                ),
            )
            row = connection.execute("SELECT * FROM attendance WHERE id = ?", (existing_record["id"],)).fetchone()
            return "updated", dict(row) if row else existing_record

        cursor = connection.execute(
            """
            INSERT INTO attendance (
                student_id,
                attendance_date,
                attendance_time,
                status,
                confidence,
                liveness_score,
                note,
                raw_label,
                claimed_roll_no,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                attendance_date,
                attendance_time,
                status,
                confidence,
                liveness_score,
                note,
                raw_label,
                claimed_roll_no,
                updated_at,
            ),
        )
        row = connection.execute("SELECT * FROM attendance WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return "created", dict(row) if row else {}


def log_attendance_attempt(
    *,
    student_id: int | None,
    claimed_roll_no: str,
    official_status: str | None,
    attempt_outcome: str,
    confidence: float,
    liveness_score: float,
    note: str,
    raw_label: str,
    predicted_student_id: int | None = None,
) -> dict:
    attempt_date = today_string()
    attempt_time = iso_timestamp().split("T", 1)[1]
    created_at = iso_timestamp()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO attendance_attempts (
                student_id,
                claimed_roll_no,
                attempt_date,
                attempt_time,
                official_status,
                attempt_outcome,
                confidence,
                liveness_score,
                note,
                raw_label,
                predicted_student_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                claimed_roll_no,
                attempt_date,
                attempt_time,
                official_status,
                attempt_outcome,
                confidence,
                liveness_score,
                note,
                raw_label,
                predicted_student_id,
                created_at,
            ),
        )
        row = connection.execute("SELECT * FROM attendance_attempts WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row) if row else {}


def list_recent_attendance(limit: int = 100) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                a.*,
                s.first_name,
                s.last_name,
                s.roll_no,
                s.email,
                s.program,
                s.course,
                s.year
            FROM attendance a
            INNER JOIN students s ON s.id = a.student_id
            ORDER BY a.attendance_date DESC, a.attendance_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_recent_attempts(limit: int = 100) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                aa.*,
                s.first_name,
                s.last_name,
                s.roll_no,
                ps.first_name AS predicted_first_name,
                ps.last_name AS predicted_last_name,
                ps.roll_no AS predicted_roll_no
            FROM attendance_attempts aa
            LEFT JOIN students s ON s.id = aa.student_id
            LEFT JOIN students ps ON ps.id = aa.predicted_student_id
            ORDER BY aa.attempt_date DESC, aa.attempt_time DESC, aa.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def attendance_overview() -> dict:
    with get_connection() as connection:
        total_students = connection.execute("SELECT COUNT(*) AS value FROM students").fetchone()["value"]
        total_records = connection.execute("SELECT COUNT(*) AS value FROM attendance").fetchone()["value"]
        total_attempts = connection.execute("SELECT COUNT(*) AS value FROM attendance_attempts").fetchone()["value"]
        present_today = connection.execute(
            """
            SELECT COUNT(*) AS value
            FROM attendance
            WHERE attendance_date = ? AND status = 'Present'
            """,
            (today_string(),),
        ).fetchone()["value"]
        absent_today = connection.execute(
            """
            SELECT COUNT(*) AS value
            FROM attendance
            WHERE attendance_date = ? AND status = 'Absent'
            """,
            (today_string(),),
        ).fetchone()["value"]
        spoof_attempts_today = connection.execute(
            """
            SELECT COUNT(*) AS value
            FROM attendance_attempts
            WHERE attempt_date = ? AND attempt_outcome = 'spoof_attempt'
            """,
            (today_string(),),
        ).fetchone()["value"]
    return {
        "total_students": int(total_students),
        "total_records": int(total_records),
        "total_attempts": int(total_attempts),
        "present_today": int(present_today),
        "absent_today": int(absent_today),
        "spoof_attempts_today": int(spoof_attempts_today),
    }


def save_evaluation_report(report_key: str, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2)
    now = iso_timestamp()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO evaluation_reports (report_key, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(report_key) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (report_key, encoded, now, now),
        )


def get_evaluation_report(report_key: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT payload FROM evaluation_reports WHERE report_key = ?",
            (report_key,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["payload"])
