import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from src.config import (
    APP_TITLE,
    ATTENDANCE_MAX_FAILURES,
    ATTENDANCE_RATE_LIMIT_WINDOW_MINUTES,
    AUTH_MAX_FAILURES,
    AUTH_RATE_LIMIT_WINDOW_MINUTES,
    DATABASE_PATH,
    DATABASE_URL as CONFIG_DATABASE_URL,
    DEFAULT_FACULTY_NAME,
    DEFAULT_FACULTY_PASSWORD,
    DEFAULT_FACULTY_USERNAME,
    INSTITUTION_NAME,
)
from src.observability import write_event
from src.security import hash_password, password_needs_rehash, validate_password_strength, verify_password
from src.utils import ensure_directories, iso_timestamp, today_string


DEFAULT_ADMIN_USERNAME = os.getenv("SMARTATTEND_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("SMARTATTEND_ADMIN_PASSWORD", "admin123")
DATABASE_URL = CONFIG_DATABASE_URL
CURRENT_SCHEMA_VERSION = "2026050201"
CURRENT_SCHEMA_NAME = "production-platform-hardening"


def row_to_dict(row: Any | None) -> dict | None:
    return dict(row) if row is not None else None


def configured_database_url() -> str:
    runtime_value = os.getenv("SMARTATTEND_DATABASE_URL", "").strip()
    return runtime_value or DATABASE_URL


def database_backend() -> str:
    if configured_database_url().startswith(("postgres://", "postgresql://")):
        return "postgres"
    return "sqlite"


def _translate_sql(query: str) -> str:
    if database_backend() != "postgres":
        return query
    return query.replace("?", "%s")


def _split_sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    for part in script.split(";"):
        statement = part.strip()
        if statement:
            statements.append(statement)
    return statements


class DBConnection:
    def __init__(self, raw: Any, backend: str) -> None:
        self.raw = raw
        self.backend = backend

    def execute(self, query: str, params: Any | None = None):
        translated = _translate_sql(query)
        if self.backend == "sqlite":
            return self.raw.execute(translated, params or ())
        cursor = self.raw.cursor()
        cursor.execute(translated, params or ())
        return cursor

    def executescript(self, script: str) -> None:
        translated_script = script
        if self.backend == "postgres":
            translated_script = translated_script.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
        for statement in _split_sql_statements(translated_script):
            self.execute(statement)

    def commit(self) -> None:
        self.raw.commit()

    def close(self) -> None:
        self.raw.close()


@contextmanager
def get_connection(database_path: Path | None = None) -> Iterator[DBConnection]:
    backend = database_backend()
    if backend == "postgres":
        import psycopg
        from psycopg.rows import dict_row

        connection = psycopg.connect(configured_database_url(), autocommit=False, row_factory=dict_row)
        wrapped = DBConnection(connection, backend)
        try:
            yield wrapped
            wrapped.commit()
        finally:
            wrapped.close()
        return

    target = database_path or DATABASE_PATH
    ensure_directories(target.parent)
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    wrapped = DBConnection(connection, backend)
    try:
        yield wrapped
        wrapped.commit()
    finally:
        wrapped.close()


def _table_columns(connection: DBConnection, table_name: str) -> set[str]:
    if connection.backend == "postgres":
        rows = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            """,
            (table_name,),
        ).fetchall()
        return {row["column_name"] for row in rows}
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_column(connection: DBConnection, table_name: str, column_name: str, ddl: str) -> None:
    if connection.backend == "postgres":
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {ddl}")
        return
    if column_name not in _table_columns(connection, table_name):
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def _ensure_migrations_table(connection: DBConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def current_schema_version() -> str | None:
    with get_connection() as connection:
        _ensure_migrations_table(connection)
        row = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY applied_at DESC, version DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return row["version"] if isinstance(row, dict) else row["version"]


def list_schema_migrations() -> list[dict]:
    with get_connection() as connection:
        _ensure_migrations_table(connection)
        rows = connection.execute(
            "SELECT version, name, applied_at FROM schema_migrations ORDER BY applied_at DESC, version DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def _record_schema_migration(connection: DBConnection, version: str, name: str) -> None:
    connection.execute(
        """
        INSERT INTO schema_migrations (version, name, applied_at)
        VALUES (?, ?, ?)
        ON CONFLICT(version) DO NOTHING
        """,
        (version, name, iso_timestamp()),
    )


def init_database() -> None:
    with get_connection() as connection:
        _ensure_migrations_table(connection)
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

            CREATE TABLE IF NOT EXISTS evaluation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_key TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                code TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department_id INTEGER,
                name TEXT NOT NULL,
                code TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY(department_id) REFERENCES departments(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_id INTEGER,
                name TEXT NOT NULL,
                year_label TEXT NOT NULL,
                semester_label TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                UNIQUE(program_id, name, year_label, semester_label),
                FOREIGN KEY(program_id) REFERENCES programs(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS courses_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department_id INTEGER,
                course_code TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                credit_hours INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                FOREIGN KEY(department_id) REFERENCES departments(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS faculty_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                department_id INTEGER,
                title TEXT,
                employee_code TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(department_id) REFERENCES departments(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS course_offerings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER NOT NULL,
                section_id INTEGER NOT NULL,
                faculty_user_id INTEGER NOT NULL,
                term_name TEXT NOT NULL,
                academic_year TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                UNIQUE(course_id, section_id, faculty_user_id, term_name, academic_year),
                FOREIGN KEY(course_id) REFERENCES courses_catalog(id) ON DELETE CASCADE,
                FOREIGN KEY(section_id) REFERENCES sections(id) ON DELETE CASCADE,
                FOREIGN KEY(faculty_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS class_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                offering_id INTEGER NOT NULL,
                session_code TEXT NOT NULL UNIQUE,
                session_title TEXT NOT NULL,
                session_date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                attendance_open_at TEXT NOT NULL,
                attendance_close_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'scheduled',
                location TEXT,
                notes TEXT,
                created_by_user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(offering_id) REFERENCES course_offerings(id) ON DELETE CASCADE,
                FOREIGN KEY(created_by_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                liveness_score REAL,
                note TEXT,
                raw_label TEXT,
                claimed_roll_no TEXT,
                source TEXT NOT NULL DEFAULT 'live_verification',
                verified_by_user_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, student_id),
                FOREIGN KEY(session_id) REFERENCES class_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY(verified_by_user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS attendance_exceptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER,
                session_id INTEGER NOT NULL,
                student_id INTEGER,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                resolution TEXT,
                resolution_note TEXT,
                resolved_attendance_status TEXT,
                created_at TEXT NOT NULL,
                reviewed_by_user_id INTEGER,
                reviewed_at TEXT,
                FOREIGN KEY(attempt_id) REFERENCES attendance_attempts(id) ON DELETE SET NULL,
                FOREIGN KEY(session_id) REFERENCES class_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE SET NULL,
                FOREIGN KEY(reviewed_by_user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_user_id INTEGER,
                actor_role TEXT,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                payload TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(actor_user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                success INTEGER NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                version_tag TEXT NOT NULL,
                threshold REAL,
                metrics_payload TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                UNIQUE(model_key, version_tag)
            );

            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        _ensure_column(connection, "students", "section_id", "section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL")
        _ensure_column(connection, "students", "active", "active INTEGER NOT NULL DEFAULT 1")
        _ensure_column(connection, "students", "created_by_user_id", "created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL")
        _ensure_column(connection, "students", "primary_face_storage_uri", "primary_face_storage_uri TEXT")
        _ensure_column(connection, "face_samples", "storage_uri", "storage_uri TEXT")
        _ensure_column(connection, "attendance_attempts", "session_id", "session_id INTEGER REFERENCES class_sessions(id) ON DELETE SET NULL")
        _ensure_column(connection, "attendance_attempts", "needs_review", "needs_review INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "attendance_attempts", "review_status", "review_status TEXT")
        _ensure_column(connection, "attendance_attempts", "reviewed_by_user_id", "reviewed_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL")
        _ensure_column(connection, "attendance_attempts", "reviewed_at", "reviewed_at TEXT")

        seed_default_users(connection)
        seed_system_settings(connection)
        _record_schema_migration(connection, CURRENT_SCHEMA_VERSION, CURRENT_SCHEMA_NAME)


def seed_system_settings(connection: DBConnection) -> None:
    for key, value in {
        "institution_name": INSTITUTION_NAME,
        "application_title": APP_TITLE,
    }.items():
        connection.execute(
            """
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, iso_timestamp()),
        )


def seed_default_users(connection: DBConnection) -> None:
    created_at = iso_timestamp()
    password_hash = hash_password(DEFAULT_ADMIN_PASSWORD)
    connection.execute(
        """
        INSERT INTO users (username, email, full_name, role, password_hash, is_active, created_at, updated_at)
        VALUES (?, ?, ?, 'admin', ?, 1, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            full_name = excluded.full_name,
            password_hash = excluded.password_hash,
            updated_at = excluded.updated_at
        """,
        (DEFAULT_ADMIN_USERNAME, None, "Platform Administrator", password_hash, created_at, created_at),
    )

    if DEFAULT_FACULTY_USERNAME and DEFAULT_FACULTY_PASSWORD:
        connection.execute(
            """
            INSERT INTO users (username, email, full_name, role, password_hash, is_active, created_at, updated_at)
            VALUES (?, ?, ?, 'faculty', ?, 1, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                full_name = excluded.full_name,
                password_hash = excluded.password_hash,
                updated_at = excluded.updated_at
            """,
            (
                DEFAULT_FACULTY_USERNAME,
                None,
                DEFAULT_FACULTY_NAME,
                hash_password(DEFAULT_FACULTY_PASSWORD),
                created_at,
                created_at,
            ),
        )


# AUTH_AND_AUDIT


def audit_action(
    *,
    actor_user_id: int | None,
    actor_role: str | None,
    action: str,
    entity_type: str,
    entity_id: str | int | None,
    payload: dict[str, Any] | None = None,
) -> None:
    created_at = iso_timestamp()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO audit_logs (actor_user_id, actor_role, action, entity_type, entity_id, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor_user_id,
                actor_role,
                action,
                entity_type,
                str(entity_id) if entity_id is not None else None,
                json.dumps(payload or {}, default=str),
                created_at,
            ),
        )
    write_event(
        "audit_action",
        {
            "actor_user_id": actor_user_id,
            "actor_role": actor_role,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "payload": payload or {},
        },
    )


def list_audit_logs(limit: int = 200) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                a.*,
                u.full_name AS actor_name,
                u.username AS actor_username
            FROM audit_logs a
            LEFT JOIN users u ON u.id = a.actor_user_id
            ORDER BY a.created_at DESC, a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    records: list[dict] = []
    for row in rows:
        record = dict(row)
        try:
            record["payload"] = json.loads(record.get("payload") or "{}")
        except json.JSONDecodeError:
            record["payload"] = {}
        records.append(record)
    return records


def create_user(*, username: str, full_name: str, password: str, role: str, email: str | None = None, is_active: bool = True) -> dict:
    role = role.strip().lower()
    if role not in {"admin", "faculty"}:
        raise ValueError("Role must be 'admin' or 'faculty'.")
    strength = validate_password_strength(password)
    if not strength.ok:
        raise ValueError(strength.message)

    created_at = iso_timestamp()
    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO users (username, email, full_name, role, password_hash, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            """,
            (
                username.strip().lower(),
                email.strip().lower() if email else None,
                full_name.strip(),
                role,
                hash_password(password),
                1 if is_active else 0,
                created_at,
                created_at,
            ),
        ).fetchone()
    user = dict(row) if row else {}
    audit_action(actor_user_id=None, actor_role="system", action="user_created", entity_type="user", entity_id=user.get("id"), payload={"username": user.get("username"), "role": user.get("role")})
    return user


def update_user_password(user_id: int, new_password: str) -> None:
    strength = validate_password_strength(new_password)
    if not strength.ok:
        raise ValueError(strength.message)
    with get_connection() as connection:
        connection.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (hash_password(new_password), iso_timestamp(), user_id),
        )


def list_users(role: str | None = None) -> list[dict]:
    query = "SELECT * FROM users"
    parameters: list[Any] = []
    if role:
        query += " WHERE role = ?"
        parameters.append(role.strip().lower())
    query += " ORDER BY role, full_name"
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def get_user_by_username(username: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()
    return row_to_dict(row)


def get_user_by_id(user_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return row_to_dict(row)


def record_login_attempt(username: str, success: bool, reason: str) -> None:
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO login_attempts (username, success, reason, created_at) VALUES (?, ?, ?, ?)",
            (username.strip().lower(), 1 if success else 0, reason, iso_timestamp()),
        )


def recent_failed_login_count(username: str) -> int:
    cutoff = (datetime.now() - timedelta(minutes=AUTH_RATE_LIMIT_WINDOW_MINUTES)).isoformat(timespec="seconds")
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS value
            FROM login_attempts
            WHERE username = ? AND success = 0 AND created_at >= ?
            """,
            (username.strip().lower(), cutoff),
        ).fetchone()
    return int(row["value"]) if row else 0


def is_login_rate_limited(username: str) -> bool:
    return recent_failed_login_count(username) >= AUTH_MAX_FAILURES


def list_login_attempts(limit: int = 200) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM login_attempts ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def authenticate_user(username: str, password: str) -> tuple[dict | None, str]:
    normalized = username.strip().lower()
    if is_login_rate_limited(normalized):
        record_login_attempt(normalized, False, "rate_limited")
        return None, "Too many failed login attempts. Wait before retrying."

    user = get_user_by_username(normalized)
    if user is None:
        record_login_attempt(normalized, False, "unknown_user")
        return None, "Invalid username or password."
    if not user["is_active"]:
        record_login_attempt(normalized, False, "inactive_user")
        return None, "This user account is inactive."
    if not verify_password(password, user["password_hash"]):
        record_login_attempt(normalized, False, "invalid_password")
        return None, "Invalid username or password."

    with get_connection() as connection:
        new_hash = hash_password(password) if password_needs_rehash(user["password_hash"]) else user["password_hash"]
        connection.execute(
            "UPDATE users SET password_hash = ?, last_login_at = ?, updated_at = ? WHERE id = ?",
            (new_hash, iso_timestamp(), iso_timestamp(), user["id"]),
        )
    record_login_attempt(normalized, True, "success")
    refreshed = get_user_by_id(user["id"])
    audit_action(actor_user_id=user["id"], actor_role=user["role"], action="login_success", entity_type="user", entity_id=user["id"], payload={"username": user["username"]})
    return refreshed, "Login successful."


def verify_admin(username: str, password: str) -> bool:
    user, _message = authenticate_user(username, password)
    return bool(user and user["role"] == "admin")


# ACADEMIC_DOMAIN


def create_department(name: str, code: str) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO departments (name, code, created_at)
            VALUES (?, ?, ?)
            RETURNING *
            """,
            (name.strip(), code.strip().upper(), created_at),
        ).fetchone()
    return dict(row) if row else {}


def list_departments() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM departments ORDER BY name").fetchall()
    return [dict(row) for row in rows]


def create_program(*, department_id: int | None, name: str, code: str) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO programs (department_id, name, code, created_at)
            VALUES (?, ?, ?, ?)
            RETURNING *
            """,
            (department_id, name.strip(), code.strip().upper(), created_at),
        ).fetchone()
    return dict(row) if row else {}


def list_programs() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.*, d.name AS department_name, d.code AS department_code
            FROM programs p
            LEFT JOIN departments d ON d.id = p.department_id
            ORDER BY p.name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_section(*, program_id: int | None, name: str, year_label: str, semester_label: str | None = None, active: bool = True) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO sections (program_id, name, year_label, semester_label, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING *
            """,
            (program_id, name.strip(), year_label.strip(), (semester_label or "").strip() or None, 1 if active else 0, created_at),
        ).fetchone()
    return dict(row) if row else {}


def list_sections(program_id: int | None = None) -> list[dict]:
    query = """
        SELECT
            s.*,
            p.name AS program_name,
            p.code AS program_code
        FROM sections s
        LEFT JOIN programs p ON p.id = s.program_id
    """
    parameters: list[Any] = []
    if program_id is not None:
        query += " WHERE s.program_id = ?"
        parameters.append(program_id)
    query += " ORDER BY p.name, s.year_label, s.name"
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def create_course(*, department_id: int | None, course_code: str, title: str, credit_hours: int = 3) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO courses_catalog (department_id, course_code, title, credit_hours, created_at)
            VALUES (?, ?, ?, ?, ?)
            RETURNING *
            """,
            (department_id, course_code.strip().upper(), title.strip(), int(credit_hours), created_at),
        ).fetchone()
    return dict(row) if row else {}


def list_courses() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.*,
                d.name AS department_name,
                d.code AS department_code
            FROM courses_catalog c
            LEFT JOIN departments d ON d.id = c.department_id
            ORDER BY c.course_code
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_faculty_profile(*, user_id: int, department_id: int | None, title: str = "", employee_code: str | None = None) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO faculty_profiles (user_id, department_id, title, employee_code, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                department_id = excluded.department_id,
                title = excluded.title,
                employee_code = excluded.employee_code
            """,
            (user_id, department_id, title.strip(), employee_code, created_at),
        )
        row = connection.execute(
            """
            SELECT fp.*, u.full_name, u.username, u.email, d.name AS department_name
            FROM faculty_profiles fp
            INNER JOIN users u ON u.id = fp.user_id
            LEFT JOIN departments d ON d.id = fp.department_id
            WHERE fp.user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else {}


def list_faculty_users() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                u.*,
                fp.title,
                fp.employee_code,
                d.name AS department_name,
                d.code AS department_code
            FROM users u
            LEFT JOIN faculty_profiles fp ON fp.user_id = u.id
            LEFT JOIN departments d ON d.id = fp.department_id
            WHERE u.role = 'faculty'
            ORDER BY u.full_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_course_offering(*, course_id: int, section_id: int, faculty_user_id: int, term_name: str, academic_year: str, active: bool = True) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        inserted = connection.execute(
            """
            INSERT INTO course_offerings (course_id, section_id, faculty_user_id, term_name, academic_year, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (course_id, section_id, faculty_user_id, term_name.strip(), academic_year.strip(), 1 if active else 0, created_at),
        ).fetchone()
        row = connection.execute(
            """
            SELECT
                o.*,
                c.course_code,
                c.title AS course_title,
                s.name AS section_name,
                s.year_label,
                u.full_name AS faculty_name
            FROM course_offerings o
            INNER JOIN courses_catalog c ON c.id = o.course_id
            INNER JOIN sections s ON s.id = o.section_id
            INNER JOIN users u ON u.id = o.faculty_user_id
            WHERE o.id = ?
            """,
            (inserted["id"],),
        ).fetchone()
    return dict(row) if row else {}


def list_course_offerings(*, user: dict | None = None, active_only: bool = False) -> list[dict]:
    query = """
        SELECT
            o.*,
            c.course_code,
            c.title AS course_title,
            s.name AS section_name,
            s.year_label,
            s.semester_label,
            p.name AS program_name,
            u.full_name AS faculty_name
        FROM course_offerings o
        INNER JOIN courses_catalog c ON c.id = o.course_id
        INNER JOIN sections s ON s.id = o.section_id
        LEFT JOIN programs p ON p.id = s.program_id
        INNER JOIN users u ON u.id = o.faculty_user_id
    """
    clauses: list[str] = []
    parameters: list[Any] = []
    if user and user.get("role") == "faculty":
        clauses.append("o.faculty_user_id = ?")
        parameters.append(user["id"])
    if active_only:
        clauses.append("o.active = 1")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY o.academic_year DESC, o.term_name DESC, c.course_code, s.year_label, s.name"
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def get_course_offering(offering_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                o.*,
                c.course_code,
                c.title AS course_title,
                s.name AS section_name,
                s.year_label,
                p.name AS program_name,
                u.full_name AS faculty_name
            FROM course_offerings o
            INNER JOIN courses_catalog c ON c.id = o.course_id
            INNER JOIN sections s ON s.id = o.section_id
            LEFT JOIN programs p ON p.id = s.program_id
            INNER JOIN users u ON u.id = o.faculty_user_id
            WHERE o.id = ?
            """,
            (offering_id,),
        ).fetchone()
    return row_to_dict(row)


def create_class_session(
    *,
    offering_id: int,
    session_title: str,
    session_date: str,
    start_time: str,
    end_time: str,
    attendance_open_at: str,
    attendance_close_at: str,
    location: str,
    notes: str,
    created_by_user_id: int,
) -> dict:
    created_at = iso_timestamp()
    session_code = f"S{datetime.now().strftime('%Y%m%d%H%M%S')}{offering_id}"
    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO class_sessions (
                offering_id,
                session_code,
                session_title,
                session_date,
                start_time,
                end_time,
                attendance_open_at,
                attendance_close_at,
                status,
                location,
                notes,
                created_by_user_id,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?, ?)
            RETURNING *
            """,
            (
                offering_id,
                session_code,
                session_title.strip(),
                session_date,
                start_time,
                end_time,
                attendance_open_at,
                attendance_close_at,
                location.strip(),
                notes.strip(),
                created_by_user_id,
                created_at,
                created_at,
            ),
        ).fetchone()
    audit_action(actor_user_id=created_by_user_id, actor_role=None, action="class_session_created", entity_type="class_session", entity_id=row["id"] if row else None, payload={"offering_id": offering_id, "session_code": session_code})
    return dict(row) if row else {}


def list_class_sessions(*, user: dict | None = None, status: str | None = None, offering_id: int | None = None, limit: int = 200) -> list[dict]:
    query = """
        SELECT
            s.*,
            o.faculty_user_id,
            c.course_code,
            c.title AS course_title,
            sec.name AS section_name,
            sec.year_label,
            u.full_name AS faculty_name
        FROM class_sessions s
        INNER JOIN course_offerings o ON o.id = s.offering_id
        INNER JOIN courses_catalog c ON c.id = o.course_id
        INNER JOIN sections sec ON sec.id = o.section_id
        INNER JOIN users u ON u.id = o.faculty_user_id
    """
    clauses: list[str] = []
    parameters: list[Any] = []
    if user and user.get("role") == "faculty":
        clauses.append("o.faculty_user_id = ?")
        parameters.append(user["id"])
    if status:
        clauses.append("s.status = ?")
        parameters.append(status)
    if offering_id is not None:
        clauses.append("s.offering_id = ?")
        parameters.append(offering_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY s.session_date DESC, s.start_time DESC, s.id DESC LIMIT ?"
    parameters.append(limit)
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def get_class_session(session_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                s.*,
                o.faculty_user_id,
                o.section_id,
                c.course_code,
                c.title AS course_title,
                sec.name AS section_name,
                sec.year_label,
                u.full_name AS faculty_name
            FROM class_sessions s
            INNER JOIN course_offerings o ON o.id = s.offering_id
            INNER JOIN courses_catalog c ON c.id = o.course_id
            INNER JOIN sections sec ON sec.id = o.section_id
            INNER JOIN users u ON u.id = o.faculty_user_id
            WHERE s.id = ?
            """,
            (session_id,),
        ).fetchone()
    return row_to_dict(row)


def update_class_session_status(session_id: int, status: str, actor_user_id: int) -> dict | None:
    with get_connection() as connection:
        connection.execute(
            "UPDATE class_sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, iso_timestamp(), session_id),
        )
        row = connection.execute("SELECT * FROM class_sessions WHERE id = ?", (session_id,)).fetchone()
    audit_action(actor_user_id=actor_user_id, actor_role=None, action=f"class_session_{status}", entity_type="class_session", entity_id=session_id, payload={})
    return row_to_dict(row)


# STUDENTS_AND_ATTENDANCE


def student_exists(roll_no: str, email: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM students WHERE roll_no = ? OR email = ?",
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
    section_id: int | None = None,
    created_by_user_id: int | None = None,
    primary_face_storage_uri: str | None = None,
) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        inserted = connection.execute(
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
                section_id,
                active,
                created_by_user_id,
                primary_face_storage_uri,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            RETURNING id
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
                section_id,
                created_by_user_id,
                primary_face_storage_uri,
                created_at,
            ),
        ).fetchone()
        student_id = inserted["id"]
        connection.execute(
            """
            INSERT INTO face_samples (student_id, image_path, storage_uri, source, created_at)
            VALUES (?, ?, ?, 'enrollment', ?)
            """,
            (student_id, primary_face_path, primary_face_storage_uri, created_at),
        )
        row = connection.execute(
            """
            SELECT
                st.*,
                sec.name AS section_name,
                sec.year_label AS section_year_label
            FROM students st
            LEFT JOIN sections sec ON sec.id = st.section_id
            WHERE st.id = ?
            """,
            (student_id,),
        ).fetchone()
    return dict(row) if row else {}


def add_face_sample(student_id: int, image_path: str, source: str = "manual", storage_uri: str | None = None) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO face_samples (student_id, image_path, storage_uri, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_id, image_path, storage_uri, source, iso_timestamp()),
        )


def get_student_by_id(student_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    return row_to_dict(row)


def get_student_by_roll_no(roll_no: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                st.*,
                sec.name AS section_name,
                sec.year_label AS section_year_label
            FROM students st
            LEFT JOIN sections sec ON sec.id = st.section_id
            WHERE st.roll_no = ?
            """,
            (roll_no.strip(),),
        ).fetchone()
    return row_to_dict(row)


def get_student_by_label(face_label: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                st.*,
                sec.name AS section_name,
                sec.year_label AS section_year_label
            FROM students st
            LEFT JOIN sections sec ON sec.id = st.section_id
            WHERE st.face_label = ?
            """,
            (face_label,),
        ).fetchone()
    return row_to_dict(row)


def list_students(*, user: dict | None = None, offering_id: int | None = None, section_id: int | None = None) -> list[dict]:
    attendance_summary = """
        SELECT
            ar.student_id,
            COUNT(ar.id) AS attendance_events,
            COALESCE(SUM(CASE WHEN ar.status = 'Present' THEN 1 ELSE 0 END), 0) AS present_count,
            COALESCE(SUM(CASE WHEN ar.status = 'Absent' THEN 1 ELSE 0 END), 0) AS absent_count,
            COALESCE(SUM(CASE WHEN ar.status = 'Late' THEN 1 ELSE 0 END), 0) AS late_count,
            COALESCE(SUM(CASE WHEN ar.status = 'Excused' THEN 1 ELSE 0 END), 0) AS excused_count
        FROM attendance_records ar
        INNER JOIN class_sessions cs ON cs.id = ar.session_id
        INNER JOIN course_offerings co ON co.id = cs.offering_id
    """
    attendance_clauses: list[str] = []
    attendance_params: list[Any] = []
    outer_clauses: list[str] = ["st.active = 1"]
    outer_params: list[Any] = []

    if user and user.get("role") == "faculty":
        attendance_clauses.append("co.faculty_user_id = ?")
        attendance_params.append(user["id"])
        outer_clauses.append(
            "EXISTS (SELECT 1 FROM course_offerings co_scope WHERE co_scope.section_id = st.section_id AND co_scope.faculty_user_id = ?)"
        )
        outer_params.append(user["id"])
    if offering_id is not None:
        attendance_clauses.append("co.id = ?")
        attendance_params.append(offering_id)
        outer_clauses.append(
            "EXISTS (SELECT 1 FROM course_offerings co_scope WHERE co_scope.id = ? AND co_scope.section_id = st.section_id)"
        )
        outer_params.append(offering_id)
    if section_id is not None:
        outer_clauses.append("st.section_id = ?")
        outer_params.append(section_id)

    if attendance_clauses:
        attendance_summary += " WHERE " + " AND ".join(attendance_clauses)
    attendance_summary += " GROUP BY ar.student_id"

    query = f"""
        SELECT
            st.*,
            sec.name AS section_name,
            sec.year_label AS section_year_label,
            p.name AS program_name,
            COALESCE(stats.attendance_events, 0) AS attendance_events,
            COALESCE(stats.present_count, 0) AS present_count,
            COALESCE(stats.absent_count, 0) AS absent_count,
            COALESCE(stats.late_count, 0) AS late_count,
            COALESCE(stats.excused_count, 0) AS excused_count
        FROM students st
        LEFT JOIN sections sec ON sec.id = st.section_id
        LEFT JOIN programs p ON p.id = sec.program_id
        LEFT JOIN ({attendance_summary}) stats ON stats.student_id = st.id
        WHERE {" AND ".join(outer_clauses)}
        ORDER BY st.first_name, st.last_name
    """
    parameters = [*attendance_params, *outer_params]
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    students: list[dict] = []
    for row in rows:
        record = dict(row)
        total = int(record["attendance_events"])
        present_like = int(record["present_count"]) + int(record["late_count"]) + int(record["excused_count"])
        record["attendance_percentage"] = round((present_like / total) * 100.0, 2) if total else 0.0
        students.append(record)
    return students


def list_face_samples() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT s.face_label, s.roll_no, fs.image_path, fs.storage_uri
            FROM face_samples fs
            INNER JOIN students s ON s.id = fs.student_id
            ORDER BY s.face_label, fs.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def is_student_in_session(student_id: int, session_id: int) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM students st
            INNER JOIN class_sessions cs ON cs.id = ?
            INNER JOIN course_offerings co ON co.id = cs.offering_id
            WHERE st.id = ? AND st.section_id = co.section_id AND st.active = 1
            """,
            (session_id, student_id),
        ).fetchone()
    return row is not None


def recent_failed_attendance_attempts(session_id: int, claimed_roll_no: str) -> int:
    cutoff = (datetime.now() - timedelta(minutes=ATTENDANCE_RATE_LIMIT_WINDOW_MINUTES)).isoformat(timespec="seconds")
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS value
            FROM attendance_attempts
            WHERE session_id = ? AND claimed_roll_no = ? AND attempt_outcome != 'verified_present' AND created_at >= ?
            """,
            (session_id, claimed_roll_no.strip(), cutoff),
        ).fetchone()
    return int(row["value"]) if row else 0


def is_attendance_rate_limited(session_id: int, claimed_roll_no: str) -> bool:
    return recent_failed_attendance_attempts(session_id, claimed_roll_no) >= ATTENDANCE_MAX_FAILURES


def upsert_session_attendance(
    *,
    session_id: int,
    student_id: int,
    status: str,
    confidence: float,
    liveness_score: float,
    note: str,
    raw_label: str,
    claimed_roll_no: str,
    source: str,
    verified_by_user_id: int | None,
) -> tuple[str, dict]:
    timestamp = iso_timestamp()
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT * FROM attendance_records WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
        if existing is not None:
            existing_record = dict(existing)
            if existing_record["status"] == "Present" and status == "Present":
                return "duplicate", existing_record
            connection.execute(
                """
                UPDATE attendance_records
                SET status = ?, confidence = ?, liveness_score = ?, note = ?, raw_label = ?,
                    claimed_roll_no = ?, source = ?, verified_by_user_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    confidence,
                    liveness_score,
                    note,
                    raw_label,
                    claimed_roll_no,
                    source,
                    verified_by_user_id,
                    timestamp,
                    existing_record["id"],
                ),
            )
            row = connection.execute("SELECT * FROM attendance_records WHERE id = ?", (existing_record["id"],)).fetchone()
            return "updated", dict(row) if row else existing_record

        cursor = connection.execute(
            """
            INSERT INTO attendance_records (
                session_id, student_id, status, confidence, liveness_score, note, raw_label,
                claimed_roll_no, source, verified_by_user_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            """,
            (
                session_id,
                student_id,
                status,
                confidence,
                liveness_score,
                note,
                raw_label,
                claimed_roll_no,
                source,
                verified_by_user_id,
                timestamp,
                timestamp,
            ),
        )
        row = cursor.fetchone()
    return "created", dict(row) if row else {}


def create_exception_from_attempt(*, attempt_id: int | None, session_id: int, student_id: int | None, reason: str) -> dict:
    created_at = iso_timestamp()
    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO attendance_exceptions (attempt_id, session_id, student_id, reason, status, created_at)
            VALUES (?, ?, ?, ?, 'open', ?)
            RETURNING *
            """,
            (attempt_id, session_id, student_id, reason, created_at),
        ).fetchone()
    return dict(row) if row else {}


def resolve_exception(
    *,
    exception_id: int,
    reviewer_user_id: int,
    resolution: str,
    resolution_note: str,
    resolved_attendance_status: str | None = None,
) -> dict | None:
    reviewed_at = iso_timestamp()
    with get_connection() as connection:
        exception_row = connection.execute("SELECT * FROM attendance_exceptions WHERE id = ?", (exception_id,)).fetchone()
        if exception_row is None:
            return None
        exception_record = dict(exception_row)
        connection.execute(
            """
            UPDATE attendance_exceptions
            SET status = 'resolved',
                resolution = ?,
                resolution_note = ?,
                resolved_attendance_status = ?,
                reviewed_by_user_id = ?,
                reviewed_at = ?
            WHERE id = ?
            """,
            (
                resolution,
                resolution_note.strip(),
                resolved_attendance_status,
                reviewer_user_id,
                reviewed_at,
                exception_id,
            ),
        )

    if resolved_attendance_status and exception_record["student_id"]:
        student = get_student_by_id(exception_record["student_id"])
        upsert_session_attendance(
            session_id=exception_record["session_id"],
            student_id=exception_record["student_id"],
            status=resolved_attendance_status,
            confidence=0.0,
            liveness_score=0.0,
            note=f"exception resolved: {resolution}",
            raw_label="manual_review",
            claimed_roll_no=student["roll_no"] if student else "",
            source="exception_review",
            verified_by_user_id=reviewer_user_id,
        )

    audit_action(actor_user_id=reviewer_user_id, actor_role=None, action="exception_resolved", entity_type="attendance_exception", entity_id=exception_id, payload={"resolution": resolution, "resolved_attendance_status": resolved_attendance_status})
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM attendance_exceptions WHERE id = ?", (exception_id,)).fetchone()
    return row_to_dict(row)


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
    session_id: int | None = None,
    needs_review: bool = False,
) -> dict:
    attempt_date = today_string()
    attempt_time = iso_timestamp().split("T", 1)[1]
    created_at = iso_timestamp()

    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO attendance_attempts (
                student_id,
                session_id,
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
                needs_review,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            """,
            (
                student_id,
                session_id,
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
                1 if needs_review else 0,
                created_at,
            ),
        ).fetchone()
    return dict(row) if row else {}


def finalize_session_absences(session_id: int, actor_user_id: int) -> int:
    session = get_class_session(session_id)
    if session is None:
        return 0
    timestamp = iso_timestamp()
    created = 0
    with get_connection() as connection:
        students = connection.execute(
            """
            SELECT id, roll_no
            FROM students
            WHERE section_id = ? AND active = 1
            """,
            (session["section_id"],),
        ).fetchall()
        for student in students:
            existing = connection.execute(
                "SELECT id FROM attendance_records WHERE session_id = ? AND student_id = ?",
                (session_id, student["id"]),
            ).fetchone()
            if existing:
                continue
            connection.execute(
                """
                INSERT INTO attendance_records (
                    session_id, student_id, status, confidence, liveness_score, note, raw_label,
                    claimed_roll_no, source, verified_by_user_id, created_at, updated_at
                )
                VALUES (?, ?, 'Absent', 0, 0, 'no successful attendance verification before session close', 'Absent', ?, 'finalized_absence', ?, ?, ?)
                """,
                (session_id, student["id"], student["roll_no"], actor_user_id, timestamp, timestamp),
            )
            created += 1
        connection.execute(
            "UPDATE class_sessions SET status = 'closed', updated_at = ? WHERE id = ?",
            (timestamp, session_id),
        )
    audit_action(actor_user_id=actor_user_id, actor_role=None, action="class_session_finalized", entity_type="class_session", entity_id=session_id, payload={"finalized_absences": created})
    return created


# REPORTING_AND_COMPAT


def list_recent_attendance(limit: int = 100, user: dict | None = None) -> list[dict]:
    query = """
        SELECT
            ar.*,
            st.first_name,
            st.last_name,
            st.roll_no,
            st.email,
            st.program,
            st.course,
            st.year,
            cs.session_title,
            cs.session_date,
            cs.session_code,
            c.course_code,
            c.title AS course_title,
            sec.name AS section_name
        FROM attendance_records ar
        INNER JOIN students st ON st.id = ar.student_id
        INNER JOIN class_sessions cs ON cs.id = ar.session_id
        INNER JOIN course_offerings co ON co.id = cs.offering_id
        INNER JOIN courses_catalog c ON c.id = co.course_id
        INNER JOIN sections sec ON sec.id = co.section_id
    """
    parameters: list[Any] = []
    clauses: list[str] = []
    if user and user.get("role") == "faculty":
        clauses.append("co.faculty_user_id = ?")
        parameters.append(user["id"])
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY cs.session_date DESC, ar.updated_at DESC, ar.id DESC LIMIT ?"
    parameters.append(limit)
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def list_recent_attempts(limit: int = 100, user: dict | None = None) -> list[dict]:
    query = """
        SELECT
            aa.*,
            st.first_name,
            st.last_name,
            st.roll_no,
            ps.first_name AS predicted_first_name,
            ps.last_name AS predicted_last_name,
            ps.roll_no AS predicted_roll_no,
            cs.session_title,
            cs.session_code,
            c.course_code,
            c.title AS course_title
        FROM attendance_attempts aa
        LEFT JOIN students st ON st.id = aa.student_id
        LEFT JOIN students ps ON ps.id = aa.predicted_student_id
        LEFT JOIN class_sessions cs ON cs.id = aa.session_id
        LEFT JOIN course_offerings co ON co.id = cs.offering_id
        LEFT JOIN courses_catalog c ON c.id = co.course_id
    """
    parameters: list[Any] = []
    clauses: list[str] = []
    if user and user.get("role") == "faculty":
        clauses.append("co.faculty_user_id = ?")
        parameters.append(user["id"])
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY aa.created_at DESC, aa.id DESC LIMIT ?"
    parameters.append(limit)
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def list_exceptions(*, user: dict | None = None, status: str | None = None, limit: int = 200) -> list[dict]:
    query = """
        SELECT
            ex.*,
            st.first_name,
            st.last_name,
            st.roll_no,
            cs.session_title,
            cs.session_code,
            cs.session_date,
            c.course_code,
            c.title AS course_title,
            sec.name AS section_name,
            reviewer.full_name AS reviewer_name
        FROM attendance_exceptions ex
        LEFT JOIN students st ON st.id = ex.student_id
        INNER JOIN class_sessions cs ON cs.id = ex.session_id
        INNER JOIN course_offerings co ON co.id = cs.offering_id
        INNER JOIN courses_catalog c ON c.id = co.course_id
        INNER JOIN sections sec ON sec.id = co.section_id
        LEFT JOIN users reviewer ON reviewer.id = ex.reviewed_by_user_id
    """
    clauses: list[str] = []
    parameters: list[Any] = []
    if user and user.get("role") == "faculty":
        clauses.append("co.faculty_user_id = ?")
        parameters.append(user["id"])
    if status:
        clauses.append("ex.status = ?")
        parameters.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY ex.created_at DESC, ex.id DESC LIMIT ?"
    parameters.append(limit)
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def list_attendance_records(*, user: dict | None = None, session_id: int | None = None, offering_id: int | None = None, limit: int = 500) -> list[dict]:
    query = """
        SELECT
            ar.*,
            st.first_name,
            st.last_name,
            st.roll_no,
            cs.session_title,
            cs.session_code,
            cs.session_date,
            c.course_code,
            c.title AS course_title,
            sec.name AS section_name
        FROM attendance_records ar
        INNER JOIN students st ON st.id = ar.student_id
        INNER JOIN class_sessions cs ON cs.id = ar.session_id
        INNER JOIN course_offerings co ON co.id = cs.offering_id
        INNER JOIN courses_catalog c ON c.id = co.course_id
        INNER JOIN sections sec ON sec.id = co.section_id
    """
    clauses: list[str] = []
    parameters: list[Any] = []
    if user and user.get("role") == "faculty":
        clauses.append("co.faculty_user_id = ?")
        parameters.append(user["id"])
    if session_id is not None:
        clauses.append("ar.session_id = ?")
        parameters.append(session_id)
    if offering_id is not None:
        clauses.append("co.id = ?")
        parameters.append(offering_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY cs.session_date DESC, ar.updated_at DESC LIMIT ?"
    parameters.append(limit)
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def attendance_overview(user: dict | None = None) -> dict:
    offerings = list_course_offerings(user=user)
    offering_ids = [offering["id"] for offering in offerings]
    with get_connection() as connection:
        total_students_query = "SELECT COUNT(*) AS value FROM students WHERE active = 1"
        total_students_params: list[Any] = []
        if user and user.get("role") == "faculty":
            total_students_query = """
                SELECT COUNT(DISTINCT st.id) AS value
                FROM students st
                INNER JOIN course_offerings co ON co.section_id = st.section_id
                WHERE st.active = 1 AND co.faculty_user_id = ?
            """
            total_students_params = [user["id"]]
        total_students = connection.execute(total_students_query, total_students_params).fetchone()["value"]
        if offering_ids:
            placeholders = ",".join("?" for _ in offering_ids)
            total_records = connection.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM attendance_records ar
                INNER JOIN class_sessions cs ON cs.id = ar.session_id
                WHERE cs.offering_id IN ({placeholders})
                """,
                offering_ids,
            ).fetchone()["value"]
            total_attempts = connection.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM attendance_attempts aa
                INNER JOIN class_sessions cs ON cs.id = aa.session_id
                WHERE cs.offering_id IN ({placeholders})
                """,
                offering_ids,
            ).fetchone()["value"]
            open_sessions = connection.execute(
                f"SELECT COUNT(*) AS value FROM class_sessions WHERE status = 'open' AND offering_id IN ({placeholders})",
                offering_ids,
            ).fetchone()["value"]
            present_today = connection.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM attendance_records ar
                INNER JOIN class_sessions cs ON cs.id = ar.session_id
                WHERE ar.status = 'Present' AND cs.session_date = ? AND cs.offering_id IN ({placeholders})
                """,
                [today_string(), *offering_ids],
            ).fetchone()["value"]
            absent_today = connection.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM attendance_records ar
                INNER JOIN class_sessions cs ON cs.id = ar.session_id
                WHERE ar.status = 'Absent' AND cs.session_date = ? AND cs.offering_id IN ({placeholders})
                """,
                [today_string(), *offering_ids],
            ).fetchone()["value"]
            spoof_attempts_today = connection.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM attendance_attempts aa
                INNER JOIN class_sessions cs ON cs.id = aa.session_id
                WHERE aa.attempt_outcome = 'spoof_attempt' AND aa.attempt_date = ? AND cs.offering_id IN ({placeholders})
                """,
                [today_string(), *offering_ids],
            ).fetchone()["value"]
            open_exceptions = connection.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM attendance_exceptions ex
                INNER JOIN class_sessions cs ON cs.id = ex.session_id
                WHERE ex.status = 'open' AND cs.offering_id IN ({placeholders})
                """,
                offering_ids,
            ).fetchone()["value"]
        elif user and user.get("role") == "faculty":
            total_records = 0
            total_attempts = 0
            open_sessions = 0
            present_today = 0
            absent_today = 0
            spoof_attempts_today = 0
            open_exceptions = 0
        else:
            total_records = connection.execute("SELECT COUNT(*) AS value FROM attendance_records").fetchone()["value"]
            total_attempts = connection.execute("SELECT COUNT(*) AS value FROM attendance_attempts").fetchone()["value"]
            open_sessions = connection.execute("SELECT COUNT(*) AS value FROM class_sessions WHERE status = 'open'").fetchone()["value"]
            present_today = connection.execute(
                """
                SELECT COUNT(*) AS value
                FROM attendance_records ar
                INNER JOIN class_sessions cs ON cs.id = ar.session_id
                WHERE ar.status = 'Present' AND cs.session_date = ?
                """,
                (today_string(),),
            ).fetchone()["value"]
            absent_today = connection.execute(
                """
                SELECT COUNT(*) AS value
                FROM attendance_records ar
                INNER JOIN class_sessions cs ON cs.id = ar.session_id
                WHERE ar.status = 'Absent' AND cs.session_date = ?
                """,
                (today_string(),),
            ).fetchone()["value"]
            spoof_attempts_today = connection.execute(
                """
                SELECT COUNT(*) AS value
                FROM attendance_attempts
                WHERE attempt_outcome = 'spoof_attempt' AND attempt_date = ?
                """,
                (today_string(),),
            ).fetchone()["value"]
            open_exceptions = connection.execute(
                "SELECT COUNT(*) AS value FROM attendance_exceptions WHERE status = 'open'"
            ).fetchone()["value"]

    return {
        "total_students": int(total_students),
        "total_records": int(total_records),
        "total_attempts": int(total_attempts),
        "present_today": int(present_today),
        "absent_today": int(absent_today),
        "spoof_attempts_today": int(spoof_attempts_today),
        "open_sessions": int(open_sessions),
        "open_exceptions": int(open_exceptions),
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

        threshold = None
        if "threshold" in payload:
            try:
                threshold = float(payload["threshold"])
            except (TypeError, ValueError):
                threshold = None
        connection.execute(
            """
            INSERT INTO model_registry (model_key, version_tag, threshold, metrics_payload, is_active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(model_key, version_tag) DO UPDATE SET
                threshold = excluded.threshold,
                metrics_payload = excluded.metrics_payload,
                is_active = excluded.is_active
            """,
            (report_key, now, threshold, encoded, now),
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


def list_model_versions() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM model_registry ORDER BY created_at DESC, id DESC").fetchall()
    records: list[dict] = []
    for row in rows:
        record = dict(row)
        try:
            record["metrics_payload"] = json.loads(record.get("metrics_payload") or "{}")
        except json.JSONDecodeError:
            record["metrics_payload"] = {}
        records.append(record)
    return records


def export_attendance_csv(*, session_id: int | None = None, offering_id: int | None = None, user: dict | None = None) -> list[dict]:
    return list_attendance_records(user=user, session_id=session_id, offering_id=offering_id, limit=5000)


def export_exception_csv(*, user: dict | None = None) -> list[dict]:
    return list_exceptions(user=user, limit=5000)


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
            "SELECT * FROM attendance WHERE student_id = ? AND attendance_date = ?",
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
                SET attendance_time = ?, status = ?, confidence = ?, liveness_score = ?, note = ?,
                    raw_label = ?, claimed_roll_no = ?, updated_at = ?
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

        row = connection.execute(
            """
            INSERT INTO attendance (
                student_id, attendance_date, attendance_time, status, confidence, liveness_score,
                note, raw_label, claimed_roll_no, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
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
        ).fetchone()
    return "created", dict(row) if row else {}
