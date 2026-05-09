from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.attendance_service import verify_attendance_attempt
from src.config import (
    APP_TITLE,
    ARTIFACTS_DIR,
    DATABASE_PATH,
    FACE_LABELS_PATH,
    FACE_MODEL_PATH,
    INSTITUTION_NAME,
    LIVENESS_MODEL_PATH,
    SESSION_TIMEOUT_MINUTES,
    STORAGE_BACKEND,
)
from src.database import (
    DEFAULT_ADMIN_USERNAME,
    attendance_overview,
    authenticate_user,
    create_course,
    create_course_offering,
    create_department,
    create_faculty_profile,
    create_program,
    create_section,
    create_user,
    export_attendance_csv,
    export_exception_csv,
    finalize_session_absences,
    get_evaluation_report,
    init_database,
    list_audit_logs,
    list_attendance_records,
    list_class_sessions,
    list_course_offerings,
    list_courses,
    list_departments,
    list_exceptions,
    list_faculty_users,
    list_login_attempts,
    list_model_versions,
    list_programs,
    list_recent_attendance,
    list_recent_attempts,
    list_sections,
    list_students,
    list_users,
    resolve_exception,
    update_class_session_status,
    create_class_session,
)
from src.enrollment_service import enroll_student
from src.evaluate_models import run_all_evaluations
from src.face_detector import FaceDetector
from src.liveness import LivenessDetector
from src.liveness_dataset_service import dataset_ready_for_training, liveness_counts, save_liveness_sample
from src.recognizer import FaceRecognizer
from src.storage import get_storage_backend
from src.train_liveness_model import train_liveness_model
from src.utils import decode_uploaded_image


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_database()
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "university_logo.png"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
              radial-gradient(circle at top left, rgba(201, 93, 61, 0.16), transparent 24%),
              linear-gradient(135deg, #f7eee2 0%, #f2e1cf 100%);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background:
              radial-gradient(circle at top left, rgba(28, 201, 178, 0.18), transparent 22%),
              linear-gradient(180deg, #0a5156 0%, #0b4148 58%, #12313d 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-testid="stSidebar"] * {
            color: #eef7f6;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #eef7f6;
        }
        .sidebar-shell {
            padding-bottom: 0.25rem;
        }
        .sidebar-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.68rem;
            font-weight: 700;
            color: rgba(235, 247, 246, 0.72);
            margin-bottom: 0.65rem;
        }
        .sidebar-app-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
        }
        .sidebar-summary {
            margin-top: 0.65rem;
            color: rgba(235, 247, 246, 0.82);
            line-height: 1.55;
            font-size: 0.95rem;
        }
        .sidebar-user-card {
            margin-top: 1rem;
            margin-bottom: 1rem;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }
        .sidebar-user-role {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(235, 247, 246, 0.68);
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .sidebar-user-name {
            font-size: 1rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.4rem;
        }
        .sidebar-user-badge {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            background: rgba(28, 201, 178, 0.18);
            border: 1px solid rgba(28, 201, 178, 0.22);
            color: #b8ffef;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .sidebar-nav-label {
            margin: 0.95rem 0 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.7rem;
            color: rgba(235, 247, 246, 0.62);
            font-weight: 700;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 2.7rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.06);
            color: #eef7f6;
            font-weight: 600;
            box-shadow: none;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(28, 201, 178, 0.36);
            background: rgba(255, 255, 255, 0.11);
            color: #ffffff;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #17a692 0%, #0d7e73 100%);
            border-color: rgba(31, 224, 196, 0.2);
            color: #ffffff;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #20bba4 0%, #10897c 100%);
            color: #ffffff;
        }
        .hero {
            padding: 1.5rem 1.6rem;
            border-radius: 24px;
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(17, 42, 53, 0.08);
            box-shadow: 0 18px 40px rgba(17, 42, 53, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0.2rem 0 0.35rem;
            font-size: 2.2rem;
            color: #18232b;
        }
        .hero p {
            margin: 0;
            color: #64727a;
            line-height: 1.55;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            font-weight: 700;
            color: #c95d3d;
        }
        .login-card {
            padding: 1.75rem;
            border-radius: 24px;
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(17, 42, 53, 0.08);
            box-shadow: 0 18px 40px rgba(17, 42, 53, 0.08);
        }
        .login-shell {
            padding-top: 0.2rem;
        }
        .login-brand-panel {
            min-height: calc(100vh - 3.45rem);
            border-radius: 0;
            padding: 2.45rem 2.25rem 2.15rem;
            background:
              radial-gradient(circle at top right, rgba(27, 201, 178, 0.18), transparent 24%),
              linear-gradient(180deg, #0a5a5e 0%, #0d5558 50%, #12313d 100%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
            color: #edf8f6;
            display: flex;
            justify-content: center;
        }
        .login-brand-inner {
            width: min(100%, 18rem);
        }
        .login-brand-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.32em;
            font-size: 0.68rem;
            font-weight: 700;
            color: rgba(236, 248, 246, 0.74);
            margin-bottom: 1.65rem;
        }
        .login-brand-title {
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
        }
        .login-brand-divider {
            width: 2.1rem;
            height: 0.16rem;
            border-radius: 999px;
            margin: 1rem 0 2.4rem;
            background: rgba(255, 255, 255, 0.65);
        }
        .login-brand-copy {
            color: rgba(236, 248, 246, 0.88);
            font-size: 0.89rem;
            line-height: 1.78;
            max-width: 15.75rem;
            margin-bottom: 2rem;
        }
        .login-brand-subeyebrow {
            text-transform: uppercase;
            letter-spacing: 0.28em;
            font-size: 0.68rem;
            font-weight: 700;
            color: rgba(236, 248, 246, 0.74);
            margin-bottom: 1.05rem;
        }
        .login-brand-highlight {
            color: #ffffff;
            font-size: 1.1rem;
            line-height: 1.42;
            font-weight: 800;
            max-width: 11.2rem;
            margin-bottom: 1.25rem;
        }
        .login-feature-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.9rem;
            margin-top: 2rem;
            max-width: 100%;
        }
        .login-feature-card {
            padding: 0.95rem 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .login-feature-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            color: #d3fff5;
            font-size: 0.84rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }
        .login-feature-title {
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .login-feature-copy {
            color: rgba(236, 248, 246, 0.74);
            font-size: 0.79rem;
            line-height: 1.55;
        }
        .login-form-frame {
            max-width: 25.5rem;
            padding-top: 0.45rem;
        }
        .login-form-anchor {
            display: none;
        }
        div[data-testid="stForm"]:has(.login-form-anchor) {
            max-width: 25.5rem;
            padding: 1.2rem 1.2rem 1rem;
            border-radius: 24px;
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(17, 42, 53, 0.08);
            box-shadow: 0 18px 40px rgba(17, 42, 53, 0.08);
        }
        div[data-testid="stForm"]:has(.login-form-anchor) form {
            gap: 0;
        }
        .login-form-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.95rem;
            margin-bottom: 0.95rem;
        }
        .login-form-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .login-form-brand-logo {
            width: 2.9rem;
            height: 2.9rem;
            object-fit: cover;
            border-radius: 12px;
            display: block;
            flex-shrink: 0;
        }
        .login-form-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.68rem;
            font-weight: 700;
            color: #6d7b85;
            margin-bottom: 0.25rem;
        }
        .login-form-title {
            color: #18232b;
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
        }
        .login-form-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(22, 166, 146, 0.08);
            color: #0c766f;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid rgba(22, 166, 146, 0.12);
            white-space: nowrap;
        }
        .login-form-copy {
            color: #6b7780;
            font-size: 0.84rem;
            line-height: 1.7;
            max-width: 21rem;
            margin-bottom: 0.95rem;
        }
        .login-footnote {
            color: #7e888e;
            font-size: 0.78rem;
            line-height: 1.55;
            margin-top: 0.65rem;
            max-width: 25.5rem;
        }
        .login-footnote code {
            color: #0f6b68;
            background: rgba(15, 107, 104, 0.08);
            border-radius: 6px;
            padding: 0.08rem 0.32rem;
        }
        .sidebar-note {
            color: rgba(255,255,255,0.8);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 0.9rem;
        }
        div[data-testid="stForm"]:has(.login-form-anchor) .stTextInput label {
            color: #31424d;
            font-weight: 600;
            font-size: 0.72rem;
        }
        div[data-testid="stForm"]:has(.login-form-anchor) .stTextInput input {
            background: #ffffff;
            border: 1px solid rgba(17, 42, 53, 0.12);
            border-radius: 10px;
        }
        div[data-testid="stForm"]:has(.login-form-anchor) .stFormSubmitButton > button {
            width: 100%;
            min-height: 2.45rem;
            border-radius: 10px;
            background: #ffffff;
            color: #27404b;
            border: 1px solid rgba(17, 42, 53, 0.12);
            font-weight: 700;
            box-shadow: none;
        }
        div[data-testid="stForm"]:has(.login-form-anchor) .stFormSubmitButton > button:hover {
            background: #f8fbfd;
            color: #18232b;
        }
        @media (max-width: 1100px) {
            .login-brand-panel {
                min-height: auto;
                padding-bottom: 2.2rem;
            }
            .login-brand-inner,
            div[data-testid="stForm"]:has(.login-form-anchor),
            .login-footnote {
                max-width: 100%;
            }
        }
        .auth-left {
            position: relative;
            padding: 3rem;
            background: linear-gradient(135deg, #004c4c 0%, #006666 100%);
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
            min-height: 43rem;
        }
        .auth-left::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
              radial-gradient(circle at top right, rgba(162, 240, 239, 0.18), transparent 28%),
              radial-gradient(circle at bottom left, rgba(255,255,255,0.12), transparent 24%);
            pointer-events: none;
        }
        .auth-left-inner,
        .auth-left-footer,
        .auth-right-inner {
            position: relative;
            z-index: 1;
        }
        .auth-left-title {
            margin: 0 0 0.4rem;
            font-size: 2.6rem;
            font-weight: 800;
            color: white;
            line-height: 1.05;
        }
        .auth-left-rule {
            width: 56px;
            height: 4px;
            border-radius: 999px;
            background: #a2f0ef;
        }
        .auth-kicker {
            margin: 0 0 1rem;
            font-size: 0.74rem;
            letter-spacing: 0.35em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(203, 231, 245, 0.7);
        }
        .auth-headline {
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.08;
            color: white;
            max-width: 14ch;
        }
        .auth-copy {
            margin: 1.25rem 0 0;
            max-width: 34rem;
            font-size: 1.05rem;
            line-height: 1.75;
            color: rgba(203, 231, 245, 0.82);
        }
        .auth-feature-grid {
            margin-top: 2rem;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }
        .auth-feature {
            padding: 1.25rem;
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(10px);
        }
        .auth-feature-icon {
            width: 48px;
            height: 48px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }
        .auth-feature h3 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: white;
        }
        .auth-feature p {
            margin: 0.55rem 0 0;
            font-size: 0.92rem;
            line-height: 1.6;
            color: rgba(203, 231, 245, 0.74);
        }
        .auth-left-footer {
            margin-top: 2rem;
            font-size: 0.75rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: rgba(203, 231, 245, 0.44);
        }
        .auth-right {
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 2rem 0 0;
        }
        .auth-right-inner {
            width: 100%;
            max-width: 27rem;
        }
        .auth-header-card {
            width: 100%;
            padding: 1.35rem 1.45rem 1.2rem;
            border-radius: 24px;
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(17, 42, 53, 0.08);
            box-shadow: 0 18px 40px rgba(17, 42, 53, 0.08);
            box-sizing: border-box;
            margin-bottom: 1rem;
        }
        .auth-brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        .auth-brand-main {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .auth-logo-tile {
            width: 64px;
            height: 64px;
            border-radius: 18px;
            background: #006666;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 24px 48px -12px rgba(25, 28, 29, 0.08);
        }
        .auth-logo-tile img {
            width: 40px;
            height: 40px;
            object-fit: contain;
            display: block;
        }
        .auth-small-tag {
            font-size: 0.74rem;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            font-weight: 700;
            color: #004c4c;
        }
        .auth-right-title {
            margin: 0.2rem 0 0;
            font-size: 2rem;
            font-weight: 800;
            color: #191c1d;
            line-height: 1.08;
        }
        .auth-back-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.65rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(0, 76, 76, 0.1);
            background: white;
            color: #004c4c;
            font-size: 0.9rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .auth-form-copy {
            margin: 0 0 1.5rem;
            font-size: 0.96rem;
            line-height: 1.7;
            color: #3f4948;
        }
        .auth-form-shell {
            width: 100%;
        }
        .auth-form-anchor {
            display: none;
        }
        div[data-testid="stForm"]:has(.auth-form-anchor) {
            width: 100%;
            max-width: 27rem;
            padding: 1rem 1rem 0.4rem;
            border-radius: 18px;
            background: rgba(247, 238, 226, 0.7);
            border: 1px solid rgba(17, 42, 53, 0.08);
            box-shadow: none;
            box-sizing: border-box;
        }
        div[data-testid="stForm"]:has(.auth-form-anchor) form {
            gap: 0;
        }
        div[data-testid="stForm"]:has(.auth-form-anchor) .stTextInput label {
            color: #31424d;
            font-weight: 600;
            font-size: 0.72rem;
        }
        div[data-testid="stForm"]:has(.auth-form-anchor) .stTextInput input {
            background: #ffffff;
            border: 1px solid rgba(17, 42, 53, 0.12);
            border-radius: 10px;
        }
        div[data-testid="stForm"]:has(.auth-form-anchor) .stFormSubmitButton > button {
            width: 100%;
            min-height: 2.55rem;
            border-radius: 10px;
            background: #ffffff;
            color: #31424d;
            border: 1px solid rgba(17, 42, 53, 0.12);
            box-shadow: none;
            font-weight: 700;
        }
        div[data-testid="stForm"]:has(.auth-form-anchor) .stFormSubmitButton > button:hover {
            background: #f9fbfc;
            color: #191c1d;
        }
        .auth-footnote {
            margin-top: 0.75rem;
            font-size: 0.78rem;
            line-height: 1.55;
            color: #7e888e;
            max-width: 27rem;
        }
        .auth-footnote code {
            color: #0f6b68;
            background: rgba(15, 107, 104, 0.08);
            border-radius: 6px;
            padding: 0.08rem 0.32rem;
        }
        @media (max-width: 900px) {
            .auth-left {
                min-height: auto;
                padding: 2rem;
            }
            .auth-headline {
                font-size: 2.2rem;
                max-width: none;
            }
            .auth-feature-grid {
                grid-template-columns: 1fr;
            }
            .auth-right {
                padding-top: 1.25rem;
            }
            .auth-right-inner,
            div[data-testid="stForm"]:has(.auth-form-anchor),
            .auth-footnote {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, eyebrow: str, subtitle: str) -> None:
    logo_html = ""
    if LOGO_PATH.exists():
        import base64

        logo_html = f'<img src="data:image/png;base64,{base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")}" alt="logo" style="height:54px;border-radius:12px;margin-bottom:0.85rem;" />'
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            {logo_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_face_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource
def load_recognizer() -> FaceRecognizer:
    return FaceRecognizer()


@st.cache_resource
def load_liveness_detector() -> LivenessDetector:
    return LivenessDetector()


def decode_camera_value(camera_value) -> object | None:
    if camera_value is None:
        return None
    return decode_uploaded_image(camera_value.getvalue())


def set_authenticated_user(user: dict | None) -> None:
    st.session_state["auth_user"] = user
    st.session_state["auth_seen_at"] = datetime.now().isoformat(timespec="seconds")


def authenticated_user() -> dict | None:
    user = st.session_state.get("auth_user")
    last_seen = st.session_state.get("auth_seen_at")
    if not user or not last_seen:
        return None
    seen_at = datetime.fromisoformat(last_seen)
    if datetime.now() - seen_at > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
        set_authenticated_user(None)
        st.session_state["auth_expired"] = True
        return None
    st.session_state["auth_seen_at"] = datetime.now().isoformat(timespec="seconds")
    return user


def logout() -> None:
    set_authenticated_user(None)
    st.rerun()


def session_label(session_row: dict) -> str:
    return f"{session_row['session_date']} | {session_row['course_code']} | {session_row['section_name']} | {session_row['session_title']} | {session_row['status']}"


def offering_label(offering_row: dict) -> str:
    return f"{offering_row['course_code']} - {offering_row['course_title']} | {offering_row['program_name'] or 'Program'} {offering_row['year_label']} {offering_row['section_name']} | {offering_row['faculty_name']}"


def section_label(section_row: dict) -> str:
    program = section_row.get("program_name") or "Program"
    return f"{program} | {section_row['year_label']} | Section {section_row['name']}"


def dataframe_or_info(rows: list[dict], message: str) -> None:
    if not rows:
        st.info(message)
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_login_page() -> None:
    if st.session_state.pop("auth_expired", False):
        st.warning("Your session expired. Sign in again.")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        .block-container {
            max-width: 1320px;
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="auth-left">
              <div class="auth-left-inner">
                <h1 class="auth-left-title">My-GITAM</h1>
                <div class="auth-left-rule"></div>
                <div style="margin-top: 4rem;">
                  <p class="auth-kicker">Administration Console</p>
                  <h2 class="auth-headline">Access your SmartAttend operations workspace.</h2>
                  <p class="auth-copy">
                    Sign in once as administrator to manage student enrollment, attendance verification,
                    liveness protection, reporting, and spoof-attempt auditing from a single academic dashboard.
                  </p>
                  <div class="auth-feature-grid">
                    <div class="auth-feature">
                      <div class="auth-feature-icon">ID</div>
                      <h3>Identity First</h3>
                      <p>Admin access controls the full attendance workflow before students are processed.</p>
                    </div>
                    <div class="auth-feature">
                      <div class="auth-feature-icon">DB</div>
                      <h3>Database Ready</h3>
                      <p>Managed Postgres, audit logs, and secure object storage keep operations durable.</p>
                    </div>
                  </div>
                </div>
              </div>
              <div class="auth-left-footer">Academic Management Portal © 2026 GITAM University</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        logo_html = ""
        if LOGO_PATH.exists():
            import base64

            logo_base64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
            logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="University Logo" />'

        st.markdown(
            f"""
            <div class="auth-right">
              <div class="auth-right-inner">
                <div class="auth-header-card">
                  <div class="auth-brand-row">
                    <div class="auth-brand-main">
                      <div class="auth-logo-tile">{logo_html}</div>
                      <div>
                        <div class="auth-small-tag">G-Learn Administration</div>
                        <h2 class="auth-right-title">Admin Login</h2>
                      </div>
                    </div>
                    <div class="auth-back-pill">Secure Access</div>
                  </div>
                  <p class="auth-form-copy">
                    Sign in with the SmartAttend administrator credentials to open the dashboard.
                    This replaces the public landing flow and keeps the portal locked until an admin session starts.
                  </p>
                </div>
                <div class="auth-form-shell">
            """,
            unsafe_allow_html=True,
        )

        with st.form("admin_login_form", clear_on_submit=False, border=False):
            st.markdown(
                '<div class="auth-form-anchor"></div>',
                unsafe_allow_html=True,
            )
            username = st.text_input("Admin Username", value=DEFAULT_ADMIN_USERNAME, key="login_username")
            password = st.text_input("Admin Password", type="password", key="login_password")
            submitted = st.form_submit_button("Access Dashboard", use_container_width=True)

        st.markdown(
            """
            <div class="auth-footnote">
              Set <code>SMARTATTEND_ADMIN_USER</code> and <code>SMARTATTEND_ADMIN_PASSWORD</code> in the environment to change the default admin credentials.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div></div></div>", unsafe_allow_html=True)
        if submitted:
            user, message = authenticate_user(username, password)
            if user is None:
                st.error(message)
            else:
                set_authenticated_user(user)
                st.success(message)
                st.rerun()


# PAGE_RENDERERS


def render_dashboard(user: dict) -> None:
    overview = attendance_overview(user=user)
    role_copy = "faculty-owned operations" if user["role"] == "faculty" else "institution-wide operations"
    render_page_header(
        title="Operations Dashboard",
        eyebrow=user["role"].title(),
        subtitle=f"Monitor {role_copy}, open attendance sessions, verification activity, and review load from one place.",
    )

    metrics = st.columns(6)
    metrics[0].metric("Students", overview["total_students"])
    metrics[1].metric("Open Sessions", overview["open_sessions"])
    metrics[2].metric("Present Today", overview["present_today"])
    metrics[3].metric("Absent Today", overview["absent_today"])
    metrics[4].metric("Spoof Attempts", overview["spoof_attempts_today"])
    metrics[5].metric("Open Exceptions", overview["open_exceptions"])

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Recent Attendance")
        dataframe_or_info(list_recent_attendance(limit=12, user=user), "No attendance records yet.")
        st.subheader("Roster Completion")
        students = list_students(user=user)
        if students:
            chart = pd.DataFrame(students)[["roll_no", "attendance_percentage"]].set_index("roll_no")
            st.bar_chart(chart, use_container_width=True)
        else:
            st.info("No students are enrolled in the visible scope yet.")
    with right:
        st.subheader("Recent Verification Attempts")
        dataframe_or_info(list_recent_attempts(limit=12, user=user), "No attempts logged yet.")
        st.subheader("Open Exceptions")
        dataframe_or_info(list_exceptions(user=user, status="open", limit=12), "No open exceptions.")


def render_user_management(user: dict) -> None:
    render_page_header(
        title="User and Faculty Management",
        eyebrow="Admin",
        subtitle="Create operator accounts, assign faculty ownership, and keep role boundaries explicit.",
    )
    departments = list_departments()
    roles = ["faculty", "admin"]
    with st.form("create_user_form"):
        left, right = st.columns(2)
        full_name = left.text_input("Full Name")
        username = right.text_input("Username")
        email = left.text_input("Email")
        password = right.text_input("Temporary Password")
        role = left.selectbox("Role", roles)
        title = right.text_input("Faculty Title", placeholder="Assistant Professor")
        department_id = left.selectbox(
            "Department",
            options=[0] + [department["id"] for department in departments],
            format_func=lambda value: "Unassigned" if value == 0 else next((department["name"] for department in departments if department["id"] == value), "Department"),
        )
        submitted = st.form_submit_button("Create Account", type="primary")
    if submitted:
        try:
            created = create_user(username=username, full_name=full_name, email=email, password=password, role=role)
            if role == "faculty":
                create_faculty_profile(
                    user_id=created["id"],
                    department_id=None if department_id == 0 else department_id,
                    title=title,
                )
            st.success(f"Created {role} account for {created['full_name']}.")
        except Exception as error:
            st.error(str(error))

    st.subheader("Faculty Accounts")
    dataframe_or_info(list_faculty_users(), "No faculty accounts yet.")
    st.subheader("All User Accounts")
    dataframe_or_info(list_users(), "No user accounts found.")


def render_academic_setup(user: dict) -> None:
    render_page_header(
        title="Academic Setup",
        eyebrow="Admin",
        subtitle="Model the institution: departments, programs, sections, courses, and faculty-owned offerings.",
    )

    departments = list_departments()
    programs = list_programs()
    sections = list_sections()
    courses = list_courses()
    faculty = list_faculty_users()

    dep_col, prog_col = st.columns(2)
    with dep_col.form("department_form"):
        st.markdown("#### Add Department")
        name = st.text_input("Department Name")
        code = st.text_input("Department Code", placeholder="CSE")
        if st.form_submit_button("Create Department"):
            try:
                create_department(name, code)
                st.success("Department created.")
            except Exception as error:
                st.error(str(error))

    with prog_col.form("program_form"):
        st.markdown("#### Add Program")
        name = st.text_input("Program Name", placeholder="B. Tech CSE")
        code = st.text_input("Program Code", placeholder="BTECH-CSE")
        department_id = st.selectbox(
            "Department",
            options=[0] + [department["id"] for department in departments],
            format_func=lambda value: "Unassigned" if value == 0 else next((department["name"] for department in departments if department["id"] == value), "Department"),
        )
        if st.form_submit_button("Create Program"):
            try:
                create_program(department_id=None if department_id == 0 else department_id, name=name, code=code)
                st.success("Program created.")
            except Exception as error:
                st.error(str(error))

    sec_col, course_col = st.columns(2)
    with sec_col.form("section_form"):
        st.markdown("#### Add Section")
        program_id = st.selectbox(
            "Program",
            options=[0] + [program["id"] for program in programs],
            format_func=lambda value: "Unassigned" if value == 0 else next((program["name"] for program in programs if program["id"] == value), "Program"),
        )
        year_label = st.text_input("Year Label", placeholder="3rd Year")
        section_name = st.text_input("Section Name", placeholder="A")
        semester_label = st.text_input("Semester Label", placeholder="Semester 6")
        if st.form_submit_button("Create Section"):
            try:
                create_section(program_id=None if program_id == 0 else program_id, name=section_name, year_label=year_label, semester_label=semester_label)
                st.success("Section created.")
            except Exception as error:
                st.error(str(error))

    with course_col.form("course_form"):
        st.markdown("#### Add Course")
        department_id = st.selectbox(
            "Course Department",
            options=[0] + [department["id"] for department in departments],
            format_func=lambda value: "Unassigned" if value == 0 else next((department["name"] for department in departments if department["id"] == value), "Department"),
            key="course_department_id",
        )
        course_code = st.text_input("Course Code", placeholder="CSE301")
        title = st.text_input("Course Title", placeholder="Operating Systems")
        credits = st.number_input("Credit Hours", min_value=1, max_value=8, value=3)
        if st.form_submit_button("Create Course"):
            try:
                create_course(department_id=None if department_id == 0 else department_id, course_code=course_code, title=title, credit_hours=int(credits))
                st.success("Course created.")
            except Exception as error:
                st.error(str(error))

    st.markdown("---")
    if not courses or not sections or not faculty:
        st.info("Create at least one course, section, and faculty account before creating course offerings.")
    else:
        with st.form("offering_form"):
            st.markdown("#### Create Course Offering")
            course_id = st.selectbox(
                "Course",
                options=[course["id"] for course in courses],
                format_func=lambda value: next((f"{course['course_code']} - {course['title']}" for course in courses if course["id"] == value), "Course"),
            )
            section_id = st.selectbox(
                "Section",
                options=[section["id"] for section in sections],
                format_func=lambda value: next((section_label(section) for section in sections if section["id"] == value), "Section"),
            )
            faculty_user_id = st.selectbox(
                "Faculty Owner",
                options=[person["id"] for person in faculty],
                format_func=lambda value: next((person["full_name"] for person in faculty if person["id"] == value), "Faculty"),
            )
            term_name = st.text_input("Term", placeholder="Monsoon")
            academic_year = st.text_input("Academic Year", placeholder="2026-2027")
            if st.form_submit_button("Create Offering", type="primary"):
                try:
                    create_course_offering(course_id=course_id, section_id=section_id, faculty_user_id=faculty_user_id, term_name=term_name, academic_year=academic_year)
                    st.success("Course offering created.")
                except Exception as error:
                    st.error(str(error))

    st.subheader("Departments")
    dataframe_or_info(departments, "No departments created yet.")
    st.subheader("Programs")
    dataframe_or_info(programs, "No programs created yet.")
    st.subheader("Sections")
    dataframe_or_info(sections, "No sections created yet.")
    st.subheader("Courses")
    dataframe_or_info(courses, "No courses created yet.")
    st.subheader("Course Offerings")
    dataframe_or_info(list_course_offerings(), "No offerings created yet.")


def render_students(user: dict) -> None:
    render_page_header(
        title="Student Enrollment and Roster",
        eyebrow="Roster",
        subtitle="Enroll students into real academic sections so attendance is tied to course offerings and not just a face scan.",
    )

    sections = list_sections()
    programs = {program["id"]: program for program in list_programs()}
    section_id = st.selectbox(
        "Section",
        options=[0] + [section["id"] for section in sections],
        format_func=lambda value: "Select a section" if value == 0 else next((section_label(section) for section in sections if section["id"] == value), "Section"),
    )
    selected_section = next((section for section in sections if section["id"] == section_id), None)
    program_name = programs.get(selected_section["program_id"], {}).get("name", "") if selected_section else ""
    year_label = selected_section["year_label"] if selected_section else ""

    with st.form("student_enrollment_form"):
        left, right = st.columns(2)
        first_name = left.text_input("First Name")
        last_name = right.text_input("Last Name")
        roll_no = left.text_input("Roll Number")
        email = right.text_input("Email")
        program = left.text_input("Program", value=program_name)
        course = right.text_input("Branch / Course", placeholder="CSE")
        year = left.text_input("Year", value=year_label)
        st.caption("Capture a face image after filling the academic record.")
        camera_value = st.camera_input("Enrollment Face Capture")
        submitted = st.form_submit_button("Enroll Student", type="primary")

    if submitted:
        capture_bgr = decode_camera_value(camera_value)
        if section_id == 0:
            st.error("Select a section before enrolling the student.")
        elif capture_bgr is None:
            st.error("Capture a face image before submitting enrollment.")
        else:
            try:
                result = enroll_student(
                    first_name=first_name,
                    last_name=last_name,
                    roll_no=roll_no,
                    email=email,
                    year=year,
                    program=program,
                    course=course,
                    section_id=section_id,
                    capture_bgr=capture_bgr,
                    detector=load_face_detector(),
                    liveness_detector=load_liveness_detector(),
                    created_by_user_id=user["id"],
                    storage_backend=get_storage_backend(),
                )
                if result.success:
                    load_recognizer.clear()
                    st.success(result.message)
                else:
                    st.error(result.message)
            except Exception as error:
                st.error(str(error))

    st.subheader("Roster")
    offerings = list_course_offerings(user=user)
    filter_offering_id = st.selectbox(
        "Filter by Offering",
        options=[0] + [offering["id"] for offering in offerings],
        format_func=lambda value: "All visible sections" if value == 0 else next((offering_label(offering) for offering in offerings if offering["id"] == value), "Offering"),
        key="student_offering_filter",
    )
    roster = list_students(user=user, offering_id=None if filter_offering_id == 0 else filter_offering_id)
    dataframe_or_info(roster, "No students in the current scope yet.")


def render_sessions(user: dict) -> None:
    render_page_header(
        title="Session Management",
        eyebrow="Attendance Windows",
        subtitle="Create course sessions, open and close attendance windows, and finalize absences at the session boundary.",
    )
    offerings = list_course_offerings(user=user, active_only=True)
    if not offerings:
        st.warning("No active offerings are available in your scope. Create offerings before creating sessions.")
    else:
        with st.form("create_session_form"):
            offering_id = st.selectbox(
                "Course Offering",
                options=[offering["id"] for offering in offerings],
                format_func=lambda value: next((offering_label(offering) for offering in offerings if offering["id"] == value), "Offering"),
            )
            session_title = st.text_input("Session Title", placeholder="Week 5 Lecture")
            left, right = st.columns(2)
            session_date = left.date_input("Session Date")
            location = right.text_input("Location", placeholder="Room C-204")
            start_time = left.time_input("Start Time")
            end_time = right.time_input("End Time")
            open_at = left.time_input("Attendance Opens")
            close_at = right.time_input("Attendance Closes")
            notes = st.text_area("Notes", height=90)
            submitted = st.form_submit_button("Create Session", type="primary")
        if submitted:
            try:
                create_class_session(
                    offering_id=offering_id,
                    session_title=session_title,
                    session_date=session_date.isoformat(),
                    start_time=start_time.strftime("%H:%M:%S"),
                    end_time=end_time.strftime("%H:%M:%S"),
                    attendance_open_at=f"{session_date.isoformat()}T{open_at.strftime('%H:%M:%S')}",
                    attendance_close_at=f"{session_date.isoformat()}T{close_at.strftime('%H:%M:%S')}",
                    location=location,
                    notes=notes,
                    created_by_user_id=user["id"],
                )
                st.success("Class session created.")
            except Exception as error:
                st.error(str(error))

    sessions = list_class_sessions(user=user, limit=300)
    st.subheader("Session Operations")
    if sessions:
        target_session_id = st.selectbox(
            "Select Session",
            options=[session["id"] for session in sessions],
            format_func=lambda value: next((session_label(session) for session in sessions if session["id"] == value), "Session"),
        )
        action = st.selectbox("Action", ["open", "scheduled", "closed", "finalize_absences"])
        if st.button("Apply Session Action"):
            try:
                if action == "finalize_absences":
                    created = finalize_session_absences(target_session_id, user["id"])
                    st.success(f"Finalized session and created {created} absent records.")
                else:
                    update_class_session_status(target_session_id, action, user["id"])
                    st.success(f"Session status updated to {action}.")
            except Exception as error:
                st.error(str(error))
    dataframe_or_info(sessions, "No class sessions created yet.")


def render_attendance(user: dict) -> None:
    render_page_header(
        title="Live Attendance Verification",
        eyebrow="Session Check-In",
        subtitle="Verify attendance only against open class sessions, with liveness gates, rate limits, and exception creation on failure.",
    )
    open_sessions = list_class_sessions(user=user, status="open", limit=100)
    if not open_sessions:
        st.warning("No open class sessions are available. Open a session first from Session Management.")
        return

    session_id = st.selectbox(
        "Open Session",
        options=[session["id"] for session in open_sessions],
        format_func=lambda value: next((session_label(session) for session in open_sessions if session["id"] == value), "Session"),
    )
    claimed_roll_no = st.text_input("Claimed Roll Number")
    camera_value = st.camera_input("Attendance Face Capture", key="attendance_camera")
    if st.button("Verify Attendance", type="primary"):
        capture_bgr = decode_camera_value(camera_value)
        if capture_bgr is None:
            st.error("Capture a face image before submitting attendance.")
            return
        if not claimed_roll_no.strip():
            st.error("Claimed roll number is required.")
            return

        decision = verify_attendance_attempt(
            session_id=session_id,
            claimed_roll_no=claimed_roll_no,
            capture_bgr=capture_bgr,
            actor_user_id=user["id"],
            actor_role=user["role"],
            detector=load_face_detector(),
            recognizer=load_recognizer(),
            liveness_detector=load_liveness_detector(),
        )
        if decision.success:
            st.success(decision.message)
        else:
            st.error(decision.message)

        if decision.student:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "student": f"{decision.student['first_name']} {decision.student['last_name']}",
                            "roll_no": decision.student["roll_no"],
                            "status": decision.status,
                            "confidence": round(decision.confidence, 4),
                            "liveness_score": round(decision.liveness_score, 4),
                            "attempt_outcome": decision.attempt_outcome,
                            "exception_id": decision.exception_id,
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        if decision.predicted_student:
            st.caption(f"Predicted identity: {decision.predicted_student['first_name']} {decision.predicted_student['last_name']} ({decision.predicted_student['roll_no']})")


def render_exceptions(user: dict) -> None:
    render_page_header(
        title="Exception Review Queue",
        eyebrow="Faculty Review",
        subtitle="Review spoof alerts, identity mismatches, and failed verifications before approving or rejecting manual attendance outcomes.",
    )
    open_exceptions = list_exceptions(user=user, status="open", limit=200)
    dataframe_or_info(open_exceptions, "No open exceptions to review.")
    if not open_exceptions:
        return

    exception_id = st.selectbox(
        "Select Exception",
        options=[row["id"] for row in open_exceptions],
        format_func=lambda value: next((f"#{row['id']} | {row.get('roll_no','Unknown')} | {row['reason']} | {row['course_code']}" for row in open_exceptions if row["id"] == value), "Exception"),
    )
    resolution = st.selectbox("Resolution", ["approved_present", "approved_excused", "rejected"])
    note = st.text_area("Review Note", height=100)
    if st.button("Resolve Exception", type="primary"):
        try:
            resolved_status = None
            if resolution == "approved_present":
                resolved_status = "Present"
            elif resolution == "approved_excused":
                resolved_status = "Excused"
            resolve_exception(
                exception_id=exception_id,
                reviewer_user_id=user["id"],
                resolution=resolution,
                resolution_note=note,
                resolved_attendance_status=resolved_status,
            )
            st.success("Exception resolved.")
        except Exception as error:
            st.error(str(error))


def render_reports(user: dict) -> None:
    render_page_header(
        title="Reports and Analytics",
        eyebrow="Operational Reporting",
        subtitle="Export attendance by session or offering, inspect evaluation outputs, and monitor model registry state.",
    )
    offerings = list_course_offerings(user=user)
    sessions = list_class_sessions(user=user, limit=300)
    offering_id = st.selectbox(
        "Offering Filter",
        options=[0] + [offering["id"] for offering in offerings],
        format_func=lambda value: "All offerings" if value == 0 else next((offering_label(offering) for offering in offerings if offering["id"] == value), "Offering"),
    )
    session_id = st.selectbox(
        "Session Filter",
        options=[0] + [session["id"] for session in sessions],
        format_func=lambda value: "All sessions" if value == 0 else next((session_label(session) for session in sessions if session["id"] == value), "Session"),
    )

    records = list_attendance_records(
        user=user,
        offering_id=None if offering_id == 0 else offering_id,
        session_id=None if session_id == 0 else session_id,
    )
    dataframe_or_info(records, "No attendance records in the selected scope.")
    if records:
        csv_bytes = pd.DataFrame(records).to_csv(index=False).encode("utf-8")
        st.download_button("Download Attendance CSV", data=csv_bytes, file_name="attendance_records.csv", mime="text/csv")

    exceptions = export_exception_csv(user=user)
    if exceptions:
        exception_csv = pd.DataFrame(exceptions).to_csv(index=False).encode("utf-8")
        st.download_button("Download Exception CSV", data=exception_csv, file_name="attendance_exceptions.csv", mime="text/csv")

    st.subheader("Evaluation Reports")
    face_report = get_evaluation_report("face_model")
    liveness_report = get_evaluation_report("liveness_model")
    left, right = st.columns(2)
    with left:
        if face_report:
            st.json(face_report)
        else:
            st.info("No face evaluation report saved yet.")
    with right:
        if liveness_report:
            st.json(liveness_report)
        else:
            st.info("No liveness evaluation report saved yet.")

    st.subheader("Model Registry")
    dataframe_or_info(list_model_versions(), "No model versions registered yet.")


def render_liveness_ops(user: dict) -> None:
    render_page_header(
        title="Liveness Operations",
        eyebrow="Model Ops",
        subtitle="Collect anti-spoof samples, train the liveness classifier, and monitor deployment readiness for the attendance gate.",
    )
    counts = liveness_counts()
    cols = st.columns(3)
    cols[0].metric("Real Samples", counts["real"])
    cols[1].metric("Fake Samples", counts["fake"])
    cols[2].metric("Model Loaded", "Yes" if load_liveness_detector().available else "No")

    left, right = st.columns(2)
    with left:
        real_capture = st.camera_input("Capture Real Sample", key="liveness_real")
        if st.button("Save Real Sample"):
            image = decode_camera_value(real_capture)
            if image is None:
                st.error("Capture a live face first.")
            else:
                result = save_liveness_sample(image, label="real", detector=load_face_detector(), source_prefix="real")
                if result.success:
                    st.success(result.message)
                else:
                    st.error(result.message)
    with right:
        fake_capture = st.camera_input("Capture Fake Sample", key="liveness_fake")
        if st.button("Save Fake Sample"):
            image = decode_camera_value(fake_capture)
            if image is None:
                st.error("Capture a spoof sample first.")
            else:
                result = save_liveness_sample(image, label="fake", detector=load_face_detector(), source_prefix="fake")
                if result.success:
                    st.success(result.message)
                else:
                    st.error(result.message)

    ready, current_counts = dataset_ready_for_training()
    st.caption(f"Training readiness: real={current_counts['real']} fake={current_counts['fake']}")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15, step=1)
    batch_size = st.number_input("Batch Size", min_value=2, max_value=64, value=8, step=2)
    validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    if not ready:
        st.warning("Collect more balanced real and fake samples before training.")
    if st.button("Train Liveness Model", disabled=not ready):
        try:
            result = train_liveness_model(epochs=int(epochs), batch_size=int(batch_size), validation_split=float(validation_split))
            load_liveness_detector.clear()
            st.success("Liveness model trained.")
            st.json(result)
        except Exception as error:
            st.error(str(error))
    if st.button("Run Evaluation Suite"):
        try:
            results = run_all_evaluations()
            st.success("Evaluation completed.")
            st.json(results)
        except Exception as error:
            st.error(str(error))


def render_security_ops(user: dict) -> None:
    render_page_header(
        title="Security and Observability",
        eyebrow="Admin",
        subtitle="Inspect audit logs, login pressure, model inventory, and deployment posture across the attendance platform.",
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Login Attempts")
        dataframe_or_info(list_login_attempts(limit=200), "No login attempts logged yet.")
        st.subheader("Audit Trail")
        dataframe_or_info(list_audit_logs(limit=200), "No audit events recorded yet.")
    with right:
        st.subheader("Model Health")
        rows = [
            {"component": "Database", "status": "ready", "details": str(DATABASE_PATH)},
            {"component": "Storage Backend", "status": STORAGE_BACKEND, "details": "Object storage mirror for enrolled faces"},
            {"component": "Face Detector", "status": load_face_detector().backend, "details": "MTCNN when available, Haar fallback otherwise"},
            {"component": "Face Model", "status": "ready" if FACE_MODEL_PATH.exists() else "missing", "details": FACE_MODEL_PATH.name},
            {"component": "Face Labels", "status": "ready" if FACE_LABELS_PATH.exists() else "missing", "details": FACE_LABELS_PATH.name},
            {"component": "Liveness Model", "status": "ready" if LIVENESS_MODEL_PATH.exists() else "missing", "details": LIVENESS_MODEL_PATH.name},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("Registered Model Versions")
        dataframe_or_info(list_model_versions(), "No model metadata registered yet.")
        st.caption(f"Artifacts directory: {ARTIFACTS_DIR}")


def sidebar_pages_for_role(role: str) -> list[str]:
    base_pages = ["Dashboard", "Students", "Sessions", "Attendance", "Exceptions", "Reports"]
    if role == "admin":
        return ["Dashboard", "Users", "Academic Setup", "Students", "Sessions", "Attendance", "Exceptions", "Reports", "Liveness Ops", "Security"]
    return base_pages


def current_page_for_role(role: str) -> str:
    allowed = sidebar_pages_for_role(role)
    current = st.session_state.get("current_page")
    if current not in allowed:
        current = allowed[0]
        st.session_state["current_page"] = current
    return current


def render_sidebar_navigation(user: dict) -> str:
    current = current_page_for_role(user["role"])
    workspace_copy = (
        "Administrator access keeps enrollment, sessions, verification review, and exports in one controlled workspace."
        if user["role"] == "admin"
        else "Faculty access keeps session control, attendance review, and reporting within owned course operations."
    )
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-shell">
              <div class="sidebar-eyebrow">G-Learn Style</div>
              <div class="sidebar-app-title">{APP_TITLE}</div>
              <div class="sidebar-summary">Academic attendance workspace with enrollment, liveness verification, and secure reporting.</div>
              <div class="sidebar-user-card">
                <div class="sidebar-user-role">Platform {user['role'].title()}</div>
                <div class="sidebar-user-name">{user['full_name']}</div>
                <div class="sidebar-user-badge">{user['username']}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=74)
        st.markdown('<div class="sidebar-nav-label">Navigate</div>', unsafe_allow_html=True)
        for candidate in sidebar_pages_for_role(user["role"]):
            button_type = "primary" if candidate == current else "secondary"
            if st.button(candidate, key=f"nav_{candidate}", type=button_type, use_container_width=True):
                st.session_state["current_page"] = candidate
                current = candidate
        st.markdown(
            f'<div class="sidebar-note">{workspace_copy}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        if st.button("Reload Models", key="reload_models_sidebar", use_container_width=True):
            load_face_detector.clear()
            load_recognizer.clear()
            load_liveness_detector.clear()
            st.success("Model caches cleared.")
        if st.button("Logout", key="logout_sidebar", use_container_width=True):
            logout()
    return current


def main() -> None:
    inject_styles()
    user = authenticated_user()
    if user is None:
        render_login_page()
        return

    page = render_sidebar_navigation(user)

    if page == "Dashboard":
        render_dashboard(user)
    elif page == "Users":
        render_user_management(user)
    elif page == "Academic Setup":
        render_academic_setup(user)
    elif page == "Students":
        render_students(user)
    elif page == "Sessions":
        render_sessions(user)
    elif page == "Attendance":
        render_attendance(user)
    elif page == "Exceptions":
        render_exceptions(user)
    elif page == "Reports":
        render_reports(user)
    elif page == "Liveness Ops":
        render_liveness_ops(user)
    elif page == "Security":
        render_security_ops(user)


if __name__ == "__main__":
    main()
