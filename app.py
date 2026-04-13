from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.attendance_service import verify_attendance_attempt
from src.config import APP_TITLE, ARTIFACTS_DIR, DATABASE_PATH, FACE_LABELS_PATH, FACE_MODEL_PATH, LIVENESS_MODEL_PATH
from src.database import DEFAULT_ADMIN_USERNAME, attendance_overview, get_evaluation_report, init_database, list_recent_attendance, list_recent_attempts, list_students, verify_admin
from src.enrollment_service import enroll_student
from src.evaluate_models import run_all_evaluations
from src.face_detector import FaceDetector
from src.liveness import LivenessDetector
from src.liveness_dataset_service import dataset_ready_for_training, liveness_counts, save_liveness_sample
from src.recognizer import FaceRecognizer
from src.train_liveness_model import train_liveness_model
from src.utils import decode_uploaded_image


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_database()
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "university_logo.png"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700;800&display=swap');

        :root {
            --bg: #f5eadc;
            --panel: rgba(255, 250, 244, 0.9);
            --panel-strong: #fffdf8;
            --sidebar: #112a35;
            --text: #18232b;
            --muted: #64727a;
            --accent: #c95d3d;
            --line: rgba(17, 42, 53, 0.12);
            --success: #1e7c5c;
            --warning: #b96d13;
            --danger: #a53e2e;
            --shadow: 0 20px 45px rgba(17, 42, 53, 0.12);
            --radius-lg: 28px;
            --radius-md: 20px;
            --radius-sm: 14px;
        }

        html, body, [class*="css"]  {
            font-family: "Inter", "Trebuchet MS", "Segoe UI", sans-serif;
            color: var(--text);
        }

        .stApp {
            background:
              radial-gradient(circle at top left, rgba(201, 93, 61, 0.18), transparent 30%),
              radial-gradient(circle at bottom right, rgba(17, 42, 53, 0.18), transparent 28%),
              linear-gradient(135deg, #f8efe5 0%, #f3e5d4 100%);
        }

        h1, h2, h3, .hero-title, .page-title, .sidebar-title {
            font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background:
              linear-gradient(180deg, rgba(17, 42, 53, 0.98), rgba(20, 52, 64, 0.96)),
              var(--sidebar);
            border-right: 0;
        }

        [data-testid="stSidebar"] * {
            color: #f4efe8;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption {
            color: rgba(244, 239, 232, 0.82);
        }

        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.55rem;
        }

        [data-testid="stSidebar"] .stRadio label {
            background: rgba(255, 255, 255, 0.06);
            padding: 0.85rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            font-weight: 700;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255, 255, 255, 0.12);
        }

        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 0;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin-bottom: 1.25rem;
        }

        .brand-copy {
            display: grid;
            gap: 0.18rem;
        }

        .eyebrow {
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--accent);
        }

        .sidebar-eyebrow {
            color: #f0b09d;
        }

        .sidebar-title {
            margin: 0;
            font-size: 1.5rem;
            color: #f4efe8;
        }

        .sidebar-copy {
            font-size: 0.95rem;
            color: rgba(244, 239, 232, 0.78);
            line-height: 1.5;
            margin-top: 0.2rem;
        }

        .hero {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1.25rem;
            padding: 1.7rem;
            border-radius: var(--radius-lg);
            background:
              linear-gradient(135deg, rgba(255, 253, 248, 0.92), rgba(255, 245, 235, 0.86)),
              white;
            border: 1px solid rgba(255, 255, 255, 0.46);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .hero-main {
            display: grid;
            gap: 0.35rem;
        }

        .hero-title {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.05;
            color: var(--text);
        }

        .hero-subtitle {
            margin: 0;
            font-size: 1rem;
            max-width: 54rem;
            color: var(--muted);
            line-height: 1.55;
        }

        .hero-chip {
            padding: 0.95rem 1.1rem;
            border-radius: 999px;
            background: rgba(17, 42, 53, 0.08);
            color: var(--sidebar);
            font-weight: 700;
            white-space: nowrap;
        }

        .page-panel {
            padding: 1.4rem;
            border-radius: var(--radius-lg);
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.45);
            box-shadow: var(--shadow);
        }

        .panel-title {
            margin: 0 0 0.2rem;
            font-size: 1.55rem;
            color: var(--text);
        }

        .panel-copy {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.45);
            border-radius: var(--radius-md);
            padding: 1rem 1rem 0.85rem;
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-family: "Source Serif 4", Georgia, serif;
        }

        .stButton > button {
            border: 0;
            border-radius: 999px;
            padding: 0.8rem 1.15rem;
            background: linear-gradient(135deg, var(--accent), #df7f55);
            color: white;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(201, 93, 61, 0.22);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
        }

        .stTextInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextArea textarea {
            border-radius: 14px !important;
            border: 1px solid rgba(17, 42, 53, 0.16) !important;
            background: var(--panel-strong) !important;
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
        }

        .logo-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.7rem 0.95rem;
            border-radius: 999px;
            background: rgba(17, 42, 53, 0.08);
            color: var(--sidebar);
            font-weight: 700;
        }

        .auth-shell {
            min-height: calc(100vh - 5rem);
            display: grid;
            grid-template-columns: 1.08fr 0.92fr;
            gap: 0;
            border-radius: 30px;
            overflow: hidden;
            box-shadow: 0 28px 56px rgba(25, 28, 29, 0.12);
            background: white;
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
            font-family: "Inter", sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            color: white;
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
            font-family: "Inter", sans-serif;
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
            font-family: "Inter", sans-serif;
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
            background: rgba(255,255,255,0.95);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }

        .auth-right-inner {
            width: 100%;
            max-width: 36rem;
        }

        .auth-brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.5rem;
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

        .auth-small-tag {
            font-size: 0.74rem;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            font-weight: 700;
            color: #004c4c;
        }

        .auth-right-title {
            margin: 0.2rem 0 0;
            font-family: "Inter", sans-serif;
            font-size: 2rem;
            font-weight: 800;
            color: #191c1d;
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
        }

        .auth-form-copy {
            margin: 0 0 1.5rem;
            font-size: 0.96rem;
            line-height: 1.7;
            color: #3f4948;
        }

        .auth-form-wrap {
            padding-top: 0.4rem;
        }

        .login-screen [data-testid="stSidebar"],
        .login-screen [data-testid="collapsedControl"] {
            display: none !important;
        }

        .login-screen .block-container {
            max-width: 1320px;
            padding-top: 1rem;
        }

        @media (max-width: 900px) {
            .hero {
                flex-direction: column;
            }

            .auth-shell {
                grid-template-columns: 1fr;
            }

            .auth-left {
                padding: 2rem;
            }

            .auth-headline {
                font-size: 2.2rem;
                max-width: none;
            }

            .auth-feature-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_sidebar() -> None:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-copy">
            <div class="eyebrow sidebar-eyebrow">GLearn Style</div>
            <h2 class="sidebar-title">SmartAttend</h2>
            <p class="sidebar-copy">Academic attendance workspace with enrollment, liveness verification, and reporting.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=84)


def render_sidebar_footer() -> None:
    st.markdown("---")
    st.caption("Administrator session active")
    if st.button("Reload Models", key="sidebar_reload_models"):
        load_recognizer.clear()
        load_liveness_detector.clear()
        load_face_detector.clear()
        st.success("Detector and model caches cleared.")
    if st.button("Logout", key="sidebar_logout"):
        set_admin_authenticated(False)
        st.rerun()


def render_page_header(title: str, eyebrow: str, subtitle: str, chip: str | None = None) -> None:
    chip_value = chip or datetime.now().strftime("%d %b %Y")
    logo_html = ""
    if LOGO_PATH.exists():
        logo_html = f'<img src="data:image/png;base64,{logo_to_base64()}" alt="University logo" style="height:56px; width:auto; border-radius:12px; margin-bottom:0.75rem;" />'
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-main">
            <div class="eyebrow">{eyebrow}</div>
            <div>{logo_html}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
          </div>
          <div class="hero-chip">{chip_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def logo_to_base64() -> str:
    import base64

    if not LOGO_PATH.exists():
        return ""
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")


def render_login_page() -> None:
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
                      <p>SQLite-backed student records, official attendance, and attempt logs stay in one place.</p>
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
            logo_html = f'<img src="data:image/png;base64,{logo_to_base64()}" alt="University Logo" style="height:40px; width:40px; object-fit:contain;" />'
        st.markdown(
            f"""
            <div class="auth-right">
              <div class="auth-right-inner">
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
            """,
            unsafe_allow_html=True,
        )

        with st.form("admin_login_form", clear_on_submit=False):
            username = st.text_input("Admin Username", value=DEFAULT_ADMIN_USERNAME, key="login_username")
            password = st.text_input("Admin Password", type="password", key="login_password")
            submitted = st.form_submit_button("Access Dashboard", use_container_width=True)

        if submitted:
            if verify_admin(username, password):
                set_admin_authenticated(True)
                st.session_state["admin_username"] = username
                st.success("Login successful. Opening dashboard...")
                st.rerun()
            else:
                st.error("Invalid admin credentials.")

        st.caption("Set `SMARTATTEND_ADMIN_USER` and `SMARTATTEND_ADMIN_PASSWORD` in the environment to change the default admin credentials.")
        st.markdown("</div></div>", unsafe_allow_html=True)


@st.cache_resource
def load_face_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource
def load_recognizer() -> FaceRecognizer:
    return FaceRecognizer()


@st.cache_resource
def load_liveness_detector() -> LivenessDetector:
    return LivenessDetector()


def set_admin_authenticated(value: bool) -> None:
    st.session_state["admin_authenticated"] = value


def is_admin_authenticated() -> bool:
    return bool(st.session_state.get("admin_authenticated", False))


def decode_camera_value(camera_value) -> object | None:
    if camera_value is None:
        return None
    return decode_uploaded_image(camera_value.getvalue())


def load_students_frame() -> pd.DataFrame:
    students = list_students()
    if not students:
        return pd.DataFrame(
            columns=[
                "first_name",
                "last_name",
                "roll_no",
                "email",
                "year",
                "program",
                "course",
                "present_count",
                "absent_count",
                "attendance_percentage",
            ]
        )
    frame = pd.DataFrame(students)
    return frame[
        [
            "first_name",
            "last_name",
            "roll_no",
            "email",
            "year",
            "program",
            "course",
            "present_count",
            "absent_count",
            "attendance_percentage",
        ]
    ]


def load_attendance_frame(limit: int = 50) -> pd.DataFrame:
    rows = list_recent_attendance(limit=limit)
    if not rows:
        return pd.DataFrame(
            columns=[
                "attendance_date",
                "attendance_time",
                "roll_no",
                "first_name",
                "last_name",
                "status",
                "confidence",
                "liveness_score",
                "note",
            ]
        )
    frame = pd.DataFrame(rows)
    return frame[
        [
            "attendance_date",
            "attendance_time",
            "roll_no",
            "first_name",
            "last_name",
            "status",
            "confidence",
            "liveness_score",
            "note",
        ]
    ]


def load_attempts_frame(limit: int = 50) -> pd.DataFrame:
    rows = list_recent_attempts(limit=limit)
    if not rows:
        return pd.DataFrame(
            columns=[
                "attempt_date",
                "attempt_time",
                "claimed_roll_no",
                "official_status",
                "attempt_outcome",
                "confidence",
                "liveness_score",
                "note",
                "predicted_roll_no",
            ]
        )

    frame = pd.DataFrame(rows)
    display_frame = pd.DataFrame(
        {
            "attempt_date": frame["attempt_date"],
            "attempt_time": frame["attempt_time"],
            "claimed_roll_no": frame["claimed_roll_no"],
            "official_status": frame["official_status"].fillna("-"),
            "attempt_outcome": frame["attempt_outcome"],
            "confidence": frame["confidence"],
            "liveness_score": frame["liveness_score"],
            "note": frame["note"],
            "predicted_roll_no": frame["predicted_roll_no"].fillna("-"),
        }
    )
    return display_frame


def model_health_rows() -> list[dict[str, str]]:
    face_detector = load_face_detector()
    recognizer = load_recognizer()
    liveness_detector = load_liveness_detector()
    counts = liveness_counts()
    return [
        {"component": "SQLite Database", "status": "ready", "details": str(DATABASE_PATH)},
        {"component": "Face Detector", "status": face_detector.backend, "details": "MTCNN when available, Haar fallback otherwise"},
        {"component": "Recognition Runtime", "status": "cnn" if recognizer.available else "fallback", "details": "CNN if trained, dataset matcher otherwise"},
        {"component": "Face Model", "status": "ready" if FACE_MODEL_PATH.exists() else "missing", "details": str(FACE_MODEL_PATH.name)},
        {"component": "Face Labels", "status": "ready" if FACE_LABELS_PATH.exists() else "missing", "details": str(FACE_LABELS_PATH.name)},
        {"component": "Liveness Model", "status": "ready" if LIVENESS_MODEL_PATH.exists() else "missing", "details": str(LIVENESS_MODEL_PATH.name)},
        {"component": "Liveness Dataset", "status": f"real={counts['real']} fake={counts['fake']}", "details": "Collect both classes before training"},
    ]


def render_dashboard() -> None:
    overview = attendance_overview()
    render_page_header(
        title=APP_TITLE,
        eyebrow="Academic Workspace",
        subtitle="Enrollment-first attendance workflow with face detection, liveness-aware verification, SQLite storage, and audit-friendly reporting.",
        chip=f"Attempts Logged: {overview['total_attempts']}",
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Enrolled Students", overview["total_students"])
    metric_cols[1].metric("Present Today", overview["present_today"])
    metric_cols[2].metric("Absent Today", overview["absent_today"])
    metric_cols[3].metric("Spoof Attempts Today", overview["spoof_attempts_today"])

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("System Health")
        st.dataframe(pd.DataFrame(model_health_rows()), use_container_width=True, hide_index=True)

        student_frame = load_students_frame()
        st.subheader("Attendance Percentage")
        st.caption("Percentages are based on recorded attendance outcomes for each enrolled student.")
        if student_frame.empty:
            st.info("No enrolled students yet.")
        else:
            chart_frame = student_frame[["roll_no", "attendance_percentage"]].set_index("roll_no")
            st.bar_chart(chart_frame, use_container_width=True)

    with right:
        st.subheader("Recent Attendance")
        attendance_frame = load_attendance_frame(limit=10)
        if attendance_frame.empty:
            st.info("No attendance scans recorded yet.")
        else:
            st.dataframe(attendance_frame, use_container_width=True, hide_index=True)

        st.subheader("Recent Attempts")
        attempts_frame = load_attempts_frame(limit=10)
        if attempts_frame.empty:
            st.info("No attendance attempts logged yet.")
        else:
            st.dataframe(attempts_frame, use_container_width=True, hide_index=True)

        face_report = get_evaluation_report("face_model")
        liveness_report = get_evaluation_report("liveness_model")
        st.subheader("Latest Evaluations")
        if face_report or liveness_report:
            report_rows = []
            if face_report and "accuracy" in face_report:
                report_rows.append({"model": "Face Recognition", "accuracy": round(face_report["accuracy"] * 100, 2)})
            if liveness_report and "accuracy" in liveness_report:
                report_rows.append({"model": "Liveness", "accuracy": round(liveness_report["accuracy"] * 100, 2)})
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No evaluation reports generated yet.")


def render_enrollment() -> None:
    render_page_header(
        title="Student Enrollment",
        eyebrow="Registration",
        subtitle="Capture a live face scan and register the student into the academic roster with verified metadata and face memory.",
    )

    col1, col2 = st.columns(2)
    first_name = col1.text_input("First Name", key="enroll_first_name")
    last_name = col2.text_input("Last Name", key="enroll_last_name")
    roll_no = col1.text_input("Roll Number", key="enroll_roll_no")
    email = col2.text_input("Email", key="enroll_email")
    year = col1.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"], key="enroll_year")
    program = col2.text_input("Program", placeholder="B. Tech", key="enroll_program")
    course = st.text_input("Course / Branch", placeholder="CSE", key="enroll_course")
    camera_value = st.camera_input("Capture Face for Enrollment", key="enroll_camera")

    if st.button("Enroll Student", type="primary", key="enroll_submit"):
        capture_bgr = decode_camera_value(camera_value)
        if capture_bgr is None:
            st.error("Capture a face image before submitting enrollment.")
            return
        required_values = [first_name, last_name, roll_no, email, year, program, course]
        if any(not value for value in required_values):
            st.error("All student details are required.")
            return

        with st.spinner("Processing enrollment..."):
            result = enroll_student(
                first_name=first_name,
                last_name=last_name,
                roll_no=roll_no,
                email=email,
                year=year,
                program=program,
                course=course,
                capture_bgr=capture_bgr,
                detector=load_face_detector(),
                liveness_detector=load_liveness_detector(),
            )

        if result.success:
            load_recognizer.clear()
            st.success(result.message)
            if result.student:
                student = result.student
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "first_name": student["first_name"],
                                "last_name": student["last_name"],
                                "roll_no": student["roll_no"],
                                "email": student["email"],
                                "year": student["year"],
                                "program": student["program"],
                                "course": student["course"],
                            }
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            if result.liveness_checked:
                st.caption(f"Liveness score: {result.liveness_score:.2f}")
            else:
                st.warning("Enrollment completed without a trained liveness model. Train the liveness model for spoof protection.")
        else:
            st.error(result.message)


def render_attendance() -> None:
    render_page_header(
        title="Attendance Verification",
        eyebrow="Live Check-In",
        subtitle="Verify the claimed roll number with a fresh face scan and liveness validation before attendance is accepted.",
    )

    liveness_ready = load_liveness_detector().available
    if not liveness_ready:
        counts = liveness_counts()
        st.error(
            "Liveness protection is required for attendance, but no trained liveness model is loaded. "
            f"Current dataset counts: real={counts['real']}, fake={counts['fake']}."
        )
        st.info("Go to `Liveness Setup`, save real and fake samples, train the liveness model, then reload the model from the app.")

    claimed_roll_no = st.text_input("Claimed Roll Number", key="attendance_roll_no")
    camera_value = st.camera_input("Capture Face for Attendance", key="attendance_camera")

    if st.button("Verify and Mark Attendance", type="primary", key="attendance_submit", disabled=not liveness_ready):
        capture_bgr = decode_camera_value(camera_value)
        if capture_bgr is None:
            st.error("Capture a face image before marking attendance.")
            return
        if not claimed_roll_no.strip():
            st.error("Claimed roll number is required.")
            return

        with st.spinner("Verifying identity and liveness..."):
            decision = verify_attendance_attempt(
                claimed_roll_no=claimed_roll_no,
                capture_bgr=capture_bgr,
                detector=load_face_detector(),
                recognizer=load_recognizer(),
                liveness_detector=load_liveness_detector(),
            )

        if decision.success:
            st.success(decision.message)
        else:
            if decision.status in {"Retry", "Unknown"}:
                st.warning(decision.message)
            else:
                st.error(decision.message)

        if decision.student:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "verified_for": f"{decision.student['first_name']} {decision.student['last_name']}",
                            "roll_no": decision.student["roll_no"],
                            "email": decision.student["email"],
                            "year": decision.student["year"],
                            "program": decision.student["program"],
                            "course": decision.student["course"],
                            "status": decision.status,
                            "confidence": round(decision.confidence, 4),
                            "liveness_score": round(decision.liveness_score, 4),
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        if decision.predicted_student:
            st.caption(
                "Predicted identity: "
                f"{decision.predicted_student['first_name']} {decision.predicted_student['last_name']} "
                f"({decision.predicted_student['roll_no']})"
            )


def render_students() -> None:
    if not is_admin_authenticated():
        st.warning("Admin login is required to view enrolled students and percentages.")
        return

    render_page_header(
        title="Enrolled Students",
        eyebrow="Roster",
        subtitle="Review student records, attendance totals, and current attendance percentage from the official SQLite roster.",
    )
    frame = load_students_frame()
    if frame.empty:
        st.info("No students are enrolled yet.")
        return

    query = st.text_input("Search by name, roll number, email, or course", key="student_search")
    if query.strip():
        mask = frame.astype(str).apply(lambda column: column.str.contains(query, case=False, na=False))
        frame = frame[mask.any(axis=1)]

    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_liveness_setup() -> None:
    if not is_admin_authenticated():
        st.warning("Admin login is required to manage liveness setup and training.")
        return

    render_page_header(
        title="Liveness Setup",
        eyebrow="Anti-Spoofing",
        subtitle="Collect real and fake face samples, train the liveness classifier, and keep spoof resistance aligned with the portal workflow.",
    )

    counts = liveness_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Real Samples", counts["real"])
    col2.metric("Fake Samples", counts["fake"])
    col3.metric("Model Status", "Ready" if load_liveness_detector().available else "Missing")

    st.markdown("#### Add Real Sample")
    real_capture = st.camera_input("Capture a live face", key="liveness_real_camera")
    if st.button("Save Real Sample", key="save_real_sample"):
        image = decode_camera_value(real_capture)
        if image is None:
            st.error("Capture a live face first.")
        else:
            result = save_liveness_sample(image, label="real", detector=load_face_detector(), source_prefix="real")
            if result.success:
                st.success(result.message)
                st.rerun()
            else:
                st.error(result.message)

    st.markdown("#### Add Fake Sample")
    st.caption("Point the camera at a printed face or a phone/laptop screen showing a face.")
    fake_capture = st.camera_input("Capture a spoof sample", key="liveness_fake_camera")
    if st.button("Save Fake Sample", key="save_fake_sample"):
        image = decode_camera_value(fake_capture)
        if image is None:
            st.error("Capture a fake face sample first.")
        else:
            result = save_liveness_sample(image, label="fake", detector=load_face_detector(), source_prefix="fake")
            if result.success:
                st.success(result.message)
                st.rerun()
            else:
                st.error(result.message)

    st.markdown("#### Train Liveness Model")
    ready, counts = dataset_ready_for_training()
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15, step=1, key="liveness_epochs")
    batch_size = st.number_input("Batch Size", min_value=2, max_value=64, value=8, step=2, key="liveness_batch")
    validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="liveness_val_split")

    if not ready:
        st.warning(
            "Collect more samples before training. "
            f"Minimum suggested starting point: 5 real and 5 fake. Current counts: real={counts['real']}, fake={counts['fake']}."
        )

    if st.button("Train Liveness Model", key="train_liveness_button"):
        try:
            with st.spinner("Training liveness model..."):
                result = train_liveness_model(
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    validation_split=float(validation_split),
                )
            load_liveness_detector.clear()
            st.success("Liveness model trained successfully.")
            st.json(result)
        except Exception as error:
            st.error(f"Liveness training failed: {error}")

    if st.button("Reload Liveness Model", key="reload_liveness_only"):
        load_liveness_detector.clear()
        st.success("Liveness model cache cleared. The next prediction will load the latest model.")


def render_reports() -> None:
    if not is_admin_authenticated():
        st.warning("Admin login is required to access evaluation and reporting.")
        return

    render_page_header(
        title="Evaluation and Reporting",
        eyebrow="Model Reports",
        subtitle="Generate confusion matrices, accuracy reports, false-acceptance metrics, and recent attempt logs for academic review.",
    )

    if st.button("Run Evaluation Suite", key="run_evaluations"):
        with st.spinner("Evaluating trained models..."):
            results = run_all_evaluations()
        st.success("Evaluation run completed.")
        st.json(results)

    face_report = get_evaluation_report("face_model")
    liveness_report = get_evaluation_report("liveness_model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Face Recognition")
        if face_report and "accuracy" in face_report:
            st.metric("Accuracy", f"{face_report['accuracy'] * 100:.2f}%")
            st.json(face_report["classification_report"])
            face_matrix = ARTIFACTS_DIR / "face_confusion_matrix.png"
            if face_matrix.exists():
                st.image(str(face_matrix))
        else:
            st.info("Face model evaluation is not available yet.")

    with col2:
        st.markdown("#### Liveness Detection")
        if liveness_report and "accuracy" in liveness_report:
            st.metric("Accuracy", f"{liveness_report['accuracy'] * 100:.2f}%")
            st.metric("False Acceptance Rate", f"{liveness_report['false_acceptance_rate'] * 100:.2f}%")
            st.metric("False Rejection Rate", f"{liveness_report['false_rejection_rate'] * 100:.2f}%")
            st.json(liveness_report["classification_report"])
            liveness_matrix = ARTIFACTS_DIR / "liveness_confusion_matrix.png"
            if liveness_matrix.exists():
                st.image(str(liveness_matrix))
        else:
            st.info("Liveness model evaluation is not available yet.")

    st.markdown("#### Attendance Attempt Log")
    attempts_frame = load_attempts_frame(limit=100)
    if attempts_frame.empty:
        st.info("No attendance attempts logged yet.")
    else:
        st.dataframe(attempts_frame, use_container_width=True, hide_index=True)


def render_admin() -> None:
    render_page_header(
        title="Admin Access",
        eyebrow="Operations",
        subtitle="Manage secure access, model reloads, and environment-backed administrator controls for the SmartAttend console.",
    )
    if not is_admin_authenticated():
        username = st.text_input("Admin Username", value=DEFAULT_ADMIN_USERNAME, key="admin_username")
        password = st.text_input("Admin Password", type="password", key="admin_password")
        if st.button("Login", key="admin_login"):
            if verify_admin(username, password):
                set_admin_authenticated(True)
                st.success("Admin login successful.")
                st.rerun()
            else:
                st.error("Invalid admin credentials.")
        st.caption("Set SMARTATTEND_ADMIN_USER and SMARTATTEND_ADMIN_PASSWORD in the environment to change the default admin credentials.")
        return

    st.success("Admin session active.")
    st.write(f"Database path: `{DATABASE_PATH}`")
    st.write("Use this section after new training runs to refresh cached model state inside the app.")
    if st.button("Reload Recognition and Liveness Models", key="reload_models"):
        load_recognizer.clear()
        load_liveness_detector.clear()
        load_face_detector.clear()
        st.success("Cached detector and model state cleared.")
    if st.button("Logout", key="admin_logout"):
        set_admin_authenticated(False)
        st.rerun()


def main() -> None:
    inject_styles()

    if not is_admin_authenticated():
        render_login_page()
        return

    with st.sidebar:
        render_brand_sidebar()
        page = st.radio(
            "Navigate",
            ["Dashboard", "Enroll Student", "Mark Attendance", "Students", "Liveness Setup", "Reports"],
        )
        st.caption("Administrator access unlocked")
        render_sidebar_footer()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Enroll Student":
        render_enrollment()
    elif page == "Mark Attendance":
        render_attendance()
    elif page == "Students":
        render_students()
    elif page == "Liveness Setup":
        render_liveness_setup()
    elif page == "Reports":
        render_reports()


if __name__ == "__main__":
    main()
