# SmartAttend

SmartAttend is a Streamlit-based smart attendance system that combines face recognition, liveness detection, SQLite storage, and an admin-first workflow for classroom attendance management.

## What It Does

- Enrolls students with a live face scan and academic details
- Stores student records in SQLite
- Verifies attendance using claimed roll number plus face scan
- Uses a CNN-based liveness model to reject spoof attempts
- Logs official attendance and separate verification attempts
- Shows roster data, attendance percentage, and model evaluation reports

## Core Workflow

1. Admin logs into the Streamlit app
2. Student is enrolled with face image, roll number, email, year, program, and course
3. Face sample is saved and linked to the student record in SQLite
4. Liveness dataset is collected from real and fake captures
5. Face recognition and liveness models are trained locally
6. Student scans again for attendance with the claimed roll number
7. System detects the face, checks identity, runs liveness verification, and marks attendance
8. Dashboard shows student records, attendance summaries, attempt logs, and evaluation reports

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- SQLite
- Matplotlib

## Project Structure

```text
SmartAttend/
|-- app.py
|-- Dockerfile
|-- requirements.txt
|-- README.md
|-- .streamlit/
|   `-- config.toml
|-- docs/
|   `-- project_blueprint.md
|-- artifacts/
|   `-- .gitkeep
|-- assets/
|   `-- university_logo.png
|-- data/
|   |-- attendance/
|   |   `-- .gitkeep
|   |-- faces/
|   |   `-- .gitkeep
|   `-- liveness/
|       |-- fake/
|       |   `-- .gitkeep
|       `-- real/
|           `-- .gitkeep
|-- models/
|   `-- .gitkeep
|-- notebooks/
|   `-- README.md
|-- scripts/
`-- src/
    |-- attendance_logger.py
    |-- attendance_service.py
    |-- collect_faces.py
    |-- config.py
    |-- database.py
    |-- detect_and_mark.py
    |-- enrollment_service.py
    |-- evaluate_models.py
    |-- face_detector.py
    |-- liveness.py
    |-- liveness_dataset_service.py
    |-- recognizer.py
    |-- train_face_model.py
    |-- train_liveness_model.py
    `-- utils.py
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Training

Train the face recognition model:

```powershell
python -m src.train_face_model --epochs 20
```

Train the liveness model:

```powershell
python -m src.train_liveness_model --epochs 15
```

Run evaluation:

```powershell
python -m src.evaluate_models
```

## Default Admin Login

- Username: `admin`
- Password: `admin123`

These can be changed with environment variables:

- `SMARTATTEND_ADMIN_USER`
- `SMARTATTEND_ADMIN_PASSWORD`

## Privacy and Repository Notes

This repository is intended to store source code and project structure. Personal face images, trained local models, SQLite databases, and generated artifacts are excluded from version control by default.

## Current Scope

- Enrollment-first attendance workflow
- Optional MTCNN detector with Haar fallback
- CNN-based identity recognition
- CNN-based liveness detection
- SQLite-backed student, attendance, and attempt logs
- Dashboard reporting and model evaluation outputs

## Limitations

- Model quality depends on locally collected training data
- Attendance percentage is event-based, not class-session based
- Liveness robustness depends on collecting varied real and spoof samples
- Public datasets and pretrained models are not bundled in the repo
