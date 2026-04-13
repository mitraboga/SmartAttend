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

1. Copy the example environment file and adjust the values you want to override:

```powershell
Copy-Item .env.example .env
```

2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Start the app:

```powershell
streamlit run app.py
```

## Environment Configuration

SmartAttend now loads configuration from a local `.env` file automatically.

Important variables:

- `SMARTATTEND_ADMIN_USER`: admin username shown on the login screen
- `SMARTATTEND_ADMIN_PASSWORD`: admin password used for the seeded admin account
- `SMARTATTEND_APP_TITLE`: dashboard title
- `SMARTATTEND_DATA_DIR`: base directory for SQLite data, face samples, and liveness samples
- `SMARTATTEND_MODELS_DIR`: trained model storage directory
- `SMARTATTEND_ARTIFACTS_DIR`: evaluation output directory
- `SMARTATTEND_DATABASE_PATH`: SQLite database file path
- `SMARTATTEND_FACE_DETECTOR_BACKEND`: detector backend, normally `auto`
- `SMARTATTEND_RECOGNITION_THRESHOLD`: confidence threshold for accepted face recognition predictions
- `SMARTATTEND_LIVENESS_THRESHOLD`: confidence threshold for accepted live-face predictions
- `SMARTATTEND_FACE_MATCHER_THRESHOLD`: fallback matcher similarity threshold

For a new deployment, set a strong admin password before first run.

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

These defaults are intended for local development only. Override them in `.env` for any real deployment.

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

## Deployment

### Local deployment

1. Copy `.env.example` to `.env`
2. Set your admin username and a strong admin password
3. Install dependencies with `pip install -r requirements.txt`
4. Start the app with `streamlit run app.py`

### Docker deployment

Build the image:

```powershell
docker build -t smartattend .
```

Run the container with a persistent data, model, and artifact mount:

```powershell
docker run --rm -p 8501:8501 --env-file .env -v "${PWD}\\data:/app/data" -v "${PWD}\\models:/app/models" -v "${PWD}\\artifacts:/app/artifacts" smartattend
```

Notes:

- The host-mounted `data` directory preserves the SQLite database and collected face/liveness samples
- The host-mounted `models` directory preserves trained CNN models
- The host-mounted `artifacts` directory preserves evaluation plots and reports
- If you deploy to a cloud service, inject the same variables from `.env.example` into the platform secret or environment settings instead of committing `.env`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
