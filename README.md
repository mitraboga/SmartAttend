# SmartAttend

SmartAttend is a Streamlit-based smart attendance system that combines face recognition, liveness detection, SQLite storage, and an admin-first workflow for classroom attendance management.

## What It Does

- Enrolls students with a live face scan and academic details
- Stores student records in SQLite
- Verifies attendance using a claimed roll number plus a fresh face scan
- Uses a CNN-based liveness model to reject spoof attempts
- Logs official attendance separately from failed or suspicious attempts
- Shows roster data, attendance percentage, and evaluation reports

## Core Workflow

1. Admin logs into the Streamlit app
2. Student is enrolled with face image, roll number, email, year, program, and course
3. Face sample is saved and linked to the student record in SQLite
4. Liveness data is collected from real and fake captures
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
|-- requirements-local.txt
|-- README.md
|-- .env.example
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

1. Copy the example environment file and adjust any values you want to override:

```powershell
Copy-Item .env.example .env
```

2. Create a virtual environment and install dependencies:

For local development and training:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
```

3. Start the app:

```powershell
streamlit run app.py
```

## Dependency Profiles

This repository uses two dependency files:

- `requirements.txt`: cloud-safe base dependencies for Streamlit deployment
- `requirements-local.txt`: full local development and training dependencies, including TensorFlow and MTCNN

Use `requirements-local.txt` on your own machine when you want training, model loading, and the full deep learning workflow.
Use `requirements.txt` for Streamlit Community Cloud deployment.

## Environment Configuration

SmartAttend loads configuration from a local `.env` file automatically.

Important variables:

- `SMARTATTEND_ADMIN_USER`
- `SMARTATTEND_ADMIN_PASSWORD`
- `SMARTATTEND_APP_TITLE`
- `SMARTATTEND_DATA_DIR`
- `SMARTATTEND_MODELS_DIR`
- `SMARTATTEND_ARTIFACTS_DIR`
- `SMARTATTEND_DATABASE_PATH`
- `SMARTATTEND_FACE_DETECTOR_BACKEND`
- `SMARTATTEND_RECOGNITION_THRESHOLD`
- `SMARTATTEND_LIVENESS_THRESHOLD`
- `SMARTATTEND_FACE_MATCHER_THRESHOLD`

For a real deployment, set a strong admin password before first run.

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

This repository stores source code and project structure only. Personal face images, trained local models, SQLite databases, and generated artifacts are excluded from version control by default.

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

### Streamlit Community Cloud

- Main file path: `app.py`
- Use the repository root `requirements.txt`
- Do not expect local trained models, SQLite data, or captured face samples to be bundled unless you move them to persistent external storage

The cloud deployment is suitable for the dashboard and application shell. The full training workflow is still best run locally because TensorFlow support, model files, and data persistence are more reliable there.

### Docker

Build the image:

```powershell
docker build -t smartattend .
```

Run the container with persistent data, model, and artifact mounts:

```powershell
docker run --rm -p 8501:8501 --env-file .env -v "${PWD}\\data:/app/data" -v "${PWD}\\models:/app/models" -v "${PWD}\\artifacts:/app/artifacts" smartattend
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
