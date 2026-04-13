<h1 align="center">🎓 SmartAttend 🧠</h1>
<h3 align="center">AI-Powered Smart Attendance System with Face Recognition, Liveness Detection & Secure Verification</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/TensorFlow-CNN-orange?logo=tensorflow" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green" alt="OpenCV" />
  <img src="https://img.shields.io/badge/SQLite-Local%20Database-lightgrey?logo=sqlite" alt="SQLite" />
  <img src="https://img.shields.io/badge/Liveness-Anti--Spoofing-yellow" alt="Liveness Detection" />
  <img src="https://img.shields.io/badge/Architecture-AI%20%7C%20CV%20%7C%20Data-brightgreen" alt="Architecture" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License" />
</p>

---

<p align="center">
  <a href="assets/demo.gif" target="_blank" rel="noopener noreferrer">
    <img src="assets/demo.gif" width="95%" alt="SmartAttend Live Demo Preview"/>
  </a>
</p>

<p align="center"><i>🔎 Click the preview above to view the SmartAttend demo workflow.</i></p>

### What You'll See:

- 🎓 Student enrollment with face capture
- 🧠 CNN-based face recognition
- 🛡️ Liveness detection against spoof attempts
- ✅ Attendance verification using claimed roll number
- 📊 Attendance logs, attempt logs, and analytics
- 📈 Model evaluation reports including ROC metrics

---

## 🚀 Executive Summary

**SmartAttend** is a **Streamlit-based smart attendance system** that combines:

- **Face Recognition**
- **Liveness Detection**
- **SQLite-backed record management**
- **Admin-first classroom workflow**

Traditional attendance systems ask:

> “Did someone mark present?”

SmartAttend asks the better question:

> **“Was the real student physically present, and was the scan genuine?”**

That is what makes this project stand out.

---

## 🎯 What It Solves

Traditional attendance workflows are weak because they often allow:

- Proxy attendance
- Manual verification overhead
- No verification of physical presence
- No fraud attempt tracking
- Poor auditability

**SmartAttend fixes this** by combining identity verification with anti-spoofing and structured logging.

---

## 💡 What It Does

- Enrolls students with a live face scan and academic details
- Stores student records in SQLite
- Verifies attendance using claimed roll number plus face scan
- Uses a CNN-based liveness model to reject spoof attempts
- Logs official attendance and separate verification attempts
- Shows roster data, attendance percentage, and model evaluation reports

---

## 🔁 Core Workflow

1. Admin logs into the Streamlit app  
2. Student is enrolled with face image, roll number, email, year, program, and course  
3. Face sample is saved and linked to the student record in SQLite  
4. Liveness dataset is collected from real and fake captures  
5. Face recognition and liveness models are trained locally  
6. Student scans again for attendance with the claimed roll number  
7. System detects the face, checks identity, runs liveness verification, and marks attendance  
8. Dashboard shows student records, attendance summaries, attempt logs, and evaluation reports  

---

# 🧩 System Architecture

<p align="center">
  <a href="assets/architecture.png">
    <img src="assets/architecture.png" width="95%" alt="SmartAttend Architecture Diagram"/>
  </a>
</p>

<p align="center"><i>End-to-end SmartAttend pipeline covering enrollment, verification, training, storage, and reporting.</i></p>

---

# 🏗 Architecture Overview

The system follows a modular AI attendance pipeline:

```text
Admin / Faculty
       ↓
Streamlit SmartAttend App
       ↓
Enrollment Module + Attendance Verification Module
       ↓
Face Detection
       ↓
Face Recognition CNN + Liveness Detection CNN
       ↓
Attendance Decision Engine
       ↓
SQLite Database + Attempt Logs + Attendance Logs
       ↓
Dashboard Reports + Evaluation Outputs
```

Separately:

```text
Collected Face Samples
        ↓
Face Training Pipeline
        ↓
Trained Face Recognition Model

Collected Real / Fake Liveness Samples
        ↓
Liveness Training Pipeline
        ↓
Trained Liveness Detection Model
```

---

## 🧠 Key Features

### 1. Anti-Spoofing Liveness Detection
Prevents fake attendance attempts using:
- Printed photos
- Mobile replay attacks
- Non-live image spoofing

### 2. Dual Logging Architecture
The system stores:
- Official attendance records
- Failed / suspicious verification attempts

This creates a much stronger audit trail.

### 3. Enrollment-First Identity Workflow
Attendance is tied to:
- Claimed roll number
- Registered face data
- Live verification result

### 4. Admin Dashboard
Admins can review:
- Student roster
- Attendance percentages
- Verification attempt logs
- Model evaluation outputs

### 5. Modular ML Design
Separate services handle:
- Enrollment
- Detection
- Recognition
- Liveness
- Logging
- Evaluation

This makes the project easier to extend and maintain.

---

## 🏗 Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- SQLite
- Matplotlib

---

## 📊 Model Performance

### Liveness Detection Model

| Metric | Value |
|--------|-------|
| Accuracy | Replace with actual value |
| Precision | Replace with actual value |
| Recall | Replace with actual value |
| F1 Score | Replace with actual value |
| ROC-AUC | Replace with actual value |

<p align="center">
  <a href="artifacts/evaluation/roc_curve.png">
    <img src="artifacts/evaluation/roc_curve.png" width="85%" alt="SmartAttend ROC Curve"/>
  </a>
</p>

<p align="center"><i>ROC curve for the liveness detection model. Replace with the real generated output from your evaluation pipeline.</i></p>

---

## 📂 Project Structure

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
|   |-- university_logo.png
|   |-- demo.gif
|   `-- architecture.png
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
    |-- generate_roc.py
    `-- utils.py
```

---

## ⚙️ Setup

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

---

## 🔑 Environment Configuration

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

---

## 🧪 Training

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

Generate ROC curve:

```powershell
python -m src.generate_roc
```

---

## 🔐 Default Admin Login

- Username: `admin`
- Password: `admin123`

These defaults are intended for local development only. Override them in `.env` for any real deployment.

---

## 🔒 Privacy and Repository Notes

This repository is intended to store source code and project structure. Personal face images, trained local models, SQLite databases, and generated artifacts are excluded from version control by default.

---

## 📌 Current Scope

- Enrollment-first attendance workflow
- Optional MTCNN detector with Haar fallback
- CNN-based identity recognition
- CNN-based liveness detection
- SQLite-backed student, attendance, and attempt logs
- Dashboard reporting and model evaluation outputs

---

## ⚠️ Limitations

- Model quality depends on locally collected training data
- Attendance percentage is event-based, not class-session based
- Liveness robustness depends on collecting varied real and spoof samples
- Public datasets and pretrained models are not bundled in the repo

---

## 🚀 Deployment

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

---

## 📈 Scalability & Production Considerations

- Multi-classroom expansion
- Role-based access control
- Cloud deployment support
- Better attendance session modelling
- Stronger spoof datasets for real-world robustness
- Future migration from SQLite to a scalable managed database
- Potential use of embeddings and vector similarity search for larger deployments

This project can grow from a classroom demo into a more production-oriented biometric attendance platform.

---

## 👤 Author

<p align="center">
  <b>Mitra Boga</b><br>
  <a href="https://www.linkedin.com/in/bogamitra/">
    <img src="https://img.shields.io/badge/LinkedIn-bogamitra-blue?logo=linkedin">
  </a>
  <a href="https://x.com/techtraboga">
    <img src="https://img.shields.io/badge/X-techtraboga-black?logo=x">
  </a>
</p>

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

---

> This repository demonstrates a real-world AI-powered attendance verification system using face recognition, liveness detection, secure logging, and a modular Streamlit-based workflow.
