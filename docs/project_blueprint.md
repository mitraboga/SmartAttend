# Project Blueprint

## Project Title

SmartAttend: A CNN-Based Smart Attendance System with Face Recognition, Liveness Detection, and SQLite Logging

## Final Scope

The project is now defined around an enrollment-first workflow rather than a standalone webcam script.

### Enrollment

1. Student scans face from the Streamlit dashboard
2. Student enters:
   - First name
   - Last name
   - Roll number
   - Email
   - Year
   - Program
   - Course
3. System stores the student record and face sample in SQLite

### Attendance

1. Student enters claimed roll number
2. Student scans face again
3. System detects the face
4. System verifies identity
5. System runs liveness detection when the model is available
6. System marks:
   - `Present` if the claimed student and scanned face match and liveness passes
   - `Absent` if the claimed student does not match or spoofing is detected

### Dashboard

The dashboard displays:

- Enrolled students and their details
- Attendance history
- Attendance percentage
- Model health
- Evaluation results

## Modules

### Module 1: Face Detection

- Uses MTCNN when available
- Falls back to OpenCV Haar detection otherwise

### Module 2: Face Recognition

- Uses a trained CNN when available
- Falls back to stored-face matching for enrollment-first usability

### Module 3: Liveness Detection

- Uses a binary CNN classifier to separate live and fake captures
- Integrates into both enrollment and attendance when trained

### Module 4: SQLite Data Layer

- Stores students
- Stores face samples
- Stores attendance events
- Stores admin users
- Stores evaluation reports

### Module 5: Streamlit Product Layer

- Public enrollment page
- Public attendance page
- Admin roster page
- Admin reporting page

## Functional Requirements

- The system shall enroll a student from the dashboard using a live face scan and academic metadata
- The system shall persist enrolled student details in SQLite
- The system shall verify attendance using a claimed roll number and a fresh face scan
- The system shall perform liveness-aware verification when a trained liveness model is present
- The system shall mark attendance as present or absent for the claimed student
- The system shall display enrolled students and attendance percentage in the dashboard
- The system shall prevent duplicate present attendance for the same student on the same day

## Non-Functional Requirements

- The system should be usable through a browser-based interface
- The system should run on a standard laptop for demonstration
- The system should be extensible for stronger detectors and larger datasets
- The system should keep a clean audit trail through SQLite records

## Architecture

```text
Streamlit Dashboard
    ->
Enrollment Page --------------------------> SQLite Student + Face Sample Storage
    ->
Attendance Page
    ->
Face Detection
    ->
Identity Verification + Liveness Verification
    ->
Attendance Decision
    ->
SQLite Attendance Storage
    ->
Roster / Percentage / Reports / Admin Views
```

## Data Schema Summary

### Students

- first_name
- last_name
- roll_no
- email
- year
- program
- course
- face_label
- primary_face_path

### Face Samples

- student_id
- image_path
- source

### Attendance

- student_id
- attendance_date
- attendance_time
- status
- confidence
- liveness_score
- note
- claimed_roll_no

## Model Plan

### Face Recognition CNN

- Input: `128 x 128 x 3`
- Conv block
- Conv block
- Conv block
- Dense + dropout
- Softmax over enrolled labels

### Liveness CNN

- Input: `128 x 128 x 3`
- Conv block
- Conv block
- Conv block
- Dense + dropout
- Sigmoid output for live or fake

## Evaluation Plan

### Face Recognition

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Liveness

- Accuracy
- Precision
- Recall
- False acceptance rate
- False rejection rate
- Confusion matrix

## Added Scope Items

The project now includes the missing system pieces that were previously listed as not yet integrated:

- SQLite storage instead of CSV-only storage
- Enrollment-first data collection from the dashboard
- Optional advanced detector support through MTCNN
- Evaluation report generation and confusion matrices
- Admin login inside the dashboard
- A more complete Streamlit product interface
- Docker-based deployment setup
- Active liveness integration in the attendance decision flow

## Remaining Practical Limits

- No public pretrained dataset is bundled in the repository
- Benchmarked accuracy still depends on running local training with real data
- Liveness robustness depends on collecting enough real and spoof samples
- Attendance percentage is event-based until a formal class-session model is added

## Expected Outcome

The finished system should allow a student to enroll once through the dashboard, persist identity data and face samples in SQLite, and later verify attendance through face recognition and liveness-aware checks. The admin side should be able to review student details, attendance percentage, and model evaluation results from the same interface.
