# Architecture and ROC Explanation

This note is written for project report and viva use. It explains:

1. the current working SmartAttend architecture
2. the full AWS reference architecture
3. the ROC curve and liveness-threshold recalibration

---

## 1. Current Working Architecture Diagram

The diagram in [mermaid-diagram.png](../assets/mermaid-diagram.png) represents the architecture that is currently applicable and working for this project.

### What It Shows

The current system is built around these active layers:

- `User Layer`
  - student browser
  - admin / faculty browser

- `Application Layer`
  - Hugging Face Space
  - SmartAttend Streamlit application

- `AI Attendance Verification Pipeline`
  - face detection
  - face recognition
  - liveness detection
  - attendance decision engine

- `Cloud Storage / Persistence Layer`
  - Neon PostgreSQL
  - AWS S3

- `Deployment and Model Training Layer`
  - GitHub repository
  - GitHub Actions
  - Hugging Face deployment workflow
  - local training pipeline
  - trained model files
  - evaluation artifacts

### How The Current System Works

1. The student or admin/faculty user opens the SmartAttend web application through the browser.
2. The application is hosted in a Hugging Face Docker Space and runs the Streamlit interface.
3. During attendance verification, the application captures a face image from the webcam.
4. The face is first passed to the face detection module.
5. The detected face is then sent to:
   - the face recognition model
   - the liveness detection model
6. The attendance decision engine combines:
   - identity match result
   - liveness score
   - session eligibility
7. If all checks pass, attendance is marked successfully.
8. If the checks fail, the system records a failed attempt or exception instead of blindly marking present.
9. Structured records such as students, sessions, attendance logs, attempt logs, and exceptions are stored in Neon PostgreSQL.
10. Face-related runtime assets and enrolled image assets are stored in AWS S3.

### Why This Diagram Is Important

This is the real architecture of the current project, not just a theoretical design. It reflects the system exactly as it exists now:

- web app hosted on Hugging Face
- managed PostgreSQL database
- S3-backed asset storage
- local model training
- GitHub Actions based deployment

### Short Viva Explanation

You can say:

> The current SmartAttend architecture is a hybrid deployment. The user accesses a Streamlit application hosted on Hugging Face Spaces. Inside the application, the face image goes through face detection, face recognition, and liveness detection. The decision engine then checks identity, liveness, and session rules before attendance is stored in Neon PostgreSQL. Face assets are stored in AWS S3, and GitHub Actions handles CI/CD and deployment.

---

## 2. Full AWS Reference Architecture

The diagram in [SmartAttend AWS Architecture Diagram.png](../assets/SmartAttend%20AWS%20Architecture%20Diagram.png) shows how the platform would look if it were implemented fully on AWS.

### What It Shows

This AWS-oriented design includes:

- `Route 53`
  - DNS routing

- `Application Load Balancer`
  - distributes incoming traffic

- `Amazon EC2`
  - application server layer
  - runs the attendance web application

- `Amazon RDS`
  - structured storage for users, students, attendance, and logs

- `Amazon S3`
  - face images
  - trained models
  - reports
  - backups

- `AWS Lambda`
  - preprocessing / automation tasks

- `Amazon SageMaker`
  - model training pipeline

- `CloudWatch / CloudTrail / X-Ray`
  - monitoring
  - logging
  - tracing

- `IAM / WAF / SNS / SES`
  - security and notification services

### Why This Is a Reference Architecture

This is not the current deployment. It is a production-grade AWS-native reference model that shows how the system could scale if the project were fully migrated into AWS infrastructure.

### Difference Between Current and AWS Reference Architectures

The current working architecture:

- uses Hugging Face Spaces for hosting
- uses Neon for managed PostgreSQL
- uses S3 only for object storage
- trains models locally

The AWS reference architecture:

- replaces Hugging Face hosting with EC2 / ALB
- replaces Neon with Amazon RDS
- uses Lambda and SageMaker in the training and inference support path
- adds AWS-native observability and security components

### Why It Is Useful In A Report

This diagram shows architectural maturity. It demonstrates that the project is not only functional as a university system, but also has a clear path toward a cloud-native deployment model.

### Short Viva Explanation

You can say:

> The AWS diagram is a future-state cloud reference architecture. In that version, the web application would run behind Route 53 and an Application Load Balancer, the backend would run on EC2, the relational data would move to Amazon RDS, the image and model assets would stay in S3, and the model training pipeline could be moved to Lambda plus SageMaker. This shows how the project can evolve from a working academic system into a scalable production cloud platform.

---

## 3. ROC Curve and Threshold Recalibration

The ROC curve is used to evaluate the liveness detection model.

The generated figure is:

- [liveness_roc_curve.png](../assets/liveness_roc_curve.png)

The confusion matrix is:

- [liveness_confusion_matrix.png](../assets/liveness_confusion_matrix.png)

The current evaluation metrics are stored in:

- [liveness_metrics.json](../artifacts/liveness_metrics.json)

### What The ROC Curve Means

ROC stands for `Receiver Operating Characteristic`.

It shows how well the liveness model separates the two classes:

- `fake`
- `real`

Instead of using one fixed threshold, the ROC curve evaluates model behavior across many possible thresholds.

The two axes are:

- `False Positive Rate (FPR)`
  - how often fake samples are incorrectly accepted

- `True Positive Rate (TPR)`
  - how often real samples are correctly accepted

A better model produces a curve that moves closer to the top-left corner.

### Current ROC Result

For the current local evaluation run:

- ROC-AUC = `1.0000`

This means the current liveness scores perfectly rank the real and fake samples in this local evaluation set.

### Why Threshold Recalibration Was Needed

Earlier, the liveness model was using a threshold of:

- original threshold = `0.3000`

That threshold was too permissive and created a mismatch between the score distribution and the operating decision boundary.

The ROC analysis showed that a better operating threshold on the current dataset is:

- recalibrated threshold = `0.4947`

The evaluation pipeline now recalibrates the threshold using the ROC outputs and updates the stored metadata accordingly.

### Effect Of Recalibration

After recalibration, the current local evaluation snapshot became:

- Accuracy = `1.0000`
- Precision = `1.0000`
- Recall = `1.0000`
- F1 Score = `1.0000`
- False Acceptance Rate = `0.0000`
- False Rejection Rate = `0.0000`

And the confusion matrix became:

- fake predicted as fake: `15`
- fake predicted as real: `0`
- real predicted as fake: `0`
- real predicted as real: `20`

### Important Academic Note

These values are excellent, but they come from a small local dataset:

- `15 fake`
- `20 real`

So the correct academic interpretation is:

- the model performs perfectly on the current local evaluation set
- the dataset is small, so stronger generalization claims should be avoided
- more diverse real and spoof samples would be needed for a stronger real-world benchmark

### Short Viva Explanation

You can say:

> The ROC curve evaluates the liveness model across multiple thresholds instead of only one fixed decision point. Our local evaluation produced an AUC of 1.0, which means the model ranked real and fake samples perfectly on the current dataset. However, the earlier threshold was not the best operating point, so we recalibrated it from 0.30 to about 0.4947 using the ROC output. After recalibration, the confusion matrix showed perfect separation on the current local evaluation set.

---

## Final Summary

The two architecture diagrams and the ROC analysis together show three important things:

1. the system is already deployed and working in a real hybrid architecture
2. the system has a clear scalable cloud architecture path
3. the liveness model is now evaluated and calibrated using proper classification metrics instead of only raw accuracy

That makes the project stronger both as a practical product and as a deep learning case study.
