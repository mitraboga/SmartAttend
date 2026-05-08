---
title: SmartAttend
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# SmartAttend

SmartAttend is a Streamlit-based smart attendance platform with face recognition, liveness detection, session-based attendance, audit logging, and exception review workflows.

## Required Hugging Face Space secrets

- `SMARTATTEND_ADMIN_USER`
- `SMARTATTEND_ADMIN_PASSWORD`
- `SMARTATTEND_DATABASE_URL`
- `SMARTATTEND_STORAGE_BACKEND`
- `SMARTATTEND_S3_BUCKET`
- `SMARTATTEND_S3_REGION`
- `SMARTATTEND_S3_PREFIX`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

## Notes

- This Space is configured as a Docker Space and serves the app on port `8501`.
- Use Hugging Face Space variables for non-sensitive config and Space secrets for credentials.
- Free CPU Basic hardware sleeps when idle, so this is a low-cost demo/runtime target rather than a strict always-on production host.
