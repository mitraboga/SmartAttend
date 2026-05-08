# Hugging Face Space Deployment

This project is now container-ready, so the best free deployment target is a public Hugging Face Docker Space backed by external Postgres and S3.

## Why this target

- Hugging Face Spaces offers a free `CPU Basic` tier for public Spaces.
- Docker Spaces can run the existing `Dockerfile` directly.
- The app already stores durable state outside the container through managed Postgres and S3, which avoids the free-space ephemeral filesystem problem.

## Recommended architecture

- App host: Hugging Face Docker Space
- Database: managed Postgres via `SMARTATTEND_DATABASE_URL`
- Object storage: AWS S3 via `SMARTATTEND_STORAGE_BACKEND=s3`

## One-time setup

1. Create a new public Space on Hugging Face.
2. Choose `Docker` as the SDK.
3. Keep the hardware on `CPU Basic` if you want the free tier.
4. Add the required Space secrets and variables from the checklist below.
5. Optional non-secret variables:
   - `SMARTATTEND_INSTITUTION_NAME`
   - `SMARTATTEND_RECOGNITION_THRESHOLD`
   - `SMARTATTEND_LIVENESS_THRESHOLD`
   - `SMARTATTEND_FACE_MATCHER_THRESHOLD`

## Exact deployment checklist

### A. Hugging Face Space settings

Set these in the Space `Settings` page.

#### Space Variables

These are non-sensitive.

- `SMARTATTEND_INSTITUTION_NAME`
  Example: `GITAM University`
- `SMARTATTEND_RECOGNITION_THRESHOLD`
  Example: `0.75`
- `SMARTATTEND_LIVENESS_THRESHOLD`
  Example: `0.50`
- `SMARTATTEND_FACE_MATCHER_THRESHOLD`
  Example: `0.90`

#### Space Secrets

These are sensitive and should go into `Secrets`, not `Variables`.

- `SMARTATTEND_ADMIN_USER`
  Example value: `admin`
- `SMARTATTEND_ADMIN_PASSWORD`
  Example value: `ReplaceWithAStrongAdminPassword123!`
- `SMARTATTEND_DATABASE_URL`
  For the running app, prefer Neon pooled format:
  `postgresql://<neon_user>:<neon_password>@<neon-pooler-host>/<neon_db>?sslmode=require`
  Example:
  `postgresql://smartattend_owner:Abc123Secure@ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech/smartattend?sslmode=require`
- `SMARTATTEND_STORAGE_BACKEND`
  Exact value: `s3`
- `SMARTATTEND_S3_BUCKET`
  Example value: `smartattend-prod-assets`
- `SMARTATTEND_S3_REGION`
  Example value: `ap-south-1`
- `SMARTATTEND_S3_PREFIX`
  Example value: `smartattend`
- `AWS_ACCESS_KEY_ID`
  Example value format: `AKIA...`
- `AWS_SECRET_ACCESS_KEY`
  Example value format: a long base64-like AWS secret
- `AWS_DEFAULT_REGION`
  Exact value should match the bucket region.
  Example: `ap-south-1`

### B. Neon project values

Create a Neon project and copy the connection strings from the Neon dashboard `Connect` dialog.

- Database provider: Neon
- Connection type for Hugging Face runtime: pooled is preferred
- Connection type for GitHub migration job: direct is preferred
- Database name example: `smartattend`
- Role/user example: `smartattend_owner`
- Required query suffix: `?sslmode=require`

Recommended Hugging Face runtime `SMARTATTEND_DATABASE_URL` shape:

```text
postgresql://smartattend_owner:<PASSWORD>@<YOUR-POOLER-HOST-WITH--POOLER>/smartattend?sslmode=require
```

Runtime example:

```text
postgresql://smartattend_owner:Abc123Secure@ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech/smartattend?sslmode=require
```

Recommended GitHub Actions migration secret `SMARTATTEND_DATABASE_URL` shape:

```text
postgresql://smartattend_owner:<PASSWORD>@<YOUR-DIRECT-HOST>/smartattend?sslmode=require
```

Migration example:

```text
postgresql://smartattend_owner:Abc123Secure@ep-cool-darkness-a1b2c3d4.us-east-2.aws.neon.tech/smartattend?sslmode=require
```

### C. AWS S3 values

Create one private S3 bucket for enrolled face assets and set:

- Bucket name example: `smartattend-prod-assets`
- Region example: `ap-south-1`
- Prefix example: `smartattend`

Map those values to:

- `SMARTATTEND_S3_BUCKET=smartattend-prod-assets`
- `SMARTATTEND_S3_REGION=ap-south-1`
- `SMARTATTEND_S3_PREFIX=smartattend`
- `AWS_DEFAULT_REGION=ap-south-1`

The app only needs to upload and download objects under the configured prefix, so the IAM user can be limited to:

- `s3:PutObject`
- `s3:GetObject`

on:

- `arn:aws:s3:::smartattend-prod-assets/smartattend/*`

### D. GitHub repository secrets

Set these in GitHub repo `Settings -> Secrets and variables -> Actions`.

- `HF_TOKEN`
  Value: your Hugging Face write token with permission to push to the target Space
- `HF_SPACE_ID`
  Exact format: `<your-hf-username>/<your-space-name>`
  Example: `mitraboga/smartattend`
- `SMARTATTEND_DATABASE_URL`
  Prefer the Neon direct connection string for the migration workflow

### E. Final pre-deploy check

Before pushing to `main`, verify:

1. Hugging Face Space SDK is `Docker`
2. Space hardware is `CPU Basic` if you want the free tier
3. Space secret `SMARTATTEND_STORAGE_BACKEND` is exactly `s3`
4. Hugging Face `SMARTATTEND_DATABASE_URL` includes `sslmode=require`
5. `AWS_DEFAULT_REGION` matches `SMARTATTEND_S3_REGION`
6. The S3 bucket exists in that same region
7. The IAM user behind `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` can read and write the configured prefix
8. GitHub secrets `HF_TOKEN` and `HF_SPACE_ID` are set if you want automatic deployment
9. GitHub Actions `SMARTATTEND_DATABASE_URL` uses the Neon direct host, not the `-pooler` host

## GitHub Actions deployment

This repo includes `.github/workflows/deploy-huggingface-space.yml`.

Configure these GitHub repository secrets:

- `HF_TOKEN`
- `HF_SPACE_ID`
  Example: `your-hf-username/smartattend`

The workflow mirrors the repo into the Hugging Face Space, swaps in the Docker-Space-specific README from `deploy/huggingface/SPACE_README.md`, and force-pushes the deployment snapshot to the Space.

## Operational notes

- Free Spaces can sleep when idle.
- Disk inside the Space is not your durable datastore.
- Keep attendance state in Postgres and face assets in S3.
- If you need always-on uptime or private hosting later, move the same container to a paid target such as ECS/Fargate, Render, Azure App Service, or another container platform.
