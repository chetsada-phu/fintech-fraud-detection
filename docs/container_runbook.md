# Local container runbook

The container starts the artifact-backed FastAPI factory and serves both
`POST /scored-decisions` and `GET /analyst-demo`. It does not build a model or
copy local data into the image.

## Prepare the trusted local files

Run the data and artifact commands from the repository root before starting the
container:

```bash
fraud-generate-data
fraud-split-data
fraud-build-features
fraud-build-provisional-model
```

The scored factory validates four files during startup:

```text
models/xgboost_engineering_provisional.joblib
models/xgboost_engineering_provisional.metadata.json
data/processed/train.csv
data/processed/features/train_temporal_features.csv
```

Treat the joblib file as trusted local input. Its checksum detects accidental
changes but does not make pickle deserialization safe for an untrusted file.

## Build the image

```bash
docker build --tag fintech-fraud-demo:local .
```

The build context uses an allowlist. It contains the package source, README,
project metadata, and versioned configs. Datasets, model artifacts, Git data,
local environments, caches, and secrets stay outside the image.

## Start the scored application

```bash
docker run --rm \
  --name fintech-fraud-demo \
  --publish 8000:8000 \
  --mount type=bind,src="$(pwd)/models/xgboost_engineering_provisional.joblib",dst=/app/models/xgboost_engineering_provisional.joblib,readonly \
  --mount type=bind,src="$(pwd)/models/xgboost_engineering_provisional.metadata.json",dst=/app/models/xgboost_engineering_provisional.metadata.json,readonly \
  --mount type=bind,src="$(pwd)/data/processed/train.csv",dst=/app/data/processed/train.csv,readonly \
  --mount type=bind,src="$(pwd)/data/processed/features/train_temporal_features.csv",dst=/app/data/processed/features/train_temporal_features.csv,readonly \
  fintech-fraud-demo:local
```

The process runs as UID and GID `10001`. Every runtime input is mounted
read-only. The application keeps no transaction history and writes no result
files.

## Check the running service

```bash
curl --fail http://127.0.0.1:8000/healthz
```

Expected response:

```json
{"status":"ok","mode":"scored"}
```

Open `http://127.0.0.1:8000/analyst-demo` and score the default transaction.
The page should return a decision with ordered reasons, policy version, and
artifact identity. Docker also calls `/healthz` every 30 seconds after its
startup grace period.

## Verified local run

The complete container workflow was verified on 2026-08-21 with Docker CLI
29.7.2, Docker Engine 29.5.2, and Colima 0.10.3 on Apple silicon. The image
built for Linux arm64, started with the four mounts above, and became healthy.
The final Phase 6 rebuild produced image
`sha256:051196f8de915dbde33f24d6cf77411c07c2e3776ab8e4af8964bf6e23c2d448`.

Docker reported UID and GID `10001` for the running process and `RW=false` for
each mounted file. `GET /healthz`, `GET /analyst-demo`, and one
`POST /scored-decisions` request all returned HTTP 200. The scored request
returned risk score `0.9744988679885864`, decision `decline`, the expected
ordered reasons, and artifact SHA-256
`8cc107c232df366a6bb4db002135270ff11ea609990aa2b8335553acee854d85`.
This verifies the local run contract on one machine and architecture; it is not
a cloud deployment, load test, or production-readiness claim.
