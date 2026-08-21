# Demo recording

[Watch the 7.88-second analyst demo](fintech-fraud-demo.webm).

The 1280 by 720 WebM recording uses the real artifact-backed application. It
loads the default precomputed transaction, submits it to
`POST /scored-decisions`, and shows the returned decline, risk score, ordered
reasons, policy version, and artifact identity. The recording has no narration
or synthetic product footage. Its SHA-256 is
`ee78fe3a11aa25c01fd9bd6151e7f8a863091d5af489246b6bd81d328355e8ee`.

## Reproduce the recording

Prepare the local data and checked artifact:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev,demo]"
playwright install chromium
fraud-generate-data
fraud-split-data
fraud-build-features
fraud-build-provisional-model
```

Start the scored application in one terminal:

```bash
.venv/bin/uvicorn fraud_detection.serving.api:create_scored_app \
  --factory --host 127.0.0.1 --port 8768
```

Record it from another terminal:

```bash
python3 scripts/record_demo.py
```

The script writes `docs/demo/fintech-fraud-demo.webm`. Playwright records the
browser viewport, so the output remains a local product walkthrough rather than
a performance measurement.
