# Provisional Model Artifact Contract

This contract covers the bounded offline lifecycle for the unchanged XGBoost
engineering provisional pipeline. It does not promote the model. A separate
FastAPI factory now consumes this contract for trusted-local serving.

## Build Command

Run from the repository root after generating chronological splits and temporal
features:

```bash
fraud-build-provisional-model
```

The command reads exactly these ignored local inputs:

- `data/processed/train.csv`
- `data/processed/features/train_temporal_features.csv`
- `configs/xgboost_baseline.toml`
- `configs/selected_decision_policy.toml`

It does not open validation or test transactions or features. It fits the
unchanged fixed XGBoost pipeline on training only and atomically replaces:

- `models/xgboost_engineering_provisional.joblib`
- `models/xgboost_engineering_provisional.metadata.json`

The `models/` directory and `*.joblib` files are ignored by Git. The artifact is
a reproducible local build output, not a source file.

## Deterministic Metadata

The JSON sidecar is rendered with sorted keys and contains no timestamp, random
identifier, absolute path, username, or host value. It records:

- artifact version and artifact SHA-256;
- frozen XGBoost configuration SHA-256;
- the exact engineering-only score-source label;
- the ordered 11-field XGBoost input contract;
- training split name, transaction and fraud counts, transaction CSV SHA-256,
  and temporal-feature CSV SHA-256.

The checked loader strictly rejects missing or additional metadata fields. It
also rejects a version, feature contract, configuration hash, score-source
label, training provenance, or artifact checksum mismatch before calling
`joblib.load`.

## Feature Boundary

The artifact uses the frozen baseline fields in this order:

1. transaction amount;
2. transaction day index;
3. derived sine and cosine time-of-day values;
4. five strictly prior customer-history fields;
5. synthetic customer and terminal identifiers.

Post-event fraud labels are used only as supervised targets during the offline
training fit. They are not model input fields. Terminal temporal history remains
excluded because the validation-only terminal challenger was rejected.

## Trust and Portability Boundary

The SHA-256 check detects accidental file changes but does not authenticate an
artifact. Joblib uses Python pickle semantics, so only artifacts produced and
controlled locally by this project may be loaded. Never load a downloaded,
user-supplied, or otherwise untrusted joblib file.

The current artifact and metadata are byte-stable across repeated builds in the
verified Python 3.13.7 environment with the pinned dependencies. Byte identity
across another Python version, operating system, or dependency build has not
been established; rebuild from the recorded training sources instead.

## Serving Boundary

`create_scored_app` uses the checked loader once during application creation,
then passes the already loaded artifact to the framework-free scored-decision
use case for each request. The decision-only `create_app` factory remains
model-free. Run the scored application only with a trusted artifact built
locally by this project:

```bash
uvicorn fraud_detection.serving.api:create_scored_app --factory
```

No application startup or request fits a model. The HTTP boundary accepts only
label-free, precomputed model inputs and does not compute or persist live
customer temporal state. This integration does not start a dashboard,
containerize the application, establish model promotion, make untrusted pickle
safe, or establish production readiness.
