# Real-time Payment Fraud Detection

This portfolio project is an end-to-end FinTech machine learning system that
turns a precomputed transaction record into a fraud risk score and an
operational decision: `approve`, `review`, or `decline`. It combines transparent
rules, machine learning, capacity-aware thresholds, stable reason codes, and
explicit artifact provenance.

## Current status

All six implementation phases are complete locally. The repository includes
deterministic data generation, chronological evaluation, leakage-safe temporal
features, rules and ML baselines, a three-way decision policy, a checked model
artifact, FastAPI serving, an analyst page, a non-root container, monitoring
examples, CI configuration, and a recorded demo.

XGBoost remains an engineering-only provisional score source. Its validation
advantage over Logistic Regression was small, and the held-out evidence was
mixed, so this project does not claim model promotion. The selected
`0.93/0.93` policy also has an empty validation review band. These limits are
documented rather than hidden behind a production claim.

The quickest overview is in [the architecture note](docs/architecture.md).
A [short browser recording](docs/demo/fintech-fraud-demo.webm) shows the real
scored application.

## Reproduce and run the demo

Python 3.11 or newer is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
fraud-generate-data
fraud-split-data
fraud-build-features
fraud-build-provisional-model
python scripts/smoke_demo.py
uvicorn fraud_detection.serving.api:create_scored_app --factory
```

Open `http://127.0.0.1:8000/analyst-demo`. The smoke command checks the real
artifact-backed health endpoint, analyst page, and scored request before the
server starts.

## Development setup

Python 3.11 or newer is required. Create an isolated environment so this
project's tools do not conflict with packages installed elsewhere:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
```

This builds and installs the local package. Re-run the install command after
changing package code or project metadata. Run the checks with:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Generate the Raw Sample

The generator reads versioned assumptions from `configs/data_generation.toml`,
validates every transaction, and atomically writes an ignored local CSV:

```bash
fraud-generate-data
```

The default output is `data/raw/transactions.csv`. The command reports the row
count, fraud-scenario counts, seed, and SHA-256 checksum so repeated runs can be
compared. See `docs/data_contract.md` for column definitions, provenance,
licensing, simplifications, and the leakage boundary.

## Create Chronological Splits

The splitter validates the raw CSV and reads versioned target fractions from
`configs/data_split.toml`. It keeps equal timestamps in the same partition so
the train, validation, and test time ranges cannot overlap:

```bash
fraud-split-data
```

The command writes `train.csv`, `validation.csv`, and `test.csv` atomically per
file under the ignored `data/processed/` directory. The labeled CSVs support
later supervised training and evaluation; reusable model code must select the
explicit `MODEL_FEATURE_COLUMNS` contract, which excludes identifiers and
post-event fraud labels.

## Refresh Focused EDA

Generate the versioned descriptive report after creating the processed splits:

```bash
fraud-profile-data
```

The command revalidates the complete dataset and strict split boundaries before
atomically refreshing `docs/eda_report.md`. The report contains measured time
ranges, class balance, transaction-amount summaries, entity coverage, fraud
scenario support, and interpretation limits. It does not claim model or
financial performance.

## Evaluate the Rule Baseline

Evaluate the versioned decision-time rules on the chronological validation and
test splits:

```bash
fraud-evaluate-rules
```

The command reads `configs/rule_baseline.toml` and refreshes
`docs/rule_baseline_report.md`. It records precision, recall, false-positive
rate, fraud-amount capture, reason-code counts, and uneven fraud-scenario
support. The binary flag is only a transparent baseline; it is not yet a final
approve, review, or decline policy.

## Compare Rules and Logistic Regression

Fit the fixed Logistic Regression pipeline on the training split only, then
compare it with the rule baseline on validation and test:

```bash
fraud-compare-baselines
```

The command reads the versioned Logistic, rule, and simulated business-cost
configurations and refreshes `docs/baseline_comparison.md`. It reports average
precision, ROC-AUC, fixed-threshold classification metrics, fraud-amount
capture, manual-review capacity, and illustrative operating cost. Test results
are reported once and must not be used to tune the fixed baseline.

## Build Past-only Customer and Terminal Features

Generate deterministic temporal features after creating the chronological
splits:

```bash
fraud-build-features
```

The command reads `configs/temporal_features.toml` and writes aligned, ignored
CSV files under `data/processed/features/`. Customer and terminal features each
include velocity over one-hour and 24-hour windows, prior mean amount, deviation
from that mean, and time since the entity's previous transaction. Transactions
at the same timestamp cannot see one another. See
`docs/temporal_feature_contract.md` for the exact leakage boundary and
missing-history behavior.

## Evaluate the Fixed XGBoost Baseline

Evaluate validation first using the frozen XGBoost configuration:

```bash
fraud-evaluate-xgboost --split validation
```

Only after the configuration is frozen, produce the one-time test report:

```bash
fraud-evaluate-xgboost --split test
```

The command validates the transaction-ID join, fits all preprocessing and models
on train only, and compares rules, Logistic Regression, and XGBoost under the
same simulated-cost assumptions. The reports are
`docs/xgboost_validation_report.md` and `docs/xgboost_test_report.md`. Neither
report constitutes model promotion. The validation report additionally reads
fixed diagnostic settings from `configs/validation_diagnostics.toml` and reports
Brier score, equal-width reliability bins, and errors by transaction-amount band
and prior-history availability. It does not fit a calibrator or change the frozen
model or threshold. The one-time test report is not rerun for these diagnostics.

## Analyze XGBoost Feature Groups

Compare the base fields, temporal features, and synthetic ID features without
scoring test:

```bash
fraud-analyze-xgboost-features
```

The command reads `configs/xgboost_feature_ablation.toml`, holds the frozen
XGBoost parameters and threshold constant, fits every variant on train, and
writes `docs/xgboost_feature_ablation_report.md`. The best simpler variant must
exceed the full baseline by the versioned absolute AP margin before the report
recommends a bounded feature revision. This is a validation-only screen, not a
significance test or model-promotion rule.

The final terminal-history challenger uses the same command with its predeclared
configuration and a separate validation-only report:

```bash
fraud-analyze-xgboost-features \
  --ablation-config configs/xgboost_terminal_challenger.toml \
  --output docs/xgboost_terminal_challenger_report.md
```

It measured AP 0.0569 versus 0.1077 for the frozen baseline, so terminal history
was rejected and test was not scored.

## Select the Provisional Decision Policy

Fit the frozen XGBoost pipeline on training, score only chronological
validation, and select capacity-feasible approve/review/decline thresholds:

```bash
fraud-select-decision-policy
```

The command reads `configs/decision_policy.toml`,
`configs/decision_reasons.toml`, and the simulated assumptions in
`configs/business_costs.toml`, then writes `docs/decision_policy_report.md`.
Risky decisions receive deterministic reasons derived from current transaction
fields and strictly prior customer history, in configured priority order and
with a score-band fallback when no feature condition matches. These reasons
describe configured patterns, not model causality. The policy layer itself
accepts supplied risk scores and is model-agnostic; XGBoost is only the current
engineering-only provisional score source. The command has no test-split
option.

## Apply the Frozen Single-Transaction Contract

`configs/selected_decision_policy.toml` freezes the provisional 0.93/0.93
thresholds and their validation provenance. The pure
`decide_transaction_from_score` function combines one externally supplied score
and one aligned decision-time feature row with the existing reason layer. It
returns a stable transaction ID, score, decision, ordered reason codes, and
policy version without fitting a model or selecting thresholds.

See `docs/inference_contract.md` for exact boundaries, inputs, outputs, leakage
controls, and current serving limitations.

## Build the Provisional Model Artifact

Fit the unchanged provisional XGBoost pipeline using only the training
transactions and training temporal-feature CSV:

```bash
fraud-build-provisional-model
```

The command atomically writes
`models/xgboost_engineering_provisional.joblib` and its deterministic JSON
metadata sidecar under the ignored `models/` directory. Metadata binds the
artifact to its checksum, version, frozen XGBoost configuration, exact
score-source label, ordered 11-field feature contract, and training hashes and
counts. The checked loader revalidates those current sources before loading.

Joblib artifacts use pickle semantics and may execute code while loading. The
checksum detects accidental changes but is not proof of authenticity, so only
trusted artifacts produced locally by this command may be loaded. See
`docs/model_artifact_contract.md` for the complete lifecycle, leakage, trust,
and portability boundaries. Only the explicit artifact-backed FastAPI factory
loads this trusted local model; the decision-only factory does not.

## Score One Precomputed Record

`ProvisionalModelInput` is the label-free framework contract for one current
transaction plus precomputed, strictly prior customer history. Its ordered
mapping is shared with the existing batch XGBoost scorer, including native
missing-history values and unknown customer or terminal IDs.

`score_provisional_model` accepts that input and an already loaded
`LoadedProvisionalModel`, calls prediction without fitting, and returns the risk
score together with the artifact version, artifact checksum, score-source
label, and XGBoost configuration checksum. This is currently a Python domain
adapter: it does not load the artifact or compute temporal features. The
artifact-backed endpoint calls it only after application startup. See
`docs/inference_contract.md` for the exact boundary.

## Score and Decide One Precomputed Record

`score_and_decide_provisional_transaction` composes the preloaded scorer with
an already loaded selected policy and deterministic reason configuration. It
checks that the artifact and policy describe the same score source, then returns
the score, approve/review/decline decision, ordered reason codes, policy version,
and artifact identity.

The function performs no fitting, artifact loading, configuration loading, or
temporal-state calculation. It remains framework-free; the scored application
factory supplies its already loaded dependencies.

## Run the Decision-only API

Start the local application from the repository root:

```bash
uvicorn fraud_detection.serving.api:app
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation. Submit a
label-free request to `POST /decisions` with a supplied `risk_score`, current
`transaction_amount`, and required past-customer context. For example:

```json
{
  "transaction_id": 42,
  "risk_score": 0.93,
  "transaction_amount": "300.00",
  "customer_tx_count_short_window": 3,
  "customer_amount_mean_prior": "100.00",
  "customer_amount_deviation_from_mean_prior": "200.00",
  "customer_seconds_since_previous": 60
}
```

The endpoint forbids fraud-label fields and other undeclared inputs. It applies
the frozen decision policy only; callers must supply the score and precomputed
past-only context.

## Run the Scored-decision API

Build the trusted local artifact first, then start the explicit factory from
the repository root:

```bash
fraud-build-provisional-model
uvicorn fraud_detection.serving.api:create_scored_app --factory
```

This application preserves `POST /decisions` and adds
`POST /scored-decisions`. The scored route accepts the frozen label-free model
inputs, including precomputed strictly prior customer history:

```json
{
  "transaction_id": 42,
  "tx_amount": "300.00",
  "tx_time_days": 2,
  "tx_datetime": "2018-04-03T06:00:00Z",
  "customer_tx_count_short_window": 3,
  "customer_tx_count_long_window": 7,
  "customer_amount_mean_prior": "100.00",
  "customer_amount_deviation_from_mean_prior": "200.00",
  "customer_seconds_since_previous": 60,
  "customer_id": 12,
  "terminal_id": 8
}
```

The factory validates and loads the artifact, policy, and reason configuration
once. Requests only predict and apply the frozen policy; they never fit a model
or derive live temporal state. Responses include the score, decision, ordered
reasons, policy version, score-source label, artifact version and checksum, and
XGBoost configuration checksum. Fraud labels, supplied risk scores, and other
undeclared fields receive HTTP 422.

## Use the Local Analyst Demo

Start the scored application, then open:

```text
http://127.0.0.1:8000/analyst-demo
```

The page sends the same 11 label-free, precomputed fields accepted by
`POST /scored-decisions`. It displays the risk score, decision, ordered reason
codes, policy version, and artifact identity. Recent submissions exist only in
the current page session and clear on refresh. The page does not calculate
temporal history, store transactions, expose fraud labels, or change policy
thresholds.

## Run the Scored App in a Container

The Docker image runs the artifact-backed factory as a non-root user. It keeps
the trusted model artifact and the two training-provenance files outside the
image as read-only mounts.

Build the image from the repository root:

```bash
docker build --tag fintech-fraud-demo:local .
```

See `docs/container_runbook.md` for the exact four-file mount command, health
check, trust boundary, and verified local run. The container serves
the analyst demo at `http://127.0.0.1:8000/analyst-demo`.

## Benchmark Local Scored-decision Latency

Run the fixed local benchmark after building the trusted artifact:

```bash
fraud-benchmark-scored-api
```

The versioned workload in `configs/serving_latency.toml` creates one checked
application, sends 10 warm-up requests, and measures 200 subsequent
`POST /scored-decisions` calls with nearest-rank percentiles. It atomically
writes `docs/serving_latency_report.md`.

The verified Python 3.13.7 macOS arm64 run measured 76.741 ms for application
factory startup validation, 1.508 ms warmed request p50, and 1.782 ms warmed
request p95. These are in-process FastAPI TestClient measurements, not
production service-level claims. They exclude TCP, Uvicorn, reverse proxies,
network delay, process launch, module imports, concurrency, and live temporal
feature calculation. Re-run the command to measure the current local machine
and artifact.

## Analyze Policy Cost Sensitivity

Reuse one validation score set across the five versioned business-cost and
review-capacity scenarios:

```bash
fraud-analyze-policy-sensitivity
```

The command reads `configs/decision_policy_sensitivity.toml`, preserves the base
assumptions in `configs/business_costs.toml`, and writes
`docs/decision_policy_sensitivity_report.md`. Scenario order is deterministic,
each scenario receives an isolated frozen cost configuration, and the command
has no test-split option.

## Monitor label-free input drift

Compare the fixed train reference with the later validation inputs:

```bash
fraud-monitor-input-drift
```

The command derives quantile bins from train only, keeps missing values in a
separate bin, and writes `docs/input_drift_report.md`. The current report finds
low PSI for amount and customer velocity counts. Prior-history deviation and
recency show high PSI because 100 early training rows have no customer history
while every validation row carries prior history. This is a pipeline effect,
not proof of model decay or a retraining signal.

## Monitor delayed outcomes

Generate the versioned delayed-label example:

```bash
fraud-monitor-performance-example
```

The example keeps pending labels out of performance metrics while reporting
their effect on label coverage. Its fraud recall, false-decline rate, fraud
amount capture, and Brier score come from the illustrative rows in
`examples/delayed_outcomes.csv`, not from model evaluation. See
`docs/performance_monitoring_example.md` for the exact boundary.

## Continuous integration

`.github/workflows/ci.yml` defines three local-to-CI checks:

- tests on Python 3.11, 3.12, and 3.13, plus lint, formatting, and dependency
  checks;
- full data, feature, artifact, monitoring-report, and real scored-app smoke
  reproduction;
- a clean container image build.

The workflow runs on pushes, pull requests, and manual dispatches. See the
repository's Actions page for the current result.

## Repository layout

- `src/fraud_detection/`: reusable data, feature, model, decisioning, serving,
  and monitoring code.
- `tests/`: unit and integration tests.
- `configs/`: versioned parameters and simulated business assumptions.
- `data/raw/`: local source data; never committed to Git.
- `data/processed/`: reproducible derived datasets; never committed to Git.
- `notebooks/`: exploration only, not reusable application logic.
- `docs/`: architecture, data provenance, limitations, and experiment notes.

## Evaluation guardrails

Transactions must be split chronologically, and every feature must use only
information available at decision time. Accuracy will not be used as the
headline metric for this imbalanced problem. Any reported model or financial
result must come from a reproducible experiment in this repository.
