# Phase 3 XGBoost Validation Report

Generated reproducibly by `fraud-evaluate-xgboost`. All models fit only
the chronological training split (3,569 rows,
20 fraud labels). The test split is not scored by this validation report.

## Frozen XGBoost Configuration

- Inputs: Phase 2 decision-time fields plus the validated past-only
  customer temporal features. Transaction ID and fraud labels are excluded.
- Missing prior history remains missing and is handled natively by XGBoost.
- Customer and terminal IDs are one-hot encoded using training data only.
- Trees 200, maximum depth 3, learning rate 0.05, minimum child weight 5.
- Row sample 0.80, column sample 0.80, L1 0, L2 1.
- Training-only class ratio weight: 177.45; random state 42; one worker for deterministic fitting.
- Fixed binary flag threshold: 0.50.

## Validation Comparison

Average precision is the primary ranking metric. Accuracy is omitted.

| Dataset | Baseline | AP | ROC-AUC | Flag rate | Precision | Recall | FPR | Fraud amount captured | Cost per 1,000 | Within capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Validation | Rules | 0.0083 | 0.4949 | 11.02% | 0.76% | 10.00% | 11.03% | 68.72% | 782.56 | no |
| Validation | Logistic Regression | 0.1056 | 0.3083 | 0.59% | 14.29% | 10.00% | 0.51% | 68.72% | 261.11 | yes |
| Validation | XGBoost | 0.1077 | 0.4573 | 0.34% | 25.00% | 10.00% | 0.25% | 68.72% | 248.49 | yes |

## Validation-only Calibration Diagnostics

These diagnostics use the frozen validation scores after scoring. They do
not fit a calibrator or change the model configuration or threshold.
Because the validation split is highly imbalanced, the Brier score can be
dominated by legitimate transactions and must be read with the reliability
table and ranking metrics.

- Brier score: 0.0095.
- Expected calibration error: 0.0085 across 5 fixed equal-width bins.

| Score interval | Rows | Frauds | Mean score | Observed fraud rate | Absolute gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0.00, 0.20) | 1,185 | 9 | 0.0011 | 0.76% | 0.0065 |
| [0.20, 0.40) | 0 | 0 | n/a | n/a | n/a |
| [0.40, 0.60) | 0 | 0 | n/a | n/a | n/a |
| [0.60, 0.80) | 0 | 0 | n/a | n/a | n/a |
| [0.80, 1.00] | 4 | 1 | 0.8781 | 25.00% | 0.6281 |

## Validation Segment Error Analysis

Amount bands are fixed in `configs/validation_diagnostics.toml`; they
are not derived from validation or test outcomes. Errors use the frozen
XGBoost threshold of 0.50.

### Transaction amount

| Segment | Rows | Frauds | Fraud rate | Mean score | Flagged | Flag rate | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Amount <= 50.00 | 682 | 7 | 1.03% | 0.0007 | 0 | 0.00% | 0 | 0 | 7 | n/a | 0.00% |
| 50.00 < amount <= 100.00 | 350 | 2 | 0.57% | 0.0016 | 0 | 0.00% | 0 | 0 | 2 | n/a | 0.00% |
| 100.00 < amount <= 220.00 | 156 | 0 | 0.00% | 0.0182 | 3 | 1.92% | 0 | 3 | 0 | 0.00% | n/a |
| Amount > 220.00 | 1 | 1 | 100.00% | 0.9350 | 1 | 100.00% | 1 | 0 | 0 | 100.00% | 100.00% |

### Prior customer history

Prior history is available only when the validated prior mean, amount
deviation, and seconds-since-previous fields are present together.

| Segment | Rows | Frauds | Fraud rate | Mean score | Flagged | Flag rate | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Missing prior history | 0 | 0 | n/a | n/a | 0 | n/a | 0 | 0 | 0 | n/a | n/a |
| Prior history available | 1,189 | 10 | 0.84% | 0.0041 | 4 | 0.34% | 1 | 3 | 9 | 25.00% | 10.00% |

No validation rows lack prior history, so behavior for that
segment cannot be assessed from this split.

## Provisional Challenger Decision

**Decision: retain XGBoost as a provisional challenger; do not promote
it as the selected main model.**

- Validation AP is 0.1077 versus 0.1056 for Logistic Regression (difference +0.0021), which is not a material advantage on this small split.
- Validation ROC-AUC is 0.4573 and recall at the frozen threshold is 10.00%; ranking and capture evidence remain weak.
- Calibration and segment tables are descriptive and have sparse fraud
  support. They do not justify fitting a calibrator or promoting the model
  from the same validation labels.
- The one-time test result is retained as a frozen report and is not used
  to revise this decision or any configuration.

## Interpretation Limits

- This is one fixed baseline, not a hyperparameter search or a claim of
  model promotion.
- The synthetic sample has few and uneven fraud scenarios, so measured
  differences can be unstable.
- The fixed threshold remains a baseline and is not optimized for the
  simulated manual-review constraint.
- Calibration and segment results describe validation only; no
  calibrator or threshold was fitted from these labels.
