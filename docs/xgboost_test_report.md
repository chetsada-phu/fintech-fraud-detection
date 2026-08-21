# Phase 3 XGBoost Test Report

Generated reproducibly by `fraud-evaluate-xgboost`. All models fit only
the chronological training split (3,569 rows,
20 fraud labels). This is the one-time test report produced after freezing the configuration.

## Frozen XGBoost Configuration

- Inputs: Phase 2 decision-time fields plus the validated past-only
  customer temporal features. Transaction ID and fraud labels are excluded.
- Missing prior history remains missing and is handled natively by XGBoost.
- Customer and terminal IDs are one-hot encoded using training data only.
- Trees 200, maximum depth 3, learning rate 0.05, minimum child weight 5.
- Row sample 0.80, column sample 0.80, L1 0, L2 1.
- Training-only class ratio weight: 177.45; random state 42; one worker for deterministic fitting.
- Fixed binary flag threshold: 0.50.

## Test Comparison

Average precision is the primary ranking metric. Accuracy is omitted.

| Dataset | Baseline | AP | ROC-AUC | Flag rate | Precision | Recall | FPR | Fraud amount captured | Cost per 1,000 | Within capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Test | Rules | 0.0094 | 0.5373 | 12.61% | 1.33% | 20.00% | 12.54% | 70.71% | 794.61 | no |
| Test | Logistic Regression | 0.0202 | 0.5175 | 0.34% | 0.00% | 0.00% | 0.34% | 0.00% | 577.92 | yes |
| Test | XGBoost | 0.1724 | 0.3975 | 0.25% | 66.67% | 20.00% | 0.08% | 70.71% | 176.96 | yes |

## Interpretation Limits

- This is one fixed baseline, not a hyperparameter search or a claim of
  model promotion.
- The synthetic sample has few and uneven fraud scenarios, so measured
  differences can be unstable.
- The fixed threshold is not calibrated or optimized for the simulated
  manual-review constraint.
- Segment error analysis, calibration, and validation-only tuning remain
  unfinished Phase 3 work.
