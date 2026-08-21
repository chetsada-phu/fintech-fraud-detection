# Phase 1 Focused EDA

Generated reproducibly by `fraud-profile-data` from the chronological
files under `data/processed/`. All values below are measured from the
current synthetic sample; they are not production fraud benchmarks.

## Split Summary

| Dataset | Rows | Start (UTC) | End (UTC) | Frauds | Fraud rate |
| --- | ---: | --- | --- | ---: | ---: |
| Train | 3,569 | 2018-04-01T00:00:51+00:00 | 2018-04-19T06:24:40+00:00 | 20 | 0.5604% |
| Validation | 1,189 | 2018-04-19T06:27:51+00:00 | 2018-04-24T22:41:06+00:00 | 10 | 0.8410% |
| Test | 1,190 | 2018-04-24T22:50:48+00:00 | 2018-04-30T23:25:39+00:00 | 10 | 0.8403% |
| Overall | 5,948 | 2018-04-01T00:00:51+00:00 | 2018-04-30T23:25:39+00:00 | 40 | 0.6725% |

## Transaction Amount Summary

Nearest-rank is used for P95. Amounts retain the synthetic currency's
two-decimal precision.

| Dataset | Minimum | Median | Mean | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.14 | 44.20 | 54.10 | 136.19 | 248.60 |
| Validation | 0.19 | 42.54 | 52.77 | 127.84 | 605.10 |
| Test | 0.08 | 41.88 | 53.55 | 135.36 | 246.10 |
| Overall | 0.08 | 43.21 | 53.72 | 134.48 | 605.10 |

## Entity Coverage

| Dataset | Unique customers | Unique terminals |
| --- | ---: | ---: |
| Train | 100 | 50 |
| Validation | 97 | 50 |
| Test | 98 | 50 |
| Overall | 100 | 50 |

## Fraud Scenario Counts

Scenario 0 is legitimate; scenarios 1-3 are synthetic fraud mechanisms
defined in `docs/data_contract.md`.

| Dataset | Scenario 0 | Scenario 1 | Scenario 2 | Scenario 3 |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,549 | 5 | 10 | 5 |
| Validation | 1,179 | 0 | 9 | 1 |
| Test | 1,180 | 2 | 8 | 0 |
| Overall | 5,908 | 7 | 27 | 6 |

## Data Quality and Leakage Checks

- All 5,948 transaction IDs are contiguous and appear exactly once across the three splits.
- Train, validation, and test timestamps are strictly ordered with no boundary overlap.
- Every row passes the raw schema checks for timestamps, amounts, IDs, and label consistency.
- The post-event fields `TX_FRAUD` and `TX_FRAUD_SCENARIO` remain evaluation labels and are excluded from the model-feature contract.

## Interpretation Limits

- The data is synthetic and scaled down for pipeline development.
- Fraud counts are small, so split-level rates can vary materially and must not be interpreted as bank or market prevalence.
- Fraud-scenario support is uneven: Validation has no scenario 1; Test has no scenario 3. Later evaluation must report scenario support and avoid treating split-level rates as stable estimates.
- These descriptive measurements do not establish model performance, financial impact, fairness, or production readiness.
