# Phase 2 Rule Baseline

Generated reproducibly by `fraud-evaluate-rules` from the validated
chronological splits under `data/processed/`. Rules read decision-time
fields only; fraud labels are used after scoring for evaluation.

## Versioned Rules

- `HIGH_TRANSACTION_AMOUNT`: flag `TX_AMOUNT` strictly greater than 220.00.
- `UNUSUAL_TRANSACTION_HOUR`: flag UTC hours at or after 23:00 and before 06:00.
- A transaction is flagged when either rule fires. This binary flag is a
  baseline, not a final approve/review/decline policy.

## Chronological Evaluation

Accuracy is intentionally omitted because fraud is highly imbalanced.

| Dataset | Rows | Frauds | Flagged | Flag rate | Precision | Recall | False-positive rate | Fraud amount captured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 1,189 | 10 | 131 | 11.02% | 0.76% | 10.00% | 11.03% | 68.72% |
| Test | 1,190 | 10 | 150 | 12.61% | 1.33% | 20.00% | 12.54% | 70.71% |

## Reason-Code Counts

Counts are rule activations, so a transaction can contribute to both
reason codes.

| Dataset | Reason code | Activations |
| --- | --- | ---: |
| Validation | `HIGH_TRANSACTION_AMOUNT` | 1 |
| Validation | `UNUSUAL_TRANSACTION_HOUR` | 130 |
| Test | `HIGH_TRANSACTION_AMOUNT` | 2 |
| Test | `UNUSUAL_TRANSACTION_HOUR` | 148 |

## Fraud-Scenario Support

Scenario labels are simulation-only evaluation details and never rule
inputs.

| Dataset | Scenario 0 | Scenario 1 | Scenario 2 | Scenario 3 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1,179 | 0 | 9 | 1 |
| Test | 1,180 | 2 | 8 | 0 |

## Interpretation Limits

- The amount threshold mirrors a synthetic-generator assumption; its
  measured performance is not evidence for a real payment system.
- The overnight UTC rule is an illustrative heuristic, not a learned
  customer-specific behavior pattern.
- Fraud counts and scenario support are small and uneven across splits.
- Threshold selection, manual-review capacity, and three-way decisions
  remain future decisioning work.
