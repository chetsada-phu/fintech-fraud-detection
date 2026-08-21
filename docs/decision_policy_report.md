# Phase 4 Provisional Decision Policy

Generated reproducibly by `fraud-select-decision-policy`. The existing
XGBoost pipeline fits training (3,569 rows, 20 fraud labels) and scores only chronological
validation (1,189 rows, 10 fraud labels) for threshold selection.
The one-time test split is not scored by this command.

## Score-source Status

- Score source: **XGBoost engineering-only provisional score source**.
- XGBoost remains unpromoted. These scores support decision-policy
  engineering only and are not a model-performance claim.
- Frozen XGBoost config SHA-256: `ad34facdde11a42073bc81964693a9e785708d5b7d036cf9ffee20ad3e81ca62`.

## Versioned Selection Contract

- Selection split: `validation` only.
- Fixed threshold grid step: 0.01 across the
  inclusive score range from 0.00 to 1.00.
- Candidate thresholds: 101; ordered
  review/decline pairs evaluated: 5,151.
- Primary objective: minimum simulated total operating cost while the
  validation review rate is at most 5.00%.
- Cost ties prefer fewer false declines, then fewer reviews, then fewer
  declines, and finally higher thresholds.
- Boundary semantics: score below the review threshold is `approve`;
  score at or above review and below decline is `review`; score at or
  above decline is `decline`. If thresholds are equal, decline takes
  precedence and the review band is empty.

## Selected Validation Policy

- Review threshold: **0.93**.
- Decline threshold: **0.93**.

| Decision | Rows | Rate | Score-band fallback |
| --- | ---: | ---: | --- |
| Approve | 1,188 | 99.92% | none |
| Review | 0 | 0.00% | `RISK_SCORE_REVIEW` |
| Decline | 1 | 0.08% | `RISK_SCORE_DECLINE` |

Manual-review capacity: **satisfied** (0.00% observed versus 5.00% maximum).

## Deterministic Decision Reasons

Feature reasons use only the current transaction and aligned strictly
past customer features. Fraud labels, fraud scenarios, and future rows
are not inputs. These reasons describe configured conditions, not model
causality or feature attribution.

- Priority order: `HIGH_AMOUNT_VS_CUSTOMER_BASELINE`, `HIGH_TRANSACTION_AMOUNT`, `UNUSUAL_TRANSACTION_VELOCITY`, `LIMITED_CUSTOMER_HISTORY`.
- At most 3 feature reasons are emitted per risky decision.
- A score-band review or decline code is retained only when no feature
  condition matches.
- Risky decisions: 1; feature-explained: 1; score-band fallbacks: 0.

| Reason code | Exact configured condition | Activations |
| --- | --- | ---: |
| `HIGH_AMOUNT_VS_CUSTOMER_BASELINE` | amount at least 3.00x prior customer mean | 1 |
| `HIGH_TRANSACTION_AMOUNT` | amount strictly above 220.00 | 1 |
| `UNUSUAL_TRANSACTION_VELOCITY` | at least 3 prior customer transactions in the short window | 0 |
| `LIMITED_CUSTOMER_HISTORY` | prior customer-history fields are all missing | 0 |
| `RISK_SCORE_REVIEW` | review score band; used only as fallback | 0 |
| `RISK_SCORE_DECLINE` | decline score band; used only as fallback | 0 |

## Simulated Validation Cost

These values use portfolio assumptions, not bank economics.

- Missed-fraud loss: 275.46.
- Manual-review cost: 0.00.
- False-decline cost: 0.00 across
  0 legitimate declines
  (0.00% of legitimate validation rows).
- Total: 275.46; cost per 1,000 transactions:
  231.67.
- Fraud transaction amount intercepted by review or decline:
  605.10.

## Interpretation Limits

- Thresholds were selected against the same small validation labels used
  to report this cost. They are provisional engineering settings, not an
  unbiased performance estimate or a production policy.
- The simulated calculation assumes every reviewed fraud is intercepted
  and charges one review cost. Real review outcomes and queue timing are
  not modeled.
- Feature-derived reasons identify configured transaction patterns; they
  do not prove why the model produced its score or establish causality.
- The score source remains the versioned
  `XGBoost engineering-only provisional score source`. The frozen test report is not reused to
  select or revise these thresholds.
