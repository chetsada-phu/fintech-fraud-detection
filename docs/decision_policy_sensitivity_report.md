# Phase 4 Decision-policy Cost Sensitivity

Generated reproducibly by `fraud-analyze-policy-sensitivity`. The frozen
XGBoost pipeline fits training (3,569 rows, 20 fraud labels), then produces one shared score set
for chronological validation (1,189 rows, 10 fraud labels). Every scenario reuses those
scores. The one-time test split is not scored by this command.

## Score-source and Isolation Contract

- Score source: **XGBoost engineering-only provisional score source**.
- XGBoost remains unpromoted and supports engineering analysis only.
- Frozen XGBoost config SHA-256: `ad34facdde11a42073bc81964693a9e785708d5b7d036cf9ffee20ad3e81ca62`.
- Scenario order is fixed in `configs/decision_policy_sensitivity.toml`;
  `base` is evaluated first.
- Every scenario creates a separate frozen `BusinessCostConfig` from the
  base assumptions. The base configuration is not mutated.
- Threshold selection uses validation only and the unchanged 0.01 grid.

## Validation Sensitivity

All monetary values are simulated portfolio assumptions. Cost values
should be interpreted within each scenario, not as production economics.

| Scenario | Fraud loss x | Review cost | False-decline cost | Capacity | Review threshold | Decline threshold | Approve | Review | Decline | Review rate | False declines | Captured amount | Cost per 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base assumptions | 1.00 | 5.00 | 25.00 | 5.00% | 0.93 | 0.93 | 1,188 | 0 | 1 | 0.00% | 0 | 605.10 | 231.67 |
| Higher fraud loss | 2.00 | 5.00 | 25.00 | 5.00% | 0.93 | 0.93 | 1,188 | 0 | 1 | 0.00% | 0 | 605.10 | 463.35 |
| Lower review cost | 1.00 | 1.00 | 25.00 | 5.00% | 0.93 | 0.93 | 1,188 | 0 | 1 | 0.00% | 0 | 605.10 | 231.67 |
| Higher false-decline cost | 1.00 | 5.00 | 100.00 | 5.00% | 0.93 | 0.93 | 1,188 | 0 | 1 | 0.00% | 0 | 605.10 | 231.67 |
| Tighter review capacity | 1.00 | 5.00 | 25.00 | 1.00% | 0.93 | 0.93 | 1,188 | 0 | 1 | 0.00% | 0 | 605.10 | 231.67 |

## Stability Summary

- Unique selected threshold pairs: 1 across 5 scenarios.
- Unique approve/review/decline mixes: 1 across 5 scenarios.
- Every selected policy satisfies its scenario-specific manual-review
  capacity constraint.

- Base `Base assumptions`: review threshold 0.93, decline threshold 0.93,
  and action mix 1,188/0/1.

- `Higher fraud loss`: thresholds unchanged; action mix unchanged.
- `Lower review cost`: thresholds unchanged; action mix unchanged.
- `Higher false-decline cost`: thresholds unchanged; action mix unchanged.
- `Tighter review capacity`: thresholds unchanged; action mix unchanged.

## Interpretation Limits

- Every scenario selects and reports thresholds on the same small
  validation labels. This describes assumption sensitivity, not
  unbiased performance or production robustness.
- Lower cost in one scenario is not directly comparable with lower
  cost in another because the simulated cost scales differ.
- Review is modeled as intercepting fraud at one fixed review cost;
  queue timing, reviewer error, and delayed outcomes are not modeled.
- The score source remains an engineering-only provisional XGBoost
  challenger. Test evidence remains frozen and is not reused here.
