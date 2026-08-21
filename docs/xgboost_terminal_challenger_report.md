# Phase 3 XGBoost Feature Ablation

Generated reproducibly by `fraud-analyze-xgboost-features`. Every variant
fits the same chronological training split (3,569 rows,
20 fraud labels) and scores only validation
(1,189 rows, 10 fraud labels).
The one-time test split is not scored by this command.

## Fixed Experiment Contract

- Frozen XGBoost config SHA-256: `ad34facdde11a42073bc81964693a9e785708d5b7d036cf9ffee20ad3e81ca62`.
- Trees 200, depth 3, learning rate 0.05, random state 42, and fixed threshold 0.50 for every variant.
- Base features: transaction amount, elapsed day, and cyclical UTC hour.
- Temporal features: one-hour and 24-hour customer counts, prior amount
  mean and deviation, and seconds since the previous customer transaction.
- Terminal temporal features mirror the customer history contract for
  each terminal using strictly earlier transactions only.
- Synthetic IDs: one-hot encoded customer and terminal identifiers.
- Only feature inclusion changes; no hyperparameter, threshold, or
  calibration search is performed.

## Validation Comparison

| Variant | Inputs | Rows | Frauds | AP | ROC-AUC | Flagged | Flag rate | TP | FP | FN | Precision | Recall | FPR | Cost per 1,000 | Within capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Portable customer history | Base + customer temporal | 1,189 | 10 | 0.1083 | 0.4844 | 4 | 0.34% | 1 | 3 | 9 | 25.00% | 10.00% | 0.25% | 248.49 | yes |
| Portable customer + terminal history | Base + customer temporal + terminal temporal | 1,189 | 10 | 0.0569 | 0.3696 | 4 | 0.34% | 1 | 3 | 9 | 25.00% | 10.00% | 0.25% | 248.49 | yes |
| Frozen full baseline | Base + customer temporal + synthetic IDs | 1,189 | 10 | 0.1077 | 0.4573 | 4 | 0.34% | 1 | 3 | 9 | 25.00% | 10.00% | 0.25% | 248.49 | yes |

## Feature Direction Decision

The versioned screen requires at least +0.0100
absolute validation AP over the frozen full baseline before recommending
a bounded feature revision.

**Decision: this ablation does not justify a bounded feature revision. Retain the frozen full baseline only as an unpromoted provisional challenger and do not score test.**

- Best simpler variant: `Portable customer history` with AP 0.1083.
- Frozen full baseline AP: 0.1077.
- Absolute AP difference: +0.0006.
- This validation-selected comparison has only 10 fraud labels and is not a significance
  test or model-promotion decision.

## Interpretation Limits

- Feature groups are compared under one fixed model configuration; a
  result may reflect interactions with that configuration.
- Synthetic customer and terminal IDs are dataset-specific categories,
  not portable behavioral risk features.
- The small and uneven validation fraud support makes differences
  unstable. Test evidence remains frozen and is not reused here.
