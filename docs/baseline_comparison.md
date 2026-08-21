# Phase 2 Baseline Comparison

Generated reproducibly by `fraud-compare-baselines`. Logistic Regression
fits only the chronological training split (3,569 rows,
20 fraud labels). Validation and test labels are used
only after scoring. No held-out hyperparameter search is performed.

## Logistic Regression Configuration

- Decision-time inputs: transaction amount, elapsed day, cyclical UTC
  hour, customer ID, and terminal ID.
- Numeric inputs are standardized using training data only; IDs are
  one-hot encoded with unseen held-out values ignored.
- Solver `liblinear`, class weight `balanced`, C=1, maximum iterations 1,000, random state 42.
- Fixed binary flag threshold: 0.50.

## Simulated Business Assumptions

- Missed fraud loss: transaction amount x 1.00.
- Manual review cost: 5.00 per flag.
- Maximum manual-review rate: 5.00%.
- Reserved false-decline cost: 25.00; not used
  until a three-way decision policy exists.
- Costs are illustrative portfolio assumptions, not bank economics.

## Held-out Comparison

Average precision is the primary ranking metric. Accuracy is omitted.
The rule baseline's binary flags are treated as scores of zero or one.

| Dataset | Baseline | AP | ROC-AUC | Flag rate | Precision | Recall | FPR | Fraud amount captured | Cost per 1,000 | Within capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Validation | Rules | 0.0083 | 0.4949 | 11.02% | 0.76% | 10.00% | 11.03% | 68.72% | 782.56 | no |
| Validation | Logistic Regression | 0.1056 | 0.3083 | 0.59% | 14.29% | 10.00% | 0.51% | 68.72% | 261.11 | yes |
| Test | Rules | 0.0094 | 0.5373 | 12.61% | 1.33% | 20.00% | 12.54% | 70.71% | 794.61 | no |
| Test | Logistic Regression | 0.0202 | 0.5175 | 0.34% | 0.00% | 0.00% | 0.34% | 0.00% | 577.92 | yes |

## Interpretation Limits

- The Logistic Regression probabilities are not calibrated; balanced
  class weights intentionally change the fitted class emphasis.
- Customer and terminal IDs are synthetic categories, not portable
  behavioral risk features.
- Fraud counts and scenario support are small and uneven across splits.
- The fixed 0.50 threshold is a baseline, not a business-optimized
  approve/review/decline policy.
- Test results are a one-time report and must not be used to revise this
  configuration. Future choices should use training and validation only.
- Neither baseline is automatically promoted from this comparison.
