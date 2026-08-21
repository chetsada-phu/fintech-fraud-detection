# Frozen Single-Transaction Decision Contract

## Status and Scope

This is a framework-free engineering contract for scoring one precomputed,
label-free model input and for converting a risk score into a reproducible
decision and explanation. One FastAPI factory wraps only the supplied-score
decision contract. A separate artifact-backed factory also exposes the composed
scored-decision contract for precomputed model inputs.

The score source remains the unpromoted XGBoost engineering provisional source.
The caller remains responsible for supplying strictly prior temporal history;
this contract does not maintain customer state or derive live aggregates.

## Label-free Model Input

`ProvisionalModelInput` accepts one precomputed record containing:

- transaction ID for result correlation;
- positive transaction amount, non-negative transaction day index, and a
  timezone-aware UTC transaction timestamp;
- non-negative customer and terminal IDs;
- customer transaction counts over the frozen short and long windows; and
- the all-present or all-missing prior customer amount mean, amount deviation,
  and seconds-since-previous fields.

It contains no `TX_FRAUD` or `TX_FRAUD_SCENARIO` field. Terminal temporal
history is also absent because it was rejected by the validation-only
challenger and is not part of the frozen baseline.

The input maps to the exact ordered 11-field artifact contract: amount, day
index, derived time-of-day sine and cosine, five customer-history values, then
the encoded customer and terminal identifiers. The existing batch XGBoost path
uses this same mapper, preventing a separate serving-time feature order.

## Pure Model Scorer

`score_provisional_model` accepts an already loaded and validated
`LoadedProvisionalModel` plus one `ProvisionalModelInput`. It calls
`predict_proba` once and returns `ProvisionalModelScore` with:

- transaction ID and risk score;
- score-source label;
- artifact version and SHA-256; and
- frozen XGBoost configuration SHA-256.

The scorer does not load an artifact, fit a pipeline, calculate temporal state,
apply thresholds, create reason codes, or call FastAPI. Missing prior history
uses the existing XGBoost native missing-value path, while unseen customer and
terminal IDs use the fitted one-hot encoder's unknown-category handling.

## Pure Scored-decision Use Case

`score_and_decide_provisional_transaction` accepts one already loaded artifact,
one `ProvisionalModelInput`, and already loaded selected-policy and reason
configurations. It first requires the artifact and policy score-source labels to
match, then delegates to `score_provisional_model` and
`decide_context_from_score` without reimplementing either contract.

`ScoredTransactionDecisionResult` returns the transaction ID, score, decision,
ordered reason codes, policy version, score-source label, artifact version and
SHA-256, and XGBoost configuration SHA-256. This makes the decision traceable to
the exact local artifact that produced its score.

The use case does not load the artifact or configuration, fit the pipeline, or
compute temporal state. Those immutable dependencies must be prepared outside
the function and supplied explicitly; the scored FastAPI factory is one adapter
that performs that preparation.

## Versioned Policy

`configs/selected_decision_policy.toml` freezes the validation-selected result:

| Field | Value |
| --- | --- |
| Policy version | `validation-provisional-v1` |
| Selection split | `validation` |
| Review threshold | `0.93` |
| Decline threshold | `0.93` |

The equal thresholds intentionally create no review band. A score below `0.93`
is approved; a score at or above `0.93` is declined because decline takes
precedence at equality. These settings are provisional and must not be revised
using test outcomes.

## Pure Function

`decide_transaction_from_score` accepts:

- one finite risk score within the inclusive range from zero to one;
- one transaction-ID-aligned `JoinedFeatureRow` containing the current
  transaction and strictly past temporal context;
- the frozen selected-policy configuration; and
- the existing deterministic reason configuration.

It returns a frozen `TransactionDecisionResult` with these stable fields:

| Field | Meaning |
| --- | --- |
| `transaction_id` | Identifier copied from the aligned transaction row |
| `risk_score` | The validated externally supplied score |
| `decision` | `approve`, `review`, or `decline` |
| `reason_codes` | Ordered deterministic feature reasons or a score-band fallback |
| `policy_version` | Version of the fixed thresholds used for the decision |

Approved decisions have no reason codes. Risky decisions receive configured
feature-derived reasons when a condition matches; otherwise they retain the
appropriate score-band fallback. Reasons describe configured patterns and do
not establish model causality or feature attribution.

## Leakage and Serving Boundary

The internal `JoinedFeatureRow` retains fraud labels because it is shared with
offline evaluation code. The inference result is tested to remain identical
when `TX_FRAUD` and `TX_FRAUD_SCENARIO` change, and the reason layer does not
read either field.

The public `DecisionRequest` schema omits and forbids those label fields. It
accepts only the transaction ID, supplied risk score, current amount, short
customer-velocity count, and the three aligned prior-history fields. Impossible
values, partial missing history, and undeclared inputs receive HTTP 422.

`POST /decisions` loads `configs/selected_decision_policy.toml` and
`configs/decision_reasons.toml` once when its application instance is created,
then delegates each request to the framework-free contract. The endpoint still
requires a supplied risk score and precomputed history, so it is deliberately
named as a decision endpoint and is not a raw-transaction scoring endpoint.

`create_scored_app` loads the same policy and reason configuration plus the
checked trusted-local provisional artifact once when its application instance
is created. It preserves `POST /decisions` and adds
`POST /scored-decisions`, whose strict `ScoredDecisionRequest` mirrors the 11
precomputed model inputs and forbids supplied scores, fraud labels, and all
undeclared fields. Each scored request delegates to
`score_and_decide_provisional_transaction` without fitting or loading. Its
response returns the decision fields plus the exact score-source label,
artifact version and SHA-256, and XGBoost configuration SHA-256.

Run the scored factory from the repository root after building the artifact:

```bash
uvicorn fraud_detection.serving.api:create_scored_app --factory
```

Neither endpoint computes or persists live customer history. Therefore the
scored route accepts precomputed decision-time features, not a raw transaction,
and must not be described as a production fraud service.

## Local Latency Boundary

`fraud-benchmark-scored-api` loads the fixed contract in
`configs/serving_latency.toml`, times one `create_scored_app` call, warms that
same application, and then records nearest-rank p50 and p95 over 200 successful
in-process `POST /scored-decisions` calls. Every response is checked against the
loaded policy and artifact identity. The report is written atomically to
`docs/serving_latency_report.md` in a stable structure; numeric timings are
intentionally machine-specific.

The request interval covers FastAPI TestClient's in-process
`client.post(...)` call and excludes response JSON validation. It also excludes
process and module startup, TCP, Uvicorn, proxies, network delay, concurrency,
and live temporal-state work. The separate startup value covers the scored app
factory's configuration, artifact-integrity, and training-provenance checks.
These results are local engineering measurements, not throughput evidence,
production capacity, or a service-level objective.
