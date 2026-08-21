# Transaction Data Contract

## Purpose

Phase 1 uses a small synthetic dataset shaped after the ULB *Reproducible
Machine Learning for Credit Card Fraud Detection* Handbook. It is intended for
learning, automated tests, and pipeline development. It is not real payment
data and must not be presented as representative of a bank's fraud rate or
economics.

## Raw Schema

| Column | Type | Availability and meaning |
| --- | --- | --- |
| `TRANSACTION_ID` | integer | Unique ID assigned after chronological sorting. |
| `TX_DATETIME` | UTC timestamp | Transaction event time, available at decision time. |
| `CUSTOMER_ID` | integer/category | Synthetic customer identifier, available at decision time. |
| `TERMINAL_ID` | integer/category | Synthetic merchant-terminal identifier, available at decision time. |
| `TX_AMOUNT` | decimal(2) | Positive transaction amount, available at decision time. |
| `TX_TIME_SECONDS` | integer | Seconds since the configured start; derived from event time. |
| `TX_TIME_DAYS` | integer | Whole days since the configured start; derived from event time. |
| `TX_FRAUD` | integer 0/1 | Post-event training label. Never use as a model input. |
| `TX_FRAUD_SCENARIO` | integer 0-3 | Simulation-only label detail. Never use as a model input. |

Scenario `0` is legitimate, `1` is a high-amount rule, `2` is a temporarily
compromised terminal, and `3` is a temporarily compromised customer with
inflated transaction amounts.

## Generation and Validation

`configs/data_generation.toml` contains every assumption and the random seed.
From the repository root, run:

```bash
fraud-generate-data
```

The command validates positive amounts, unique chronological IDs, UTC timestamp
consistency, binary labels, and scenario consistency before atomically writing
`data/raw/transactions.csv`. The raw CSV is deliberately ignored by Git.

## Provenance and Licensing

The schema and simulation concepts were informed by the ULB Handbook's
[transaction simulator](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html),
authored by Yann-Aël Le Borgne, Wissam Siblini, Bertrand Lebichot, and Gianluca
Bontempi (2022). The Handbook states that its notebook code is GNU GPLv3 and
its prose and images are CC BY-SA 4.0.

This repository contains a smaller, independently written adaptation and does
not vendor the Handbook notebook. It omits geographic profiles and uses a few
scaled compromise events instead of starting new events every day. A license
for this repository itself has not yet been selected; add one before external
redistribution.

## Leakage Boundary

`TX_FRAUD` and `TX_FRAUD_SCENARIO` are outcomes known only after investigation.
Future feature code must explicitly exclude both columns. Customer and terminal
aggregates must be calculated from earlier transactions only; chronological
feature calculations must never read later rows.

## Chronological Splits

`configs/data_split.toml` versions the target train, validation, and test
fractions. From the repository root, run:

```bash
fraud-split-data
```

The command validates the complete raw dataset, chooses boundaries only between
distinct timestamps, and writes `data/processed/train.csv`,
`data/processed/validation.csv`, and `data/processed/test.csv`. Because equal
timestamps remain together, realized row fractions can differ slightly from the
configured targets. The outputs preserve every raw row exactly once and retain
the two labels for supervised training and evaluation.

Model code must use `MODEL_FEATURE_COLUMNS` from
`fraud_detection.data.splitter`. That contract explicitly excludes
`TRANSACTION_ID`, `TX_FRAUD`, and `TX_FRAUD_SCENARIO`; retaining labels in the
split files does not authorize treating them as model inputs.

## Focused EDA

Run `fraud-profile-data` after splitting to validate the three datasets together
and refresh `docs/eda_report.md`. The deterministic report records measured split
time ranges, fraud counts and rates, amount summaries, customer and terminal
coverage, and fraud-scenario support. Missing scenario support in a split must be
carried forward as an evaluation limitation rather than hidden by aggregate
metrics.

## Phase 2 Baseline Features

The Logistic Regression baseline uses only decision-time fields from
`MODEL_FEATURE_COLUMNS`: transaction amount, elapsed day, cyclical UTC hour,
customer ID, and terminal ID. Numeric preprocessing and one-hot category fitting
use the training split only. Held-out customer or terminal values are ignored by
the encoder rather than causing a failure. Neither fraud label, the transaction
identifier, nor future transactions are model inputs.

The Phase 2 pipeline does not yet calculate customer or terminal history. Those
past-only aggregates are generated separately by the leakage-tested Phase 3
temporal feature pipeline. See `docs/temporal_feature_contract.md` for its
schema, exact window boundaries, missing-history behavior, and cross-split
history rules.
