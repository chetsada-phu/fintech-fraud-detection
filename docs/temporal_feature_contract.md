# Temporal Feature Contract

## Purpose

Phase 3 uses deterministic customer- and terminal-history features that are
available at transaction decision time. Run `fraud-build-features` after the
chronological splits exist. Generated feature CSVs are local derived data under
`data/processed/features/` and are intentionally ignored by Git.

## Output Schema

| Column | Meaning |
| --- | --- |
| `TRANSACTION_ID` | Alignment key only; never a model input. |
| `CUSTOMER_TX_COUNT_SHORT_WINDOW` | Prior customer transactions in the configured one-hour interval. |
| `CUSTOMER_TX_COUNT_LONG_WINDOW` | Prior customer transactions in the configured 24-hour interval. |
| `CUSTOMER_AMOUNT_MEAN_PRIOR` | Mean amount across all strictly earlier customer transactions. |
| `CUSTOMER_AMOUNT_DEVIATION_FROM_MEAN_PRIOR` | Current amount minus the prior customer mean. |
| `CUSTOMER_SECONDS_SINCE_PREVIOUS` | Seconds since the customer's most recent earlier timestamp. |
| `TERMINAL_TX_COUNT_SHORT_WINDOW` | Prior terminal transactions in the configured one-hour interval. |
| `TERMINAL_TX_COUNT_LONG_WINDOW` | Prior terminal transactions in the configured 24-hour interval. |
| `TERMINAL_AMOUNT_MEAN_PRIOR` | Mean amount across all strictly earlier terminal transactions. |
| `TERMINAL_AMOUNT_DEVIATION_FROM_MEAN_PRIOR` | Current amount minus the prior terminal mean. |
| `TERMINAL_SECONDS_SINCE_PREVIOUS` | Seconds since the terminal's most recent earlier timestamp. |

The amount mean and deviation use four decimal places from the versioned
configuration. A customer or terminal's first transaction has empty
amount-history and time-since-previous values. Missing history is not silently
replaced with zero.

## Time and Leakage Semantics

- Velocity windows use `[current timestamp - window, current timestamp)`.
  A transaction exactly at the lower boundary is included.
- Transactions sharing the current timestamp are evaluated from the same prior
  state and cannot see one another. The input row order therefore cannot create
  artificial within-timestamp history.
- Customer and terminal histories use the same strict lower-window and
  equal-timestamp semantics.
- Validation features carry forward earlier training transactions. Test
  features carry forward earlier training and validation transactions because
  those events already exist at test decision time.
- `TX_FRAUD` and `TX_FRAUD_SCENARIO` are never read by feature generation.
  Relabeling outcomes or appending future rows cannot alter an earlier feature
  row.

## Reproducibility

For the current seed-42 sample, the command produces 3,569 train, 1,189
validation, and 1,190 test feature rows. The current SHA-256 values are:

- Train: `32e3bdb406bab0140661ea7cd3e5b061f8378c898834164af7eb70240cdd86b7`
- Validation: `6fa62638087a8cc4160616932ddc2ac0c3ae358dbddad184e3e0360237ffc732`
- Test: `60936c24d2f0d638352fe201240de08d6997c1767f66f00c7faced105b38a4e0`

These hashes describe generated synthetic artifacts, not model performance or
production data.

## Main-model Join

The XGBoost workflow reloads each generated feature CSV and requires its ordered
`TRANSACTION_ID` values to match the corresponding validated transaction split
exactly. Missing, duplicate, shifted, or malformed feature rows fail before
model fitting. The identifier is discarded after alignment and is never a
predictor.

The frozen XGBoost baseline continues to use only the original customer-history
fields. Terminal-history fields were evaluated by a separate validation-only
challenger and were not added to the frozen baseline or scored on test.
