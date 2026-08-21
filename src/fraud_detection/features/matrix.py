"""Validated joining of base transactions and generated temporal features."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fraud_detection.data.eda import ProcessedSplits
from fraud_detection.data.schema import Transaction
from fraud_detection.features.temporal import (
    FEATURE_FILENAMES,
    TEMPORAL_FEATURE_COLUMNS,
    TemporalFeatureRow,
)


@dataclass(frozen=True, slots=True)
class JoinedFeatureRow:
    """One transaction and its transaction-ID-aligned temporal features."""

    transaction: Transaction
    temporal: TemporalFeatureRow


@dataclass(frozen=True, slots=True)
class JoinedFeatureSplits:
    """Joined model rows grouped by chronological split."""

    train: tuple[JoinedFeatureRow, ...]
    validation: tuple[JoinedFeatureRow, ...]
    test: tuple[JoinedFeatureRow, ...]


def load_joined_feature_splits(
    transactions: ProcessedSplits, feature_directory: Path
) -> JoinedFeatureSplits:
    """Load each generated feature CSV and enforce exact transaction alignment."""
    return JoinedFeatureSplits(
        train=join_transaction_features(
            transactions.train,
            load_temporal_feature_csv(feature_directory / FEATURE_FILENAMES["train"]),
        ),
        validation=join_transaction_features(
            transactions.validation,
            load_temporal_feature_csv(
                feature_directory / FEATURE_FILENAMES["validation"]
            ),
        ),
        test=join_transaction_features(
            transactions.test,
            load_temporal_feature_csv(feature_directory / FEATURE_FILENAMES["test"]),
        ),
    )


def load_temporal_feature_csv(path: Path) -> tuple[TemporalFeatureRow, ...]:
    """Load and type-check one generated temporal feature CSV."""
    with path.open(encoding="utf-8", newline="") as feature_file:
        reader = csv.DictReader(feature_file)
        if tuple(reader.fieldnames or ()) != TEMPORAL_FEATURE_COLUMNS:
            raise ValueError(f"{path}: columns must exactly match the feature contract")
        try:
            rows = tuple(_feature_from_row(row) for row in reader)
        except (InvalidOperation, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: invalid temporal feature value: {error}"
            ) from error
    if not rows:
        raise ValueError(f"{path}: feature CSV must contain at least one row")
    return rows


def join_transaction_features(
    transactions: Sequence[Transaction],
    temporal_features: Sequence[TemporalFeatureRow],
) -> tuple[JoinedFeatureRow, ...]:
    """Join exact ordered IDs and reject missing, duplicate, or shifted rows."""
    if len(transactions) != len(temporal_features):
        raise ValueError("transaction and temporal feature row counts must match")
    joined = []
    seen_ids = set()
    for transaction, temporal in zip(transactions, temporal_features, strict=True):
        if temporal.transaction_id in seen_ids:
            raise ValueError("temporal feature transaction IDs must be unique")
        seen_ids.add(temporal.transaction_id)
        if transaction.transaction_id != temporal.transaction_id:
            raise ValueError("temporal feature transaction IDs must align in order")
        joined.append(JoinedFeatureRow(transaction=transaction, temporal=temporal))
    return tuple(joined)


def _feature_from_row(row: Mapping[str, str]) -> TemporalFeatureRow:
    if any(row.get(column) is None for column in TEMPORAL_FEATURE_COLUMNS):
        raise ValueError("every temporal feature column must be present")
    transaction_id = _parse_non_negative_int(row["TRANSACTION_ID"], "TRANSACTION_ID")
    customer = _parse_history_fields(row, "CUSTOMER")
    terminal = _parse_history_fields(row, "TERMINAL")
    return TemporalFeatureRow(
        transaction_id=transaction_id,
        customer_tx_count_short_window=customer[0],
        customer_tx_count_long_window=customer[1],
        customer_amount_mean_prior=customer[2],
        customer_amount_deviation_from_mean_prior=customer[3],
        customer_seconds_since_previous=customer[4],
        terminal_tx_count_short_window=terminal[0],
        terminal_tx_count_long_window=terminal[1],
        terminal_amount_mean_prior=terminal[2],
        terminal_amount_deviation_from_mean_prior=terminal[3],
        terminal_seconds_since_previous=terminal[4],
    )


def _parse_history_fields(
    row: Mapping[str, str], entity: str
) -> tuple[int, int, Decimal | None, Decimal | None, int | None]:
    short_column = f"{entity}_TX_COUNT_SHORT_WINDOW"
    long_column = f"{entity}_TX_COUNT_LONG_WINDOW"
    mean_column = f"{entity}_AMOUNT_MEAN_PRIOR"
    deviation_column = f"{entity}_AMOUNT_DEVIATION_FROM_MEAN_PRIOR"
    previous_column = f"{entity}_SECONDS_SINCE_PREVIOUS"
    short_count = _parse_non_negative_int(row[short_column], short_column)
    long_count = _parse_non_negative_int(row[long_column], long_column)
    if short_count > long_count:
        raise ValueError(
            f"{entity.lower()} short-window count cannot exceed long-window count"
        )
    prior_mean = _parse_optional_decimal(row[mean_column])
    deviation = _parse_optional_decimal(row[deviation_column])
    seconds_since_previous = _parse_optional_non_negative_int(
        row[previous_column], previous_column
    )
    missing_history = (
        prior_mean is None,
        deviation is None,
        seconds_since_previous is None,
    )
    if len(set(missing_history)) != 1:
        raise ValueError(
            f"prior {entity.lower()}-history fields must be missing together"
        )
    if prior_mean is not None and (not prior_mean.is_finite() or prior_mean <= 0):
        raise ValueError(f"{mean_column} must be finite and positive")
    if deviation is not None and not deviation.is_finite():
        raise ValueError(f"{deviation_column} must be finite")
    return (
        short_count,
        long_count,
        prior_mean,
        deviation,
        seconds_since_previous,
    )


def _parse_non_negative_int(value: str, column: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{column} must be non-negative")
    return parsed


def _parse_optional_non_negative_int(value: str, column: str) -> int | None:
    if value == "":
        return None
    return _parse_non_negative_int(value, column)


def _parse_optional_decimal(value: str) -> Decimal | None:
    if value == "":
        return None
    return Decimal(value)
