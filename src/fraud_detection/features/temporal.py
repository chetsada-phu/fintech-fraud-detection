"""Past-only customer and terminal features in strict chronological time."""

from __future__ import annotations

import csv
import os
import tomllib
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from itertools import groupby
from pathlib import Path
from typing import Final

from fraud_detection.data.eda import ProcessedSplits, load_processed_splits
from fraud_detection.data.schema import Transaction

TEMPORAL_FEATURE_COLUMNS: Final = (
    "TRANSACTION_ID",
    "CUSTOMER_TX_COUNT_SHORT_WINDOW",
    "CUSTOMER_TX_COUNT_LONG_WINDOW",
    "CUSTOMER_AMOUNT_MEAN_PRIOR",
    "CUSTOMER_AMOUNT_DEVIATION_FROM_MEAN_PRIOR",
    "CUSTOMER_SECONDS_SINCE_PREVIOUS",
    "TERMINAL_TX_COUNT_SHORT_WINDOW",
    "TERMINAL_TX_COUNT_LONG_WINDOW",
    "TERMINAL_AMOUNT_MEAN_PRIOR",
    "TERMINAL_AMOUNT_DEVIATION_FROM_MEAN_PRIOR",
    "TERMINAL_SECONDS_SINCE_PREVIOUS",
)
FEATURE_FILENAMES: Final = {
    "train": "train_temporal_features.csv",
    "validation": "validation_temporal_features.csv",
    "test": "test_temporal_features.csv",
}


@dataclass(frozen=True, slots=True)
class TemporalFeatureConfig:
    """Versioned velocity windows and decimal output precision."""

    short_window_seconds: int
    long_window_seconds: int
    amount_decimal_places: int

    def __post_init__(self) -> None:
        if type(self.short_window_seconds) is not int or self.short_window_seconds <= 0:
            raise ValueError("short_window_seconds must be a positive integer")
        if type(self.long_window_seconds) is not int or self.long_window_seconds <= 0:
            raise ValueError("long_window_seconds must be a positive integer")
        if self.short_window_seconds >= self.long_window_seconds:
            raise ValueError(
                "short_window_seconds must be less than long_window_seconds"
            )
        if type(self.amount_decimal_places) is not int or not (
            0 <= self.amount_decimal_places <= 8
        ):
            raise ValueError("amount_decimal_places must be an integer from 0 to 8")


@dataclass(frozen=True, slots=True)
class TemporalFeatureRow:
    """Entity-history values available immediately before one timestamp."""

    transaction_id: int
    customer_tx_count_short_window: int
    customer_tx_count_long_window: int
    customer_amount_mean_prior: Decimal | None
    customer_amount_deviation_from_mean_prior: Decimal | None
    customer_seconds_since_previous: int | None
    terminal_tx_count_short_window: int
    terminal_tx_count_long_window: int
    terminal_amount_mean_prior: Decimal | None
    terminal_amount_deviation_from_mean_prior: Decimal | None
    terminal_seconds_since_previous: int | None

    def to_csv_row(self) -> dict[str, int | str]:
        """Return stable public column names with empty missing-history values."""
        return {
            "TRANSACTION_ID": self.transaction_id,
            "CUSTOMER_TX_COUNT_SHORT_WINDOW": (self.customer_tx_count_short_window),
            "CUSTOMER_TX_COUNT_LONG_WINDOW": self.customer_tx_count_long_window,
            "CUSTOMER_AMOUNT_MEAN_PRIOR": _format_optional_decimal(
                self.customer_amount_mean_prior
            ),
            "CUSTOMER_AMOUNT_DEVIATION_FROM_MEAN_PRIOR": (
                _format_optional_decimal(self.customer_amount_deviation_from_mean_prior)
            ),
            "CUSTOMER_SECONDS_SINCE_PREVIOUS": (
                ""
                if self.customer_seconds_since_previous is None
                else self.customer_seconds_since_previous
            ),
            "TERMINAL_TX_COUNT_SHORT_WINDOW": self.terminal_tx_count_short_window,
            "TERMINAL_TX_COUNT_LONG_WINDOW": self.terminal_tx_count_long_window,
            "TERMINAL_AMOUNT_MEAN_PRIOR": _format_optional_decimal(
                self.terminal_amount_mean_prior
            ),
            "TERMINAL_AMOUNT_DEVIATION_FROM_MEAN_PRIOR": (
                _format_optional_decimal(self.terminal_amount_deviation_from_mean_prior)
            ),
            "TERMINAL_SECONDS_SINCE_PREVIOUS": (
                ""
                if self.terminal_seconds_since_previous is None
                else self.terminal_seconds_since_previous
            ),
        }


@dataclass(frozen=True, slots=True)
class TemporalFeatureBuildResult:
    """Generated paths and row counts for each chronological split."""

    train_path: Path
    validation_path: Path
    test_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int


@dataclass(slots=True)
class _EntityHistory:
    short_timestamps: deque[datetime] = field(default_factory=deque)
    long_timestamps: deque[datetime] = field(default_factory=deque)
    amount_total: Decimal = Decimal(0)
    transaction_count: int = 0
    last_timestamp: datetime | None = None


def load_temporal_feature_config(path: Path) -> TemporalFeatureConfig:
    """Load and type-check temporal feature assumptions from TOML."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("temporal_features")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [temporal_features] table")

    try:
        return TemporalFeatureConfig(
            short_window_seconds=_require_int(section, "short_window_seconds"),
            long_window_seconds=_require_int(section, "long_window_seconds"),
            amount_decimal_places=_require_int(section, "amount_decimal_places"),
        )
    except KeyError as error:
        raise ValueError(
            f"missing temporal feature setting: {error.args[0]}"
        ) from error
    except ValueError as error:
        raise ValueError(f"invalid temporal feature configuration: {error}") from error


def build_temporal_features(
    transactions: Sequence[Transaction], config: TemporalFeatureConfig
) -> tuple[TemporalFeatureRow, ...]:
    """Compute customer and terminal features strictly before each row."""
    if not transactions:
        raise ValueError("transactions must not be empty")
    _validate_input_order(transactions)

    customer_histories: defaultdict[int, _EntityHistory] = defaultdict(_EntityHistory)
    terminal_histories: defaultdict[int, _EntityHistory] = defaultdict(_EntityHistory)
    feature_rows = []
    quantizer = Decimal(1).scaleb(-config.amount_decimal_places)

    for timestamp, grouped_iterator in groupby(
        transactions, key=lambda transaction: transaction.tx_datetime
    ):
        timestamp_group = tuple(grouped_iterator)

        for transaction in timestamp_group:
            customer_values = _history_values(
                customer_histories[transaction.customer_id],
                transaction,
                timestamp,
                config,
                quantizer,
            )
            terminal_values = _history_values(
                terminal_histories[transaction.terminal_id],
                transaction,
                timestamp,
                config,
                quantizer,
            )
            feature_rows.append(
                TemporalFeatureRow(
                    transaction_id=transaction.transaction_id,
                    customer_tx_count_short_window=customer_values[0],
                    customer_tx_count_long_window=customer_values[1],
                    customer_amount_mean_prior=customer_values[2],
                    customer_amount_deviation_from_mean_prior=customer_values[3],
                    customer_seconds_since_previous=customer_values[4],
                    terminal_tx_count_short_window=terminal_values[0],
                    terminal_tx_count_long_window=terminal_values[1],
                    terminal_amount_mean_prior=terminal_values[2],
                    terminal_amount_deviation_from_mean_prior=terminal_values[3],
                    terminal_seconds_since_previous=terminal_values[4],
                )
            )

        for transaction in timestamp_group:
            _update_history(
                customer_histories[transaction.customer_id], transaction, timestamp
            )
            _update_history(
                terminal_histories[transaction.terminal_id], transaction, timestamp
            )

    return tuple(feature_rows)


def _history_values(
    history: _EntityHistory,
    transaction: Transaction,
    timestamp: datetime,
    config: TemporalFeatureConfig,
    quantizer: Decimal,
) -> tuple[int, int, Decimal | None, Decimal | None, int | None]:
    _discard_expired(history.short_timestamps, timestamp, config.short_window_seconds)
    _discard_expired(history.long_timestamps, timestamp, config.long_window_seconds)
    prior_mean = (
        (history.amount_total / history.transaction_count).quantize(
            quantizer, rounding=ROUND_HALF_UP
        )
        if history.transaction_count
        else None
    )
    deviation = (
        (transaction.tx_amount - prior_mean).quantize(quantizer, rounding=ROUND_HALF_UP)
        if prior_mean is not None
        else None
    )
    seconds_since_previous = (
        int((timestamp - history.last_timestamp).total_seconds())
        if history.last_timestamp is not None
        else None
    )
    return (
        len(history.short_timestamps),
        len(history.long_timestamps),
        prior_mean,
        deviation,
        seconds_since_previous,
    )


def _update_history(
    history: _EntityHistory, transaction: Transaction, timestamp: datetime
) -> None:
    history.short_timestamps.append(timestamp)
    history.long_timestamps.append(timestamp)
    history.amount_total += transaction.tx_amount
    history.transaction_count += 1
    history.last_timestamp = timestamp


def build_processed_temporal_features(
    input_directory: Path,
    output_directory: Path,
    config: TemporalFeatureConfig,
) -> TemporalFeatureBuildResult:
    """Validate all splits and write aligned temporal feature CSVs atomically."""
    splits = load_processed_splits(input_directory)
    all_transactions = tuple(
        transaction for transactions in splits.ordered for transaction in transactions
    )
    all_features = build_temporal_features(all_transactions, config)
    feature_splits = _partition_features(all_features, splits)

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_directory / filename
        for name, filename in FEATURE_FILENAMES.items()
    }
    temporary_paths = {
        name: path.with_name(f".{path.name}.tmp") for name, path in paths.items()
    }
    try:
        for name, rows in feature_splits.items():
            _write_feature_csv(rows, temporary_paths[name])
        for name in FEATURE_FILENAMES:
            os.replace(temporary_paths[name], paths[name])
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)

    return TemporalFeatureBuildResult(
        train_path=paths["train"],
        validation_path=paths["validation"],
        test_path=paths["test"],
        train_rows=len(feature_splits["train"]),
        validation_rows=len(feature_splits["validation"]),
        test_rows=len(feature_splits["test"]),
    )


def _validate_input_order(transactions: Sequence[Transaction]) -> None:
    previous_timestamp: datetime | None = None
    seen_ids = set()
    for transaction in transactions:
        if transaction.transaction_id in seen_ids:
            raise ValueError("transaction IDs must be unique")
        seen_ids.add(transaction.transaction_id)
        if transaction.tx_datetime.tzinfo is None or (
            transaction.tx_datetime.utcoffset() != timedelta(0)
        ):
            raise ValueError("transaction timestamps must be timezone-aware UTC")
        if (
            previous_timestamp is not None
            and transaction.tx_datetime < previous_timestamp
        ):
            raise ValueError("transactions must be sorted chronologically")
        previous_timestamp = transaction.tx_datetime


def _discard_expired(
    timestamps: deque[datetime], current_timestamp: datetime, window_seconds: int
) -> None:
    cutoff = current_timestamp - timedelta(seconds=window_seconds)
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()


def _partition_features(
    features: tuple[TemporalFeatureRow, ...], splits: ProcessedSplits
) -> dict[str, tuple[TemporalFeatureRow, ...]]:
    train_end = len(splits.train)
    validation_end = train_end + len(splits.validation)
    return {
        "train": features[:train_end],
        "validation": features[train_end:validation_end],
        "test": features[validation_end:],
    }


def _write_feature_csv(
    feature_rows: Sequence[TemporalFeatureRow], output_path: Path
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=TEMPORAL_FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(feature.to_csv_row() for feature in feature_rows)


def _format_optional_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format(value, "f")


def _require_int(section: Mapping[str, object], key: str) -> int:
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value
