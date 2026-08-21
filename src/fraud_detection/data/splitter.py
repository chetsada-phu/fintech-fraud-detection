"""Leakage-safe chronological splitting for validated transaction data."""

from __future__ import annotations

import csv
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final

from fraud_detection.data.schema import (
    CSV_COLUMNS,
    Transaction,
    validate_transactions,
)

LABEL_COLUMNS: Final = ("TX_FRAUD", "TX_FRAUD_SCENARIO")
NON_FEATURE_COLUMNS: Final = ("TRANSACTION_ID", *LABEL_COLUMNS)
MODEL_FEATURE_COLUMNS: Final = tuple(
    column for column in CSV_COLUMNS if column not in NON_FEATURE_COLUMNS
)
SPLIT_FILENAMES: Final = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Versioned target fractions for chronological dataset splits."""

    train_fraction: float
    validation_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(not math.isfinite(value) or value <= 0 for value in fractions):
            raise ValueError("split fractions must be finite and positive")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("split fractions must sum to 1")


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Paths, sizes, and temporal boundaries produced by one split run."""

    train_path: Path
    validation_path: Path
    test_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime


def load_split_config(path: Path) -> SplitConfig:
    """Load and type-check chronological split fractions from TOML."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("split")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [split] table")

    try:
        return SplitConfig(
            train_fraction=_require_number(section, "train_fraction"),
            validation_fraction=_require_number(section, "validation_fraction"),
            test_fraction=_require_number(section, "test_fraction"),
        )
    except KeyError as error:
        raise ValueError(f"missing split setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid split configuration: {error}") from error


def split_transactions_csv(
    input_path: Path,
    output_directory: Path,
    config: SplitConfig,
) -> SplitResult:
    """Validate a raw CSV and write each chronological split atomically."""
    transactions = _read_transactions_csv(input_path)
    train_end_index, validation_end_index = _choose_boundaries(transactions, config)
    train = transactions[:train_end_index]
    validation = transactions[train_end_index:validation_end_index]
    test = transactions[validation_end_index:]

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_directory / filename for name, filename in SPLIT_FILENAMES.items()
    }
    temporary_paths = {
        name: path.with_name(f".{path.name}.tmp") for name, path in paths.items()
    }
    try:
        _write_transactions_csv(train, temporary_paths["train"])
        _write_transactions_csv(validation, temporary_paths["validation"])
        _write_transactions_csv(test, temporary_paths["test"])
        for name in SPLIT_FILENAMES:
            os.replace(temporary_paths[name], paths[name])
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)

    return SplitResult(
        train_path=paths["train"],
        validation_path=paths["validation"],
        test_path=paths["test"],
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_end=train[-1].tx_datetime,
        validation_start=validation[0].tx_datetime,
        validation_end=validation[-1].tx_datetime,
        test_start=test[0].tx_datetime,
    )


def _read_transactions_csv(path: Path) -> tuple[Transaction, ...]:
    with path.open(encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                "input CSV columns must exactly match the transaction data contract"
            )
        try:
            transactions = tuple(_transaction_from_row(row) for row in reader)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid transaction CSV value: {error}") from error

    if not transactions:
        raise ValueError("input CSV must contain at least one transaction")
    start_datetime = transactions[0].tx_datetime - timedelta(
        seconds=transactions[0].tx_time_seconds
    )
    validate_transactions(transactions, start_datetime)
    return transactions


def _transaction_from_row(row: Mapping[str, str]) -> Transaction:
    return Transaction(
        transaction_id=int(row["TRANSACTION_ID"]),
        tx_datetime=datetime.fromisoformat(row["TX_DATETIME"]),
        customer_id=int(row["CUSTOMER_ID"]),
        terminal_id=int(row["TERMINAL_ID"]),
        tx_amount=Decimal(row["TX_AMOUNT"]),
        tx_time_seconds=int(row["TX_TIME_SECONDS"]),
        tx_time_days=int(row["TX_TIME_DAYS"]),
        tx_fraud=int(row["TX_FRAUD"]),
        tx_fraud_scenario=int(row["TX_FRAUD_SCENARIO"]),
    )


def _choose_boundaries(
    transactions: Sequence[Transaction], config: SplitConfig
) -> tuple[int, int]:
    boundaries = [
        index
        for index in range(1, len(transactions))
        if transactions[index - 1].tx_datetime < transactions[index].tx_datetime
    ]
    if len(boundaries) < 2:
        raise ValueError(
            "at least three distinct transaction timestamps are required to split data"
        )

    train_target = len(transactions) * config.train_fraction
    train_boundary = min(
        boundaries[:-1], key=lambda boundary: (abs(boundary - train_target), boundary)
    )
    validation_target = len(transactions) * (
        config.train_fraction + config.validation_fraction
    )
    validation_boundary = min(
        (boundary for boundary in boundaries if boundary > train_boundary),
        key=lambda boundary: (abs(boundary - validation_target), boundary),
    )
    return train_boundary, validation_boundary


def _write_transactions_csv(
    transactions: Sequence[Transaction], output_path: Path
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(transaction.to_csv_row() for transaction in transactions)


def _require_number(section: Mapping[str, object], key: str) -> float:
    value = section[key]
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be a number")
    return float(value)
