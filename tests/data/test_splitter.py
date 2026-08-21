"""Tests for leakage-safe chronological transaction splitting."""

import csv
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import CSV_COLUMNS, SECONDS_PER_DAY, Transaction
from fraud_detection.data.splitter import (
    LABEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SplitConfig,
    load_split_config,
    split_transactions_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_split.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_default_split_config_is_versioned_and_valid() -> None:
    """The repository configuration should define three complete split targets."""
    config = load_split_config(CONFIG_PATH)

    assert config == SplitConfig(0.60, 0.20, 0.20)


def test_split_is_strictly_chronological_without_overlap_or_row_loss(
    tmp_path: Path,
) -> None:
    """Whole timestamp groups should cross no boundary or disappear."""
    transactions = _transactions()
    input_path = tmp_path / "raw.csv"
    output_directory = tmp_path / "processed"
    write_transactions_csv(transactions, input_path)

    result = split_transactions_csv(
        input_path, output_directory, SplitConfig(0.50, 0.25, 0.25)
    )

    train = _read_rows(result.train_path)
    validation = _read_rows(result.validation_path)
    test = _read_rows(result.test_path)
    split_ids = [
        {int(row["TRANSACTION_ID"]) for row in rows}
        for rows in (train, validation, test)
    ]

    assert all(split_ids)
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    assert set.union(*split_ids) == set(range(len(transactions)))
    assert result.train_rows + result.validation_rows + result.test_rows == len(
        transactions
    )
    assert _latest_time(train) < _earliest_time(validation)
    assert _latest_time(validation) < _earliest_time(test)
    assert tuple(train[0]) == CSV_COLUMNS


def test_split_output_is_byte_stable(tmp_path: Path) -> None:
    """The same input and versioned fractions should create identical bytes."""
    input_path = tmp_path / "raw.csv"
    write_transactions_csv(_transactions(), input_path)
    config = SplitConfig(0.50, 0.25, 0.25)

    first = split_transactions_csv(input_path, tmp_path / "first", config)
    second = split_transactions_csv(input_path, tmp_path / "second", config)

    assert first.train_path.read_bytes() == second.train_path.read_bytes()
    assert first.validation_path.read_bytes() == second.validation_path.read_bytes()
    assert first.test_path.read_bytes() == second.test_path.read_bytes()


def test_model_features_explicitly_exclude_post_event_labels() -> None:
    """Neither fraud outcome may silently enter a downstream model matrix."""
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(LABEL_COLUMNS)
    assert "TX_FRAUD" not in MODEL_FEATURE_COLUMNS
    assert "TX_FRAUD_SCENARIO" not in MODEL_FEATURE_COLUMNS


def _transactions() -> tuple[Transaction, ...]:
    seconds = (0, 1, 1, 2, 3, 4, 5, 6, 7, 8)
    transactions = []
    for transaction_id, tx_seconds in enumerate(seconds):
        scenario = 2 if transaction_id in {7, 8} else 0
        transactions.append(
            Transaction(
                transaction_id=transaction_id,
                tx_datetime=START + timedelta(seconds=tx_seconds),
                customer_id=transaction_id % 3,
                terminal_id=transaction_id % 2,
                tx_amount=Decimal("10.00") + transaction_id,
                tx_time_seconds=tx_seconds,
                tx_time_days=tx_seconds // SECONDS_PER_DAY,
                tx_fraud=int(scenario != 0),
                tx_fraud_scenario=scenario,
            )
        )
    return tuple(transactions)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as data_file:
        return list(csv.DictReader(data_file))


def _latest_time(rows: list[dict[str, str]]) -> datetime:
    return max(datetime.fromisoformat(row["TX_DATETIME"]) for row in rows)


def _earliest_time(rows: list[dict[str, str]]) -> datetime:
    return min(datetime.fromisoformat(row["TX_DATETIME"]) for row in rows)
