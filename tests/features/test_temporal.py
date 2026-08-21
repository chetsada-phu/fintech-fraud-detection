"""Boundary and leakage tests for past-only customer features."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.features.temporal import (
    TEMPORAL_FEATURE_COLUMNS,
    TemporalFeatureConfig,
    build_temporal_features,
    load_temporal_feature_config,
)
from fraud_detection.features.temporal_cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "temporal_features.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)
CONFIG = TemporalFeatureConfig(3_600, 86_400, 4)


def test_exact_windows_and_equal_timestamps_use_strictly_prior_history() -> None:
    """Lower window edges count, while current-timestamp peers do not."""
    transactions = (
        _transaction(0, 0, "10.00", customer_id=1),
        _transaction(1, 100, "99.00", customer_id=2),
        _transaction(2, 3_600, "30.00", customer_id=1),
        _transaction(3, 3_601, "20.00", customer_id=1),
        _transaction(4, 3_601, "40.00", customer_id=1),
        _transaction(5, 7_201, "50.00", customer_id=1),
    )

    features = build_temporal_features(transactions, CONFIG)

    assert features[0].customer_amount_mean_prior is None
    assert features[0].customer_seconds_since_previous is None
    assert features[1].customer_tx_count_long_window == 0
    assert features[2].customer_tx_count_short_window == 1
    assert features[2].customer_tx_count_long_window == 1
    assert features[2].customer_amount_mean_prior == Decimal("10.0000")
    assert features[2].customer_amount_deviation_from_mean_prior == Decimal("20.0000")
    assert features[2].customer_seconds_since_previous == 3_600

    for feature in features[3:5]:
        assert feature.customer_tx_count_short_window == 1
        assert feature.customer_tx_count_long_window == 2
        assert feature.customer_amount_mean_prior == Decimal("20.0000")
        assert feature.customer_seconds_since_previous == 1
    assert features[3].customer_amount_deviation_from_mean_prior == Decimal("0.0000")
    assert features[4].customer_amount_deviation_from_mean_prior == Decimal("20.0000")

    assert features[5].customer_tx_count_short_window == 2
    assert features[5].customer_tx_count_long_window == 4
    assert features[5].customer_amount_mean_prior == Decimal("25.0000")
    assert features[5].customer_amount_deviation_from_mean_prior == Decimal("25.0000")


def test_future_rows_and_post_event_labels_cannot_change_earlier_features() -> None:
    """Appending the future or relabeling outcomes must preserve prior rows."""
    transactions = tuple(
        _transaction(index, index * 600, str(10 + index), customer_id=index % 2)
        for index in range(6)
    )
    prefix = transactions[:4]
    prefix_features = build_temporal_features(prefix, CONFIG)
    extended_features = build_temporal_features(transactions, CONFIG)
    relabeled = tuple(
        replace(transaction, tx_fraud=1, tx_fraud_scenario=3)
        for transaction in transactions
    )

    assert extended_features[: len(prefix)] == prefix_features
    assert build_temporal_features(relabeled, CONFIG) == extended_features


def test_terminal_windows_and_equal_timestamps_use_strictly_prior_history() -> None:
    """Terminal peers at one timestamp must share the same prior state."""
    transactions = (
        _transaction(0, 0, "10.00", customer_id=1, terminal_id=7),
        _transaction(1, 100, "99.00", customer_id=2, terminal_id=8),
        _transaction(2, 3_600, "30.00", customer_id=3, terminal_id=7),
        _transaction(3, 3_601, "20.00", customer_id=4, terminal_id=7),
        _transaction(4, 3_601, "40.00", customer_id=5, terminal_id=7),
    )

    features = build_temporal_features(transactions, CONFIG)

    assert features[0].terminal_amount_mean_prior is None
    assert features[2].terminal_tx_count_short_window == 1
    assert features[2].terminal_tx_count_long_window == 1
    assert features[2].terminal_amount_mean_prior == Decimal("10.0000")
    assert features[2].terminal_amount_deviation_from_mean_prior == Decimal("20.0000")
    assert features[2].terminal_seconds_since_previous == 3_600
    for feature in features[3:5]:
        assert feature.terminal_tx_count_short_window == 1
        assert feature.terminal_tx_count_long_window == 2
        assert feature.terminal_amount_mean_prior == Decimal("20.0000")
        assert feature.terminal_seconds_since_previous == 1
    assert features[3].terminal_amount_deviation_from_mean_prior == Decimal("0.0000")
    assert features[4].terminal_amount_deviation_from_mean_prior == Decimal("20.0000")


def test_temporal_config_is_versioned_and_rejects_reversed_windows() -> None:
    """The repository windows should load exactly and remain ordered."""
    assert load_temporal_feature_config(CONFIG_PATH) == CONFIG

    with pytest.raises(ValueError, match="less than"):
        TemporalFeatureConfig(86_400, 3_600, 4)


def test_cli_writes_aligned_byte_stable_features_across_splits(
    tmp_path: Path,
) -> None:
    """Validation features should carry forward training history without labels."""
    input_directory = tmp_path / "processed"
    output_directory = tmp_path / "features"
    input_directory.mkdir()
    transactions = tuple(
        _transaction(
            index,
            index * 3_600,
            str(10 + index),
            customer_id=1,
            fraud=index in {2, 7, 10},
        )
        for index in range(12)
    )
    write_transactions_csv(transactions[:6], input_directory / "train.csv")
    write_transactions_csv(transactions[6:9], input_directory / "validation.csv")
    write_transactions_csv(transactions[9:], input_directory / "test.csv")

    first_exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--input-directory",
            str(input_directory),
            "--output-directory",
            str(output_directory),
        ]
    )
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(output_directory.glob("*.csv"))
    }
    second_exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--input-directory",
            str(input_directory),
            "--output-directory",
            str(output_directory),
        ]
    )

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_bytes == {
        path.name: path.read_bytes() for path in sorted(output_directory.glob("*.csv"))
    }
    validation_path = output_directory / "validation_temporal_features.csv"
    with validation_path.open(encoding="utf-8", newline="") as feature_file:
        rows = list(csv.DictReader(feature_file))
    assert tuple(rows[0]) == TEMPORAL_FEATURE_COLUMNS
    assert [row["TRANSACTION_ID"] for row in rows] == ["6", "7", "8"]
    assert rows[0]["CUSTOMER_TX_COUNT_SHORT_WINDOW"] == "1"
    assert rows[0]["CUSTOMER_TX_COUNT_LONG_WINDOW"] == "6"
    assert rows[0]["CUSTOMER_AMOUNT_MEAN_PRIOR"] == "12.5000"
    assert rows[0]["TERMINAL_TX_COUNT_SHORT_WINDOW"] == "0"
    assert rows[0]["TERMINAL_TX_COUNT_LONG_WINDOW"] == "2"
    assert rows[0]["TERMINAL_AMOUNT_MEAN_PRIOR"] == "11.5000"
    assert rows[0]["TERMINAL_SECONDS_SINCE_PREVIOUS"] == "10800"


def _transaction(
    transaction_id: int,
    seconds: int,
    amount: str,
    *,
    customer_id: int,
    terminal_id: int | None = None,
    fraud: bool = False,
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id,
        tx_datetime=START + timedelta(seconds=seconds),
        customer_id=customer_id,
        terminal_id=(transaction_id % 3 if terminal_id is None else terminal_id),
        tx_amount=Decimal(amount),
        tx_time_seconds=seconds,
        tx_time_days=seconds // SECONDS_PER_DAY,
        tx_fraud=int(fraud),
        tx_fraud_scenario=int(fraud),
    )
