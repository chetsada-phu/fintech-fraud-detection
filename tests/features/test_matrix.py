"""Tests for transaction-ID-aligned model feature joins."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.features.matrix import (
    join_transaction_features,
    load_temporal_feature_csv,
)
from fraud_detection.features.temporal import (
    TEMPORAL_FEATURE_COLUMNS,
    TemporalFeatureConfig,
    build_temporal_features,
)

START = datetime(2018, 4, 1, tzinfo=UTC)
CONFIG = TemporalFeatureConfig(3_600, 86_400, 4)


def test_join_requires_exact_ordered_transaction_ids() -> None:
    """A shifted temporal row must fail instead of silently joining badly."""
    transactions = _transactions()
    features = build_temporal_features(transactions, CONFIG)

    joined = join_transaction_features(transactions, features)

    assert [row.transaction.transaction_id for row in joined] == list(range(6))
    assert [row.temporal.transaction_id for row in joined] == list(range(6))
    shifted = (replace(features[0], transaction_id=99), *features[1:])
    with pytest.raises(ValueError, match="align in order"):
        join_transaction_features(transactions, shifted)


def test_feature_csv_rejects_partially_missing_customer_history(
    tmp_path: Path,
) -> None:
    """Corrupt missing-history groups must fail before model fitting."""
    path = tmp_path / "features.csv"
    path.write_text(
        ",".join(TEMPORAL_FEATURE_COLUMNS) + "\n0,0,0,,1.0000,,0,0,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing together"):
        load_temporal_feature_csv(path)


def test_feature_csv_rejects_partially_missing_terminal_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.csv"
    path.write_text(
        ",".join(TEMPORAL_FEATURE_COLUMNS) + "\n0,0,0,,,,0,0,,1.0000,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="terminal-history fields"):
        load_temporal_feature_csv(path)


def _transactions() -> tuple[Transaction, ...]:
    return tuple(
        Transaction(
            transaction_id=index,
            tx_datetime=START + timedelta(hours=index),
            customer_id=index % 2,
            terminal_id=index % 3,
            tx_amount=Decimal("50.00") + index,
            tx_time_seconds=index * 3_600,
            tx_time_days=0,
            tx_fraud=int(index == 3),
            tx_fraud_scenario=int(index == 3),
        )
        for index in range(6)
    )
