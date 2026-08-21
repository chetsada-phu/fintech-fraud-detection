"""Tests for train-only Logistic Regression preprocessing and scoring."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.data.splitter import MODEL_FEATURE_COLUMNS
from fraud_detection.models.logistic import (
    LOGISTIC_SOURCE_COLUMNS,
    LogisticConfig,
    fit_logistic_baseline,
    load_logistic_config,
    predict_fraud_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "logistic_baseline.toml"


def test_versioned_logistic_config_and_feature_contract() -> None:
    """Fixed model inputs must remain inside the no-label feature contract."""
    config = load_logistic_config(CONFIG_PATH)

    assert config == LogisticConfig(1.0, "balanced", "liblinear", 1000, 42, 0.5)
    assert set(LOGISTIC_SOURCE_COLUMNS).issubset(MODEL_FEATURE_COLUMNS)
    assert "TRANSACTION_ID" not in LOGISTIC_SOURCE_COLUMNS
    assert "TX_FRAUD" not in LOGISTIC_SOURCE_COLUMNS
    assert "TX_FRAUD_SCENARIO" not in LOGISTIC_SOURCE_COLUMNS


def test_training_and_scoring_are_deterministic_and_label_isolated() -> None:
    """Held-out labels and unseen IDs cannot alter fraud scores."""
    config = load_logistic_config(CONFIG_PATH)
    train = _transactions(12, fraud_ids={2, 5, 9})
    held_out = (
        replace(_transactions(1, start_id=20)[0], customer_id=999, terminal_id=999),
        _transactions(1, start_id=21)[0],
    )
    relabeled = tuple(
        replace(transaction, tx_fraud=1, tx_fraud_scenario=3)
        for transaction in held_out
    )

    first_model = fit_logistic_baseline(train, config)
    second_model = fit_logistic_baseline(train, config)
    first_scores = predict_fraud_scores(first_model, held_out)
    second_scores = predict_fraud_scores(second_model, held_out)

    assert first_scores == pytest.approx(second_scores, abs=0.0)
    assert first_scores == pytest.approx(
        predict_fraud_scores(first_model, relabeled), abs=0.0
    )
    assert all(0 <= score <= 1 for score in first_scores)


def test_training_requires_both_classes() -> None:
    """A misleading single-class training run must fail clearly."""
    with pytest.raises(ValueError, match="both fraud classes"):
        fit_logistic_baseline(_transactions(5), load_logistic_config(CONFIG_PATH))


def _transactions(
    count: int,
    *,
    start_id: int = 0,
    fraud_ids: set[int] | None = None,
) -> tuple[Transaction, ...]:
    fraud_ids = fraud_ids or set()
    return tuple(
        Transaction(
            transaction_id=transaction_id,
            tx_datetime=datetime(2018, 4, 1, tzinfo=UTC)
            + timedelta(hours=transaction_id),
            customer_id=transaction_id % 4,
            terminal_id=transaction_id % 3,
            tx_amount=Decimal("250.00")
            if transaction_id in fraud_ids
            else Decimal("50.00") + transaction_id,
            tx_time_seconds=transaction_id * 3_600,
            tx_time_days=(transaction_id * 3_600) // 86_400,
            tx_fraud=int(transaction_id in fraud_ids),
            tx_fraud_scenario=int(transaction_id in fraud_ids),
        )
        for transaction_id in range(start_id, start_id + count)
    )
