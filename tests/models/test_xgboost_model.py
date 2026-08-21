"""Tests for the fixed train-only XGBoost main-model baseline."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.features.matrix import join_transaction_features
from fraud_detection.features.temporal import (
    TemporalFeatureConfig,
    build_temporal_features,
)
from fraud_detection.models.xgboost_model import (
    XGBoostConfig,
    fit_xgboost_baseline,
    fit_xgboost_feature_variant,
    load_xgboost_config,
    predict_xgboost_feature_variant_scores,
    predict_xgboost_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "xgboost_baseline.toml"
TEMPORAL_CONFIG = TemporalFeatureConfig(3_600, 86_400, 4)
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_versioned_xgboost_config_is_frozen() -> None:
    """The first main-model configuration should load without hidden tuning."""
    assert load_xgboost_config(CONFIG_PATH) == XGBoostConfig(
        200,
        3,
        0.05,
        5.0,
        0.8,
        0.8,
        1.0,
        0.0,
        True,
        42,
        1,
        0.5,
    )


def test_xgboost_is_deterministic_and_held_out_labels_do_not_affect_scores() -> None:
    """Train-only fitting and decision-time features must isolate held-out labels."""
    config = load_xgboost_config(CONFIG_PATH)
    transactions = _transactions(36, fraud_ids={3, 9, 17, 25, 31})
    features = build_temporal_features(transactions, TEMPORAL_CONFIG)
    train = join_transaction_features(transactions[:30], features[:30])
    held_out = join_transaction_features(transactions[30:], features[30:])
    relabeled = tuple(
        replace(
            row,
            transaction=replace(row.transaction, tx_fraud=1, tx_fraud_scenario=3),
        )
        for row in held_out
    )

    first_model = fit_xgboost_baseline(train, config)
    second_model = fit_xgboost_baseline(train, config)
    first_scores = predict_xgboost_scores(first_model, held_out)

    assert first_scores == pytest.approx(
        predict_xgboost_scores(second_model, held_out), abs=0.0
    )
    assert first_scores == pytest.approx(
        predict_xgboost_scores(first_model, relabeled), abs=0.0
    )
    assert all(0 <= score <= 1 for score in first_scores)


def test_xgboost_training_requires_both_classes() -> None:
    """A single-class training set must not produce a misleading model."""
    transactions = _transactions(8)
    rows = join_transaction_features(
        transactions,
        build_temporal_features(transactions, TEMPORAL_CONFIG),
    )

    with pytest.raises(ValueError, match="both fraud classes"):
        fit_xgboost_baseline(rows, load_xgboost_config(CONFIG_PATH))


def test_full_feature_variant_matches_frozen_baseline_wrapper() -> None:
    """The ablation path must preserve the frozen full-baseline behavior."""
    config = load_xgboost_config(CONFIG_PATH)
    transactions = _transactions(36, fraud_ids={3, 9, 17, 25, 31})
    features = build_temporal_features(transactions, TEMPORAL_CONFIG)
    train = join_transaction_features(transactions[:30], features[:30])
    validation = join_transaction_features(transactions[30:], features[30:])

    baseline = fit_xgboost_baseline(train, config)
    full_variant = fit_xgboost_feature_variant(
        train,
        config,
        include_temporal_features=True,
        include_synthetic_ids=True,
    )

    assert predict_xgboost_scores(baseline, validation) == pytest.approx(
        predict_xgboost_feature_variant_scores(
            full_variant,
            validation,
            include_temporal_features=True,
            include_synthetic_ids=True,
        ),
        abs=0.0,
    )


def _transactions(
    count: int, *, fraud_ids: set[int] | None = None
) -> tuple[Transaction, ...]:
    fraud_ids = fraud_ids or set()
    return tuple(
        Transaction(
            transaction_id=index,
            tx_datetime=START + timedelta(minutes=index * 45),
            customer_id=index % 6,
            terminal_id=index % 4,
            tx_amount=(
                Decimal("260.00") if index in fraud_ids else Decimal("30.00") + index
            ),
            tx_time_seconds=index * 45 * 60,
            tx_time_days=(index * 45 * 60) // SECONDS_PER_DAY,
            tx_fraud=int(index in fraud_ids),
            tx_fraud_scenario=int(index in fraud_ids),
        )
        for index in range(count)
    )
