"""Tests for the framework-free one-transaction inference contract."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.inference import (
    SelectedDecisionPolicyConfig,
    TransactionDecisionResult,
    decide_transaction_from_score,
    load_selected_decision_policy_config,
)
from fraud_detection.decisioning.policy import (
    RISK_SCORE_REVIEW,
    Decision,
    DecisionThresholds,
)
from fraud_detection.decisioning.reasons import (
    HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
    HIGH_TRANSACTION_AMOUNT,
    load_decision_reason_config,
)
from fraud_detection.features.matrix import JoinedFeatureRow
from fraud_detection.features.temporal import TemporalFeatureRow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_decision_policy.toml"
REASON_CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_reasons.toml"


def test_versioned_selected_policy_loads_exact_frozen_thresholds() -> None:
    config = load_selected_decision_policy_config(POLICY_CONFIG_PATH)

    assert config == SelectedDecisionPolicyConfig(
        policy_version="validation-provisional-v1",
        selection_split="validation",
        score_source_label="XGBoost engineering-only provisional score source",
        review_threshold=Decimal("0.93"),
        decline_threshold=Decimal("0.93"),
    )
    assert config.thresholds == DecisionThresholds(0.93, 0.93)


def test_exact_boundaries_return_stable_decisions_and_reasons() -> None:
    policy_config = _policy_config("0.25", "0.75")
    reason_config = load_decision_reason_config(REASON_CONFIG_PATH)
    results = (
        decide_transaction_from_score(
            _joined_row("300.00", prior_mean="100.00"),
            0.2499,
            policy_config,
            reason_config,
        ),
        decide_transaction_from_score(
            _joined_row("100.00", prior_mean="100.00"),
            0.25,
            policy_config,
            reason_config,
        ),
        decide_transaction_from_score(
            _joined_row("300.00", prior_mean="100.00"),
            0.7499,
            policy_config,
            reason_config,
        ),
        decide_transaction_from_score(
            _joined_row("220.01", prior_mean="100.00"),
            0.75,
            policy_config,
            reason_config,
        ),
    )

    assert tuple(result.decision for result in results) == (
        Decision.APPROVE,
        Decision.REVIEW,
        Decision.REVIEW,
        Decision.DECLINE,
    )
    assert tuple(result.reason_codes for result in results) == (
        (),
        (RISK_SCORE_REVIEW,),
        (HIGH_AMOUNT_VS_CUSTOMER_BASELINE, HIGH_TRANSACTION_AMOUNT),
        (HIGH_TRANSACTION_AMOUNT,),
    )
    assert all(result.policy_version == "fixture-v1" for result in results)


def test_checked_in_equal_thresholds_have_no_review_band() -> None:
    policy_config = load_selected_decision_policy_config(POLICY_CONFIG_PATH)
    reason_config = load_decision_reason_config(REASON_CONFIG_PATH)
    row = _joined_row("100.00", prior_mean="100.00")

    below = decide_transaction_from_score(row, 0.9299, policy_config, reason_config)
    at_boundary = decide_transaction_from_score(row, 0.93, policy_config, reason_config)

    assert below == TransactionDecisionResult(
        transaction_id=0,
        risk_score=0.9299,
        decision=Decision.APPROVE,
        reason_codes=(),
        policy_version="validation-provisional-v1",
    )
    assert at_boundary.decision is Decision.DECLINE


def test_invalid_score_and_misaligned_feature_row_are_rejected() -> None:
    policy_config = load_selected_decision_policy_config(POLICY_CONFIG_PATH)
    reason_config = load_decision_reason_config(REASON_CONFIG_PATH)
    row = _joined_row("100.00", prior_mean="100.00")

    for risk_score in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="risk_score"):
            decide_transaction_from_score(
                row,
                risk_score,
                policy_config,
                reason_config,
            )

    misaligned = replace(
        row,
        temporal=replace(row.temporal, transaction_id=1),
    )
    with pytest.raises(ValueError, match="IDs must align"):
        decide_transaction_from_score(
            misaligned,
            0.50,
            policy_config,
            reason_config,
        )


def test_post_event_labels_cannot_change_inference_result() -> None:
    policy_config = _policy_config("0.25", "0.75")
    reason_config = load_decision_reason_config(REASON_CONFIG_PATH)
    row = _joined_row("300.00", prior_mean="100.00")
    relabeled = replace(
        row,
        transaction=replace(row.transaction, tx_fraud=1, tx_fraud_scenario=3),
    )

    original = decide_transaction_from_score(
        row,
        0.75,
        policy_config,
        reason_config,
    )
    after_relabel = decide_transaction_from_score(
        relabeled,
        0.75,
        policy_config,
        reason_config,
    )

    assert original == after_relabel


def _policy_config(
    review_threshold: str, decline_threshold: str
) -> SelectedDecisionPolicyConfig:
    return SelectedDecisionPolicyConfig(
        policy_version="fixture-v1",
        selection_split="validation",
        score_source_label="Fixture supplied scores",
        review_threshold=Decimal(review_threshold),
        decline_threshold=Decimal(decline_threshold),
    )


def _joined_row(amount: str, *, prior_mean: str | None) -> JoinedFeatureRow:
    transaction = Transaction(
        transaction_id=0,
        tx_datetime=datetime(2018, 4, 1, tzinfo=UTC),
        customer_id=1,
        terminal_id=1,
        tx_amount=Decimal(amount),
        tx_time_seconds=0,
        tx_time_days=0,
        tx_fraud=0,
        tx_fraud_scenario=0,
    )
    mean = Decimal(prior_mean) if prior_mean is not None else None
    temporal = TemporalFeatureRow(
        transaction_id=0,
        customer_tx_count_short_window=0,
        customer_tx_count_long_window=0,
        customer_amount_mean_prior=mean,
        customer_amount_deviation_from_mean_prior=(
            transaction.tx_amount - mean if mean is not None else None
        ),
        customer_seconds_since_previous=60 if mean is not None else None,
        terminal_tx_count_short_window=0,
        terminal_tx_count_long_window=0,
        terminal_amount_mean_prior=None,
        terminal_amount_deviation_from_mean_prior=None,
        terminal_seconds_since_previous=None,
    )
    return JoinedFeatureRow(transaction=transaction, temporal=temporal)
