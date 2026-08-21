"""Tests for the model-agnostic three-way decision policy."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.costs import BusinessCostConfig
from fraud_detection.decisioning.policy import (
    RISK_SCORE_DECLINE,
    RISK_SCORE_REVIEW,
    Decision,
    DecisionPolicyConfig,
    DecisionThresholds,
    apply_decision_policy,
    calculate_policy_operating_cost,
    load_decision_policy_config,
    select_policy_thresholds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_policy.toml"


def test_versioned_policy_contract_is_validation_only() -> None:
    """The checked-in search grid must not permit test-based selection."""
    assert load_decision_policy_config(CONFIG_PATH) == DecisionPolicyConfig(
        selection_split="validation",
        score_source_label="XGBoost engineering-only provisional score source",
        threshold_grid_step=Decimal("0.01"),
    )


def test_exact_boundaries_and_reason_codes_are_stable() -> None:
    """Equality belongs to the riskier action at each boundary."""
    results = apply_decision_policy(
        (0.2499, 0.25, 0.7499, 0.75),
        DecisionThresholds(review_threshold=0.25, decline_threshold=0.75),
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
        (RISK_SCORE_REVIEW,),
        (RISK_SCORE_DECLINE,),
    )


def test_equal_thresholds_create_no_review_band() -> None:
    """Decline must take precedence when both boundaries are identical."""
    results = apply_decision_policy(
        (0.49, 0.50, 0.51),
        DecisionThresholds(review_threshold=0.50, decline_threshold=0.50),
    )

    assert tuple(result.decision for result in results) == (
        Decision.APPROVE,
        Decision.DECLINE,
        Decision.DECLINE,
    )


def test_three_way_cost_counts_each_operational_outcome() -> None:
    """Missed fraud, reviews, and legitimate declines use distinct costs."""
    transactions = (
        _transaction(0, "100.00", fraud=1),
        _transaction(1, "200.00", fraud=1),
        _transaction(2, "50.00", fraud=0),
        _transaction(3, "40.00", fraud=0),
    )
    decisions = apply_decision_policy(
        (0.10, 0.50, 0.60, 0.90),
        DecisionThresholds(review_threshold=0.50, decline_threshold=0.80),
    )
    costs = BusinessCostConfig(
        fraud_loss_multiplier=Decimal("1.00"),
        manual_review_cost=Decimal("5.00"),
        max_manual_review_rate=0.50,
        false_decline_cost=Decimal("25.00"),
    )

    result = calculate_policy_operating_cost(transactions, decisions, costs)

    assert (result.approvals, result.reviews, result.declines) == (1, 2, 1)
    assert result.false_declines == 1
    assert result.missed_fraud_loss == Decimal("100.0000")
    assert result.manual_review_cost == Decimal("10.00")
    assert result.false_decline_cost == Decimal("25.00")
    assert result.total == Decimal("135.0000")
    assert result.total_per_1000_transactions == Decimal("33750.0000")
    assert result.fraud_amount_captured == Decimal("200.00")
    assert result.review_rate == 0.50
    assert result.false_decline_rate == 0.50
    assert result.within_review_capacity is True


def test_threshold_selection_is_deterministic_and_respects_capacity() -> None:
    """The selected validation policy must never exceed review capacity."""
    transactions = tuple(
        _transaction(index, "100.00", fraud=int(index in {18, 19}))
        for index in range(20)
    )
    scores = tuple(index / 20 for index in range(20))
    policy_config = DecisionPolicyConfig(
        selection_split="validation",
        score_source_label="Provisional fixture scores",
        threshold_grid_step=Decimal("0.10"),
    )
    cost_config = BusinessCostConfig(
        fraud_loss_multiplier=Decimal("1.00"),
        manual_review_cost=Decimal("5.00"),
        max_manual_review_rate=0.05,
        false_decline_cost=Decimal("25.00"),
    )

    first = select_policy_thresholds(
        transactions,
        scores,
        policy_config,
        cost_config,
        split_name="validation",
    )
    second = select_policy_thresholds(
        transactions,
        scores,
        policy_config,
        cost_config,
        split_name="validation",
    )

    assert first == second
    assert first.operating_cost.within_review_capacity is True
    assert first.operating_cost.reviews <= 1
    assert first.threshold_candidate_count == 11
    assert first.evaluated_candidate_pairs == 66


def test_threshold_selection_rejects_test_scores() -> None:
    """The public selector must make test-based tuning impossible by default."""
    with pytest.raises(ValueError, match="validation only"):
        select_policy_thresholds(
            (_transaction(0, "100.00", fraud=0),),
            (0.50,),
            load_decision_policy_config(CONFIG_PATH),
            BusinessCostConfig(
                fraud_loss_multiplier=Decimal("1.00"),
                manual_review_cost=Decimal("5.00"),
                max_manual_review_rate=0.05,
                false_decline_cost=Decimal("25.00"),
            ),
            split_name="test",
        )


def test_policy_rejects_non_finite_or_out_of_range_scores() -> None:
    thresholds = DecisionThresholds(0.25, 0.75)

    for score in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="risk_score"):
            apply_decision_policy((score,), thresholds)


def _transaction(transaction_id: int, amount: str, *, fraud: int) -> Transaction:
    return Transaction(
        transaction_id=transaction_id,
        tx_datetime=datetime(2018, 4, 1, tzinfo=UTC)
        + timedelta(seconds=transaction_id),
        customer_id=transaction_id,
        terminal_id=transaction_id,
        tx_amount=Decimal(amount),
        tx_time_seconds=transaction_id,
        tx_time_days=0,
        tx_fraud=fraud,
        tx_fraud_scenario=fraud,
    )
