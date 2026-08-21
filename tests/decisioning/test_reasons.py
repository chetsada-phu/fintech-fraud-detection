"""Tests for deterministic feature-derived decision reasons."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.policy import (
    RISK_SCORE_DECLINE,
    RISK_SCORE_REVIEW,
    DecisionThresholds,
    decide_risk_score,
)
from fraud_detection.decisioning.reasons import (
    HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
    HIGH_TRANSACTION_AMOUNT,
    LIMITED_CUSTOMER_HISTORY,
    UNUSUAL_TRANSACTION_VELOCITY,
    DecisionReasonConfig,
    explain_policy_decisions,
    load_decision_reason_config,
)
from fraud_detection.features.matrix import JoinedFeatureRow
from fraud_detection.features.temporal import TemporalFeatureRow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_reasons.toml"


def test_versioned_reason_boundaries_and_priority_load_exactly() -> None:
    assert load_decision_reason_config(CONFIG_PATH) == DecisionReasonConfig(
        high_transaction_amount_threshold=Decimal("220.00"),
        customer_amount_ratio_threshold=Decimal("3.00"),
        customer_short_window_count_threshold=3,
        priority_order=(
            HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
            HIGH_TRANSACTION_AMOUNT,
            UNUSUAL_TRANSACTION_VELOCITY,
            LIMITED_CUSTOMER_HISTORY,
        ),
        max_feature_reason_codes=3,
    )


def test_exact_feature_boundaries_follow_configured_priority() -> None:
    """Ratio and velocity include equality; absolute amount is strictly above."""
    config = load_decision_reason_config(CONFIG_PATH)
    decline = decide_risk_score(0.90, DecisionThresholds(0.25, 0.75))
    all_three = explain_policy_decisions(
        (_joined_row("300.00", prior_mean="100.00", short_count=3),),
        (decline,),
        config,
    )
    amount_boundary = explain_policy_decisions(
        (_joined_row("220.00", prior_mean="100.00", short_count=2),),
        (decline,),
        config,
    )
    amount_above = explain_policy_decisions(
        (_joined_row("220.01", prior_mean="100.00", short_count=2),),
        (decline,),
        config,
    )

    assert all_three.decisions[0].reason_codes == (
        HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
        HIGH_TRANSACTION_AMOUNT,
        UNUSUAL_TRANSACTION_VELOCITY,
    )
    assert amount_boundary.decisions[0].reason_codes == (RISK_SCORE_DECLINE,)
    assert amount_above.decisions[0].reason_codes == (HIGH_TRANSACTION_AMOUNT,)


def test_reason_cap_missing_history_and_score_fallback_are_explicit() -> None:
    """Feature reasons are capped, while unmatched risky scores keep a fallback."""
    base_config = load_decision_reason_config(CONFIG_PATH)
    capped_config = replace(base_config, max_feature_reason_codes=2)
    thresholds = DecisionThresholds(0.25, 0.75)
    decisions = (
        decide_risk_score(0.90, thresholds),
        decide_risk_score(0.50, thresholds),
        decide_risk_score(0.50, thresholds),
        decide_risk_score(0.10, thresholds),
    )
    rows = (
        _joined_row("300.00", prior_mean="100.00", short_count=3),
        _joined_row("50.00", prior_mean=None, short_count=0),
        _joined_row("100.00", prior_mean="100.00", short_count=0),
        _joined_row("500.00", prior_mean="100.00", short_count=4),
    )

    summary = explain_policy_decisions(rows, decisions, capped_config)

    assert summary.decisions[0].reason_codes == (
        HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
        HIGH_TRANSACTION_AMOUNT,
    )
    assert summary.decisions[1].reason_codes == (LIMITED_CUSTOMER_HISTORY,)
    assert summary.decisions[2].reason_codes == (RISK_SCORE_REVIEW,)
    assert summary.decisions[3].reason_codes == ()
    assert summary.risky_decisions == 3
    assert summary.feature_explained_decisions == 2
    assert summary.fallback_decisions == 1


def test_post_event_labels_cannot_change_feature_reasons() -> None:
    """Reason generation must not inspect fraud labels or fraud scenarios."""
    config = load_decision_reason_config(CONFIG_PATH)
    row = _joined_row("300.00", prior_mean="100.00", short_count=3)
    relabeled = replace(
        row,
        transaction=replace(row.transaction, tx_fraud=1, tx_fraud_scenario=3),
    )
    decision = decide_risk_score(0.90, DecisionThresholds(0.25, 0.75))

    original = explain_policy_decisions((row,), (decision,), config)
    after_relabel = explain_policy_decisions((relabeled,), (decision,), config)

    assert original == after_relabel


def _joined_row(
    amount: str,
    *,
    prior_mean: str | None,
    short_count: int,
) -> JoinedFeatureRow:
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
        customer_tx_count_short_window=short_count,
        customer_tx_count_long_window=short_count,
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
