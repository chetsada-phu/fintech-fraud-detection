"""Tests for isolated validation-only decision-policy cost sensitivity."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.costs import load_business_cost_config
from fraud_detection.decisioning.policy import (
    DecisionPolicyConfig,
    select_policy_thresholds,
)
from fraud_detection.decisioning.policy_sensitivity import (
    CostSensitivityScenario,
    PolicySensitivityConfig,
    analyze_policy_cost_sensitivity,
    load_policy_sensitivity_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_policy_sensitivity.toml"
COST_CONFIG_PATH = PROJECT_ROOT / "configs" / "business_costs.toml"


def test_versioned_scenarios_preserve_predeclared_order() -> None:
    """Every business assumption should vary in one isolated scenario."""
    assert load_policy_sensitivity_config(CONFIG_PATH) == PolicySensitivityConfig(
        evaluation_split="validation",
        base_scenario_key="base",
        scenarios=(
            CostSensitivityScenario("base", "Base assumptions"),
            CostSensitivityScenario(
                "higher_fraud_loss",
                "Higher fraud loss",
                fraud_loss_multiplier=Decimal("2.00"),
            ),
            CostSensitivityScenario(
                "lower_review_cost",
                "Lower review cost",
                manual_review_cost=Decimal("1.00"),
            ),
            CostSensitivityScenario(
                "higher_false_decline_cost",
                "Higher false-decline cost",
                false_decline_cost=Decimal("100.00"),
            ),
            CostSensitivityScenario(
                "tighter_review_capacity",
                "Tighter review capacity",
                max_manual_review_rate=0.01,
            ),
        ),
    )


def test_scenarios_are_isolated_deterministic_and_capacity_feasible() -> None:
    """Scenario overrides must not mutate or leak into the base assumptions."""
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
    sensitivity_config = load_policy_sensitivity_config(CONFIG_PATH)
    base_costs = load_business_cost_config(COST_CONFIG_PATH)
    original_base_costs = base_costs

    first = analyze_policy_cost_sensitivity(
        transactions,
        scores,
        policy_config,
        sensitivity_config,
        base_costs,
        split_name="validation",
    )
    second = analyze_policy_cost_sensitivity(
        transactions,
        scores,
        policy_config,
        sensitivity_config,
        base_costs,
        split_name="validation",
    )

    assert first == second
    assert base_costs == original_base_costs
    assert first.results[0].cost_config == base_costs
    assert first.results[0].cost_config is not base_costs
    assert tuple(result.scenario.key for result in first.results) == (
        "base",
        "higher_fraud_loss",
        "lower_review_cost",
        "higher_false_decline_cost",
        "tighter_review_capacity",
    )
    assert first.results[1].cost_config.fraud_loss_multiplier == Decimal("2.00")
    assert first.results[1].cost_config.manual_review_cost == Decimal("5.00")
    assert first.results[2].cost_config.manual_review_cost == Decimal("1.00")
    assert first.results[2].cost_config.fraud_loss_multiplier == Decimal("1.00")
    assert first.results[3].cost_config.false_decline_cost == Decimal("100.00")
    assert first.results[4].cost_config.max_manual_review_rate == 0.01
    assert all(
        result.selection.operating_cost.within_review_capacity
        for result in first.results
    )
    assert first.results[0].selection == select_policy_thresholds(
        transactions,
        scores,
        policy_config,
        base_costs,
        split_name="validation",
    )


def test_sensitivity_rejects_test_scores() -> None:
    """Scenario analysis must not provide a route to test-based tuning."""
    with pytest.raises(ValueError, match="validation only"):
        analyze_policy_cost_sensitivity(
            (_transaction(0, "100.00", fraud=0),),
            (0.50,),
            DecisionPolicyConfig(
                selection_split="validation",
                score_source_label="Provisional fixture scores",
                threshold_grid_step=Decimal("0.10"),
            ),
            load_policy_sensitivity_config(CONFIG_PATH),
            load_business_cost_config(COST_CONFIG_PATH),
            split_name="test",
        )


def test_sensitivity_config_requires_unmodified_base_first() -> None:
    """The first scenario must be the named base with no hidden overrides."""
    with pytest.raises(ValueError, match="first scenario"):
        PolicySensitivityConfig(
            evaluation_split="validation",
            base_scenario_key="base",
            scenarios=(
                CostSensitivityScenario(
                    "changed", "Changed", manual_review_cost=Decimal("1.00")
                ),
                CostSensitivityScenario("base", "Base"),
            ),
        )


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
