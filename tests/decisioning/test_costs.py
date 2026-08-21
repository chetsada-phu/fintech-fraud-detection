"""Tests for versioned simulated business-cost arithmetic."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.costs import (
    BusinessCostConfig,
    calculate_operating_cost,
    load_business_cost_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "business_costs.toml"


def test_versioned_business_costs_load_as_simulated_assumptions() -> None:
    """The repository config must preserve exact money values and capacity."""
    config = load_business_cost_config(CONFIG_PATH)

    assert config == BusinessCostConfig(
        fraud_loss_multiplier=Decimal("1.00"),
        manual_review_cost=Decimal("5.00"),
        max_manual_review_rate=0.05,
        false_decline_cost=Decimal("25.00"),
    )


def test_binary_operating_cost_counts_missed_fraud_and_every_review() -> None:
    """Phase 2 cost excludes the reserved false-decline assumption."""
    transactions = (
        _transaction(0, "100.00", fraud=1),
        _transaction(1, "200.00", fraud=1),
        _transaction(2, "50.00", fraud=0),
    )
    config = load_business_cost_config(CONFIG_PATH)

    result = calculate_operating_cost(transactions, (False, True, True), config)

    assert result.missed_fraud_loss == Decimal("100.0000")
    assert result.manual_review_cost == Decimal("10.00")
    assert result.total == Decimal("110.0000")
    assert result.total_per_1000_transactions == Decimal("110000.0000") / 3
    assert result.within_review_capacity is False


def test_operating_cost_rejects_misaligned_flags() -> None:
    """Every evaluated transaction must have exactly one binary flag."""
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_operating_cost(
            (_transaction(0, "100.00", fraud=1),),
            (),
            load_business_cost_config(CONFIG_PATH),
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
