"""Tests for transaction data-contract validation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from fraud_detection.data.schema import (
    SECONDS_PER_DAY,
    Transaction,
    TransactionValidationError,
    validate_transactions,
)

START = datetime(2018, 4, 1, tzinfo=UTC)


def _transaction(
    transaction_id: int,
    seconds: int,
    *,
    fraud: int = 0,
    scenario: int = 0,
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id,
        tx_datetime=START + timedelta(seconds=seconds),
        customer_id=1,
        terminal_id=2,
        tx_amount=Decimal("10.00"),
        tx_time_seconds=seconds,
        tx_time_days=seconds // SECONDS_PER_DAY,
        tx_fraud=fraud,
        tx_fraud_scenario=scenario,
    )


def test_validator_accepts_consistent_chronological_records() -> None:
    """A well-formed chronological collection should pass without error."""
    validate_transactions([_transaction(0, 10), _transaction(1, 20)], START)


def test_validator_rejects_label_scenario_disagreement() -> None:
    """A post-event label and its scenario must describe the same outcome."""
    with pytest.raises(TransactionValidationError, match="disagree"):
        validate_transactions([_transaction(0, 10, fraud=0, scenario=2)], START)


def test_validator_rejects_future_to_past_row_order() -> None:
    """Later rows may not move backward in event time."""
    with pytest.raises(TransactionValidationError, match="sorted chronologically"):
        validate_transactions([_transaction(0, 20), _transaction(1, 10)], START)
