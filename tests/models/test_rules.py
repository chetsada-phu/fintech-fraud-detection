"""Boundary and leakage tests for the transparent rule baseline."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.models.rules import (
    HIGH_TRANSACTION_AMOUNT,
    UNUSUAL_TRANSACTION_HOUR,
    RuleConfig,
    apply_rules,
    load_rule_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "rule_baseline.toml"
CONFIG = RuleConfig(
    high_amount_threshold=Decimal("220.00"),
    unusual_hour_start_utc=23,
    unusual_hour_end_utc=6,
)


@pytest.mark.parametrize(
    ("amount", "hour", "expected_codes"),
    [
        ("220.00", 12, ()),
        ("220.01", 12, (HIGH_TRANSACTION_AMOUNT,)),
        ("100.00", 22, ()),
        ("100.00", 23, (UNUSUAL_TRANSACTION_HOUR,)),
        ("100.00", 5, (UNUSUAL_TRANSACTION_HOUR,)),
        ("100.00", 6, ()),
        (
            "220.01",
            23,
            (HIGH_TRANSACTION_AMOUNT, UNUSUAL_TRANSACTION_HOUR),
        ),
    ],
)
def test_exact_rule_boundaries_and_reason_order(
    amount: str, hour: int, expected_codes: tuple[str, ...]
) -> None:
    """Exact threshold edges must remain explicit and stable."""
    result = apply_rules(_transaction(amount=amount, hour=hour), CONFIG)

    assert result.flagged is bool(expected_codes)
    assert result.reason_codes == expected_codes


def test_post_event_labels_cannot_change_rule_output() -> None:
    """Scoring must not read either post-event fraud label."""
    legitimate = _transaction(amount="220.01", hour=12)
    labeled_fraud = replace(legitimate, tx_fraud=1, tx_fraud_scenario=3)

    assert apply_rules(legitimate, CONFIG) == apply_rules(labeled_fraud, CONFIG)


def test_versioned_rule_config_loads_exact_decimal() -> None:
    """The repository config should preserve the documented money boundary."""
    config = load_rule_config(CONFIG_PATH)

    assert config == CONFIG
    assert config.high_amount_threshold.as_tuple().exponent == -2


def test_rule_config_rejects_non_crossing_hour_window() -> None:
    """The configured overnight interval must cross midnight."""
    with pytest.raises(ValueError, match="cross midnight"):
        RuleConfig(Decimal("220.00"), 6, 23)


def _transaction(*, amount: str, hour: int) -> Transaction:
    return Transaction(
        transaction_id=0,
        tx_datetime=datetime(2018, 4, 1, hour, tzinfo=UTC),
        customer_id=1,
        terminal_id=2,
        tx_amount=Decimal(amount),
        tx_time_seconds=hour * 3_600,
        tx_time_days=0,
        tx_fraud=0,
        tx_fraud_scenario=0,
    )
