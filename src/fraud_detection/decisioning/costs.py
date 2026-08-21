"""Simulated business-cost assumptions for baseline comparison."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fraud_detection.data.schema import Transaction


@dataclass(frozen=True, slots=True)
class BusinessCostConfig:
    """Versioned simulated costs and manual-review capacity."""

    fraud_loss_multiplier: Decimal
    manual_review_cost: Decimal
    max_manual_review_rate: float
    false_decline_cost: Decimal

    def __post_init__(self) -> None:
        decimal_values = (
            self.fraud_loss_multiplier,
            self.manual_review_cost,
            self.false_decline_cost,
        )
        if any(not value.is_finite() or value < 0 for value in decimal_values):
            raise ValueError("business cost values must be finite and non-negative")
        if self.fraud_loss_multiplier == 0:
            raise ValueError("fraud_loss_multiplier must be positive")
        if not math.isfinite(self.max_manual_review_rate) or not (
            0 < self.max_manual_review_rate <= 1
        ):
            raise ValueError("max_manual_review_rate must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class OperatingCost:
    """Measured simulated cost for one binary-flag evaluation."""

    missed_fraud_loss: Decimal
    manual_review_cost: Decimal
    total: Decimal
    total_per_1000_transactions: Decimal
    within_review_capacity: bool


def load_business_cost_config(path: Path) -> BusinessCostConfig:
    """Load and type-check simulated business assumptions from TOML."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("business_costs")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [business_costs] table")

    try:
        return BusinessCostConfig(
            fraud_loss_multiplier=_require_decimal_string(
                section, "fraud_loss_multiplier"
            ),
            manual_review_cost=_require_decimal_string(section, "manual_review_cost"),
            max_manual_review_rate=_require_number(section, "max_manual_review_rate"),
            false_decline_cost=_require_decimal_string(section, "false_decline_cost"),
        )
    except KeyError as error:
        raise ValueError(f"missing business cost setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid business cost configuration: {error}") from error


def calculate_operating_cost(
    transactions: Sequence[Transaction],
    flags: Sequence[bool],
    config: BusinessCostConfig,
) -> OperatingCost:
    """Calculate binary-review cost without applying future decline costs."""
    if not transactions:
        raise ValueError("at least one transaction is required for cost evaluation")
    if len(transactions) != len(flags):
        raise ValueError("transactions and flags must have equal lengths")

    missed_fraud_amount = sum(
        (
            transaction.tx_amount
            for transaction, flagged in zip(transactions, flags, strict=True)
            if transaction.tx_fraud == 1 and not flagged
        ),
        start=Decimal(0),
    )
    missed_fraud_loss = missed_fraud_amount * config.fraud_loss_multiplier
    review_cost = config.manual_review_cost * sum(flags)
    total = missed_fraud_loss + review_cost
    total_per_1000 = (total * Decimal(1000)) / Decimal(len(transactions))
    review_rate = sum(flags) / len(flags)
    return OperatingCost(
        missed_fraud_loss=missed_fraud_loss,
        manual_review_cost=review_cost,
        total=total,
        total_per_1000_transactions=total_per_1000,
        within_review_capacity=review_rate <= config.max_manual_review_rate,
    )


def _require_decimal_string(section: Mapping[str, object], key: str) -> Decimal:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a decimal string")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{key} must be a valid decimal string") from error


def _require_number(section: Mapping[str, object], key: str) -> float:
    value = section[key]
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be a number")
    return float(value)
