"""Transparent decision-time rules for the Phase 2 fraud baseline."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

from fraud_detection.data.schema import Transaction

HIGH_TRANSACTION_AMOUNT: Final = "HIGH_TRANSACTION_AMOUNT"
UNUSUAL_TRANSACTION_HOUR: Final = "UNUSUAL_TRANSACTION_HOUR"
REASON_CODE_ORDER: Final = (
    HIGH_TRANSACTION_AMOUNT,
    UNUSUAL_TRANSACTION_HOUR,
)


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Versioned thresholds for the transparent rule baseline."""

    high_amount_threshold: Decimal
    unusual_hour_start_utc: int
    unusual_hour_end_utc: int

    def __post_init__(self) -> None:
        if (
            not self.high_amount_threshold.is_finite()
            or self.high_amount_threshold <= 0
        ):
            raise ValueError("high_amount_threshold must be finite and positive")
        if type(self.unusual_hour_start_utc) is not int or not (
            0 <= self.unusual_hour_start_utc <= 23
        ):
            raise ValueError("unusual_hour_start_utc must be an integer from 0 to 23")
        if type(self.unusual_hour_end_utc) is not int or not (
            0 <= self.unusual_hour_end_utc <= 23
        ):
            raise ValueError("unusual_hour_end_utc must be an integer from 0 to 23")
        if self.unusual_hour_start_utc <= self.unusual_hour_end_utc:
            raise ValueError("the unusual-hour interval must cross midnight")


@dataclass(frozen=True, slots=True)
class RuleResult:
    """A binary rule flag and its stable, ordered explanations."""

    flagged: bool
    reason_codes: tuple[str, ...]


def load_rule_config(path: Path) -> RuleConfig:
    """Load and type-check the versioned rule baseline configuration."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("rule_baseline")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [rule_baseline] table")

    try:
        return RuleConfig(
            high_amount_threshold=_require_decimal_string(
                section, "high_amount_threshold"
            ),
            unusual_hour_start_utc=_require_int(section, "unusual_hour_start_utc"),
            unusual_hour_end_utc=_require_int(section, "unusual_hour_end_utc"),
        )
    except KeyError as error:
        raise ValueError(f"missing rule setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid rule configuration: {error}") from error


def apply_rules(transaction: Transaction, config: RuleConfig) -> RuleResult:
    """Flag one transaction using decision-time fields only."""
    reason_codes = []
    if transaction.tx_amount > config.high_amount_threshold:
        reason_codes.append(HIGH_TRANSACTION_AMOUNT)

    hour = transaction.tx_datetime.hour
    if hour >= config.unusual_hour_start_utc or hour < config.unusual_hour_end_utc:
        reason_codes.append(UNUSUAL_TRANSACTION_HOUR)

    return RuleResult(
        flagged=bool(reason_codes),
        reason_codes=tuple(reason_codes),
    )


def _require_decimal_string(section: Mapping[str, object], key: str) -> Decimal:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a decimal string")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{key} must be a valid decimal string") from error


def _require_int(section: Mapping[str, object], key: str) -> int:
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value
