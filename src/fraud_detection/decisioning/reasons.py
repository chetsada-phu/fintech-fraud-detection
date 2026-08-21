"""Deterministic feature-derived reasons for risky policy decisions."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

from fraud_detection.decisioning.policy import (
    RISK_SCORE_DECLINE,
    RISK_SCORE_REVIEW,
    Decision,
    PolicyDecision,
)
from fraud_detection.features.matrix import JoinedFeatureRow

HIGH_AMOUNT_VS_CUSTOMER_BASELINE: Final = "HIGH_AMOUNT_VS_CUSTOMER_BASELINE"
HIGH_TRANSACTION_AMOUNT: Final = "HIGH_TRANSACTION_AMOUNT"
UNUSUAL_TRANSACTION_VELOCITY: Final = "UNUSUAL_TRANSACTION_VELOCITY"
LIMITED_CUSTOMER_HISTORY: Final = "LIMITED_CUSTOMER_HISTORY"
SUPPORTED_FEATURE_REASON_CODES: Final = (
    HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
    HIGH_TRANSACTION_AMOUNT,
    UNUSUAL_TRANSACTION_VELOCITY,
    LIMITED_CUSTOMER_HISTORY,
)


@dataclass(frozen=True, slots=True)
class DecisionReasonConfig:
    """Versioned explanation boundaries, priority, and output cap."""

    high_transaction_amount_threshold: Decimal
    customer_amount_ratio_threshold: Decimal
    customer_short_window_count_threshold: int
    priority_order: tuple[str, ...]
    max_feature_reason_codes: int

    def __post_init__(self) -> None:
        if (
            not self.high_transaction_amount_threshold.is_finite()
            or self.high_transaction_amount_threshold <= 0
        ):
            raise ValueError(
                "high_transaction_amount_threshold must be finite and positive"
            )
        if (
            not self.customer_amount_ratio_threshold.is_finite()
            or self.customer_amount_ratio_threshold <= 1
        ):
            raise ValueError(
                "customer_amount_ratio_threshold must be finite and greater than 1"
            )
        if (
            type(self.customer_short_window_count_threshold) is not int
            or self.customer_short_window_count_threshold <= 0
        ):
            raise ValueError(
                "customer_short_window_count_threshold must be a positive integer"
            )
        if len(set(self.priority_order)) != len(self.priority_order):
            raise ValueError("priority_order reason codes must be unique")
        if set(self.priority_order) != set(SUPPORTED_FEATURE_REASON_CODES):
            raise ValueError(
                "priority_order must contain every supported feature reason once"
            )
        if type(self.max_feature_reason_codes) is not int or not (
            1 <= self.max_feature_reason_codes <= len(self.priority_order)
        ):
            raise ValueError(
                "max_feature_reason_codes must be between 1 and the priority size"
            )


@dataclass(frozen=True, slots=True)
class DecisionReasonContext:
    """Label-free transaction and past-customer fields used by explanations."""

    transaction_amount: Decimal
    customer_tx_count_short_window: int
    customer_amount_mean_prior: Decimal | None
    customer_amount_deviation_from_mean_prior: Decimal | None
    customer_seconds_since_previous: int | None

    def __post_init__(self) -> None:
        if not self.transaction_amount.is_finite() or self.transaction_amount <= 0:
            raise ValueError("transaction_amount must be finite and positive")
        if (
            type(self.customer_tx_count_short_window) is not int
            or self.customer_tx_count_short_window < 0
        ):
            raise ValueError(
                "customer_tx_count_short_window must be a non-negative integer"
            )
        history_fields = (
            self.customer_amount_mean_prior,
            self.customer_amount_deviation_from_mean_prior,
            self.customer_seconds_since_previous,
        )
        missing_history = tuple(value is None for value in history_fields)
        if len(set(missing_history)) != 1:
            raise ValueError("prior customer-history fields must be missing together")
        if self.customer_amount_mean_prior is not None and (
            not self.customer_amount_mean_prior.is_finite()
            or self.customer_amount_mean_prior <= 0
        ):
            raise ValueError("customer_amount_mean_prior must be finite and positive")
        if self.customer_amount_deviation_from_mean_prior is not None and (
            not self.customer_amount_deviation_from_mean_prior.is_finite()
        ):
            raise ValueError("customer_amount_deviation_from_mean_prior must be finite")
        if self.customer_seconds_since_previous is not None and (
            type(self.customer_seconds_since_previous) is not int
            or self.customer_seconds_since_previous < 0
        ):
            raise ValueError(
                "customer_seconds_since_previous must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class DecisionExplanationSummary:
    """Enriched decisions and aggregate reason-code coverage."""

    decisions: tuple[PolicyDecision, ...]
    reason_counts: tuple[tuple[str, int], ...]
    risky_decisions: int
    feature_explained_decisions: int
    fallback_decisions: int


def load_decision_reason_config(path: Path) -> DecisionReasonConfig:
    """Load and type-check the versioned decision-reason contract."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("decision_reasons")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [decision_reasons] table")
    try:
        raw_priority = section["priority_order"]
        if not isinstance(raw_priority, list) or any(
            not isinstance(value, str) for value in raw_priority
        ):
            raise ValueError("priority_order must be an array of strings")
        return DecisionReasonConfig(
            high_transaction_amount_threshold=_require_decimal_string(
                section, "high_transaction_amount_threshold"
            ),
            customer_amount_ratio_threshold=_require_decimal_string(
                section, "customer_amount_ratio_threshold"
            ),
            customer_short_window_count_threshold=_require_int(
                section, "customer_short_window_count_threshold"
            ),
            priority_order=tuple(raw_priority),
            max_feature_reason_codes=_require_int(section, "max_feature_reason_codes"),
        )
    except KeyError as error:
        raise ValueError(f"missing decision-reason setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid decision-reason configuration: {error}") from error


def explain_policy_decisions(
    rows: Sequence[JoinedFeatureRow],
    decisions: Sequence[PolicyDecision],
    config: DecisionReasonConfig,
) -> DecisionExplanationSummary:
    """Enrich aligned risky decisions using transaction-time features only."""
    if not rows:
        raise ValueError("joined feature rows must not be empty")
    if len(rows) != len(decisions):
        raise ValueError("joined feature rows and decisions must have equal lengths")

    explained = []
    for row, decision in zip(rows, decisions, strict=True):
        if row.transaction.transaction_id != row.temporal.transaction_id:
            raise ValueError("transaction and temporal feature IDs must align")
        explained.append(
            explain_policy_decision(
                DecisionReasonContext(
                    transaction_amount=row.transaction.tx_amount,
                    customer_tx_count_short_window=(
                        row.temporal.customer_tx_count_short_window
                    ),
                    customer_amount_mean_prior=(
                        row.temporal.customer_amount_mean_prior
                    ),
                    customer_amount_deviation_from_mean_prior=(
                        row.temporal.customer_amount_deviation_from_mean_prior
                    ),
                    customer_seconds_since_previous=(
                        row.temporal.customer_seconds_since_previous
                    ),
                ),
                decision,
                config,
            )
        )

    explained_tuple = tuple(explained)
    reason_order = (
        *config.priority_order,
        RISK_SCORE_REVIEW,
        RISK_SCORE_DECLINE,
    )
    reason_counts = tuple(
        (
            reason_code,
            sum(reason_code in decision.reason_codes for decision in explained_tuple),
        )
        for reason_code in reason_order
    )
    risky_decisions = sum(
        decision.decision is not Decision.APPROVE for decision in explained_tuple
    )
    feature_explained = sum(
        decision.decision is not Decision.APPROVE
        and any(
            reason_code in SUPPORTED_FEATURE_REASON_CODES
            for reason_code in decision.reason_codes
        )
        for decision in explained_tuple
    )
    fallback_decisions = sum(
        RISK_SCORE_REVIEW in decision.reason_codes
        or RISK_SCORE_DECLINE in decision.reason_codes
        for decision in explained_tuple
    )
    return DecisionExplanationSummary(
        decisions=explained_tuple,
        reason_counts=reason_counts,
        risky_decisions=risky_decisions,
        feature_explained_decisions=feature_explained,
        fallback_decisions=fallback_decisions,
    )


def explain_policy_decision(
    context: DecisionReasonContext,
    decision: PolicyDecision,
    config: DecisionReasonConfig,
) -> PolicyDecision:
    """Explain one risky decision from label-free decision-time context."""
    if decision.decision is Decision.APPROVE:
        return replace(decision, reason_codes=())
    feature_reasons = _feature_reasons(context, config)
    reasons = (
        feature_reasons[: config.max_feature_reason_codes]
        if feature_reasons
        else _score_fallback(decision.decision)
    )
    return replace(decision, reason_codes=reasons)


def _feature_reasons(
    context: DecisionReasonContext, config: DecisionReasonConfig
) -> tuple[str, ...]:
    customer_history_missing = context.customer_amount_mean_prior is None
    matches = {
        HIGH_AMOUNT_VS_CUSTOMER_BASELINE: (
            not customer_history_missing
            and context.transaction_amount
            >= context.customer_amount_mean_prior
            * config.customer_amount_ratio_threshold
        ),
        HIGH_TRANSACTION_AMOUNT: (
            context.transaction_amount > config.high_transaction_amount_threshold
        ),
        UNUSUAL_TRANSACTION_VELOCITY: (
            context.customer_tx_count_short_window
            >= config.customer_short_window_count_threshold
        ),
        LIMITED_CUSTOMER_HISTORY: customer_history_missing,
    }
    return tuple(
        reason_code for reason_code in config.priority_order if matches[reason_code]
    )


def _score_fallback(decision: Decision) -> tuple[str, ...]:
    if decision is Decision.REVIEW:
        return (RISK_SCORE_REVIEW,)
    if decision is Decision.DECLINE:
        return (RISK_SCORE_DECLINE,)
    return ()


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
