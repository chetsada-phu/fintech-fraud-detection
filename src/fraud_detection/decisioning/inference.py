"""Framework-free one-transaction decision contract for supplied risk scores."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fraud_detection.decisioning.policy import (
    Decision,
    DecisionThresholds,
    decide_risk_score,
)
from fraud_detection.decisioning.reasons import (
    DecisionReasonConfig,
    DecisionReasonContext,
    explain_policy_decision,
)
from fraud_detection.features.matrix import JoinedFeatureRow


@dataclass(frozen=True, slots=True)
class SelectedDecisionPolicyConfig:
    """Versioned thresholds and provenance selected before inference."""

    policy_version: str
    selection_split: str
    score_source_label: str
    review_threshold: Decimal
    decline_threshold: Decimal

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty")
        if self.selection_split != "validation":
            raise ValueError("selection_split must be 'validation'")
        if not self.score_source_label.strip():
            raise ValueError("score_source_label must not be empty")
        for name, value in (
            ("review_threshold", self.review_threshold),
            ("decline_threshold", self.decline_threshold),
        ):
            if not value.is_finite() or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.review_threshold > self.decline_threshold:
            raise ValueError("review_threshold must not exceed decline_threshold")

    @property
    def thresholds(self) -> DecisionThresholds:
        """Return the fixed boundaries used by the model-agnostic policy."""
        return DecisionThresholds(
            review_threshold=float(self.review_threshold),
            decline_threshold=float(self.decline_threshold),
        )


@dataclass(frozen=True, slots=True)
class TransactionDecisionResult:
    """Stable framework-neutral output for one supplied risk score."""

    transaction_id: int
    risk_score: float
    decision: Decision
    reason_codes: tuple[str, ...]
    policy_version: str


@dataclass(frozen=True, slots=True)
class TransactionDecisionContext:
    """Label-free fields needed to decide and explain one supplied score."""

    transaction_id: int
    transaction_amount: Decimal
    customer_tx_count_short_window: int
    customer_amount_mean_prior: Decimal | None
    customer_amount_deviation_from_mean_prior: Decimal | None
    customer_seconds_since_previous: int | None

    def __post_init__(self) -> None:
        if type(self.transaction_id) is not int or self.transaction_id < 0:
            raise ValueError("transaction_id must be a non-negative integer")
        _ = self.reason_context

    @property
    def reason_context(self) -> DecisionReasonContext:
        """Return the validated subset consumed by deterministic reasons."""
        return DecisionReasonContext(
            transaction_amount=self.transaction_amount,
            customer_tx_count_short_window=self.customer_tx_count_short_window,
            customer_amount_mean_prior=self.customer_amount_mean_prior,
            customer_amount_deviation_from_mean_prior=(
                self.customer_amount_deviation_from_mean_prior
            ),
            customer_seconds_since_previous=self.customer_seconds_since_previous,
        )


def load_selected_decision_policy_config(path: Path) -> SelectedDecisionPolicyConfig:
    """Load and validate the frozen provisional policy contract."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("selected_decision_policy")
    if not isinstance(section, dict):
        raise ValueError(
            "configuration must contain a [selected_decision_policy] table"
        )
    try:
        return SelectedDecisionPolicyConfig(
            policy_version=_require_str(section, "policy_version"),
            selection_split=_require_str(section, "selection_split"),
            score_source_label=_require_str(section, "score_source_label"),
            review_threshold=_require_decimal_string(section, "review_threshold"),
            decline_threshold=_require_decimal_string(section, "decline_threshold"),
        )
    except KeyError as error:
        raise ValueError(f"missing selected-policy setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid selected-policy configuration: {error}") from error


def decide_transaction_from_score(
    row: JoinedFeatureRow,
    risk_score: float,
    policy_config: SelectedDecisionPolicyConfig,
    reason_config: DecisionReasonConfig,
) -> TransactionDecisionResult:
    """Combine one externally supplied score with fixed policy and reasons."""
    if row.transaction.transaction_id != row.temporal.transaction_id:
        raise ValueError("transaction and temporal feature IDs must align")
    context = TransactionDecisionContext(
        transaction_id=row.transaction.transaction_id,
        transaction_amount=row.transaction.tx_amount,
        customer_tx_count_short_window=row.temporal.customer_tx_count_short_window,
        customer_amount_mean_prior=row.temporal.customer_amount_mean_prior,
        customer_amount_deviation_from_mean_prior=(
            row.temporal.customer_amount_deviation_from_mean_prior
        ),
        customer_seconds_since_previous=row.temporal.customer_seconds_since_previous,
    )
    return decide_context_from_score(
        context,
        risk_score,
        policy_config,
        reason_config,
    )


def decide_context_from_score(
    context: TransactionDecisionContext,
    risk_score: float,
    policy_config: SelectedDecisionPolicyConfig,
    reason_config: DecisionReasonConfig,
) -> TransactionDecisionResult:
    """Decide one supplied score from label-free decision-time context."""
    base_decision = decide_risk_score(risk_score, policy_config.thresholds)
    explained_decision = explain_policy_decision(
        context.reason_context,
        base_decision,
        reason_config,
    )
    return TransactionDecisionResult(
        transaction_id=context.transaction_id,
        risk_score=explained_decision.risk_score,
        decision=explained_decision.decision,
        reason_codes=explained_decision.reason_codes,
        policy_version=policy_config.policy_version,
    )


def _require_str(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_decimal_string(section: Mapping[str, object], key: str) -> Decimal:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a decimal string")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{key} must be a valid decimal string") from error
