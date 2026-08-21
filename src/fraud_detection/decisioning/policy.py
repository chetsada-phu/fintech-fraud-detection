"""Model-agnostic three-way fraud decision policy and threshold selection."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Final

from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.costs import BusinessCostConfig

RISK_SCORE_REVIEW: Final = "RISK_SCORE_REVIEW"
RISK_SCORE_DECLINE: Final = "RISK_SCORE_DECLINE"


class Decision(StrEnum):
    """Operational action produced from one risk score."""

    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


@dataclass(frozen=True, slots=True)
class DecisionPolicyConfig:
    """Versioned validation-only threshold-search contract."""

    selection_split: str
    score_source_label: str
    threshold_grid_step: Decimal

    def __post_init__(self) -> None:
        if self.selection_split != "validation":
            raise ValueError("selection_split must be 'validation'")
        if not self.score_source_label.strip():
            raise ValueError("score_source_label must not be empty")
        if (
            not self.threshold_grid_step.is_finite()
            or self.threshold_grid_step <= 0
            or self.threshold_grid_step > 1
        ):
            raise ValueError("threshold_grid_step must be within (0, 1]")
        if Decimal(1) % self.threshold_grid_step != 0:
            raise ValueError("threshold_grid_step must divide 1 exactly")

    @property
    def threshold_candidates(self) -> tuple[float, ...]:
        """Return the fixed inclusive threshold grid from zero to one."""
        steps = int(Decimal(1) / self.threshold_grid_step)
        return tuple(
            float(self.threshold_grid_step * index) for index in range(steps + 1)
        )


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Exact boundaries for approve, review, and decline decisions."""

    review_threshold: float
    decline_threshold: float

    def __post_init__(self) -> None:
        for name, value in (
            ("review_threshold", self.review_threshold),
            ("decline_threshold", self.decline_threshold),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.review_threshold > self.decline_threshold:
            raise ValueError("review_threshold must not exceed decline_threshold")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One supplied score converted to an operational action and reasons."""

    risk_score: float
    decision: Decision
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyOperatingCost:
    """Simulated three-way operating result for one labeled validation batch."""

    approvals: int
    reviews: int
    declines: int
    false_declines: int
    missed_fraud_loss: Decimal
    manual_review_cost: Decimal
    false_decline_cost: Decimal
    total: Decimal
    total_per_1000_transactions: Decimal
    fraud_amount_captured: Decimal
    review_rate: float
    false_decline_rate: float
    within_review_capacity: bool


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """Best capacity-feasible policy under the versioned validation search."""

    thresholds: DecisionThresholds
    decisions: tuple[PolicyDecision, ...]
    operating_cost: PolicyOperatingCost
    threshold_candidate_count: int
    evaluated_candidate_pairs: int


def load_decision_policy_config(path: Path) -> DecisionPolicyConfig:
    """Load and type-check the versioned decision-policy search contract."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("decision_policy")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [decision_policy] table")
    try:
        return DecisionPolicyConfig(
            selection_split=_require_str(section, "selection_split"),
            score_source_label=_require_str(section, "score_source_label"),
            threshold_grid_step=_require_decimal_string(section, "threshold_grid_step"),
        )
    except KeyError as error:
        raise ValueError(f"missing decision-policy setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid decision-policy configuration: {error}") from error


def decide_risk_score(
    risk_score: float, thresholds: DecisionThresholds
) -> PolicyDecision:
    """Apply exact score boundaries to one model-agnostic risk score."""
    _validate_risk_score(risk_score)
    decision = _decision_for_validated_score(risk_score, thresholds)
    return PolicyDecision(
        risk_score=risk_score,
        decision=decision,
        reason_codes=_reason_codes(decision),
    )


def apply_decision_policy(
    risk_scores: Sequence[float], thresholds: DecisionThresholds
) -> tuple[PolicyDecision, ...]:
    """Apply one fixed policy to aligned supplied risk scores."""
    if not risk_scores:
        raise ValueError("risk_scores must not be empty")
    return tuple(decide_risk_score(score, thresholds) for score in risk_scores)


def calculate_policy_operating_cost(
    transactions: Sequence[Transaction],
    decisions: Sequence[PolicyDecision],
    config: BusinessCostConfig,
) -> PolicyOperatingCost:
    """Calculate simulated approve/review/decline cost for aligned rows."""
    if not transactions:
        raise ValueError("at least one transaction is required for policy evaluation")
    if len(transactions) != len(decisions):
        raise ValueError("transactions and decisions must have equal lengths")
    return _calculate_cost_from_actions(
        transactions,
        tuple(result.decision for result in decisions),
        config,
    )


def select_policy_thresholds(
    transactions: Sequence[Transaction],
    risk_scores: Sequence[float],
    policy_config: DecisionPolicyConfig,
    cost_config: BusinessCostConfig,
    *,
    split_name: str,
) -> PolicySelection:
    """Select capacity-feasible thresholds from supplied validation scores only."""
    if split_name != "validation" or split_name != policy_config.selection_split:
        raise ValueError(
            "decision-policy thresholds may be selected on validation only"
        )
    if not transactions:
        raise ValueError("at least one validation transaction is required")
    if len(transactions) != len(risk_scores):
        raise ValueError("transactions and risk_scores must have equal lengths")
    for risk_score in risk_scores:
        _validate_risk_score(risk_score)

    candidates = policy_config.threshold_candidates
    evaluated_pairs = 0
    best_thresholds: DecisionThresholds | None = None
    best_cost: PolicyOperatingCost | None = None
    best_key: tuple[Decimal, int, int, int, float, float] | None = None

    for review_index, review_threshold in enumerate(candidates):
        for decline_threshold in candidates[review_index:]:
            evaluated_pairs += 1
            thresholds = DecisionThresholds(review_threshold, decline_threshold)
            actions = tuple(
                _decision_for_validated_score(score, thresholds)
                for score in risk_scores
            )
            operating_cost = _calculate_cost_from_actions(
                transactions, actions, cost_config
            )
            if not operating_cost.within_review_capacity:
                continue
            key = (
                operating_cost.total,
                operating_cost.false_declines,
                operating_cost.reviews,
                operating_cost.declines,
                -review_threshold,
                -decline_threshold,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_thresholds = thresholds
                best_cost = operating_cost

    if best_thresholds is None or best_cost is None:
        raise ValueError("no threshold pair satisfies manual-review capacity")
    decisions = apply_decision_policy(risk_scores, best_thresholds)
    return PolicySelection(
        thresholds=best_thresholds,
        decisions=decisions,
        operating_cost=best_cost,
        threshold_candidate_count=len(candidates),
        evaluated_candidate_pairs=evaluated_pairs,
    )


def _calculate_cost_from_actions(
    transactions: Sequence[Transaction],
    actions: Sequence[Decision],
    config: BusinessCostConfig,
) -> PolicyOperatingCost:
    if len(transactions) != len(actions):
        raise ValueError("transactions and actions must have equal lengths")
    approvals = sum(action is Decision.APPROVE for action in actions)
    reviews = sum(action is Decision.REVIEW for action in actions)
    declines = sum(action is Decision.DECLINE for action in actions)
    false_declines = sum(
        transaction.tx_fraud == 0 and action is Decision.DECLINE
        for transaction, action in zip(transactions, actions, strict=True)
    )
    missed_fraud_amount = sum(
        (
            transaction.tx_amount
            for transaction, action in zip(transactions, actions, strict=True)
            if transaction.tx_fraud == 1 and action is Decision.APPROVE
        ),
        start=Decimal(0),
    )
    fraud_amount_captured = sum(
        (
            transaction.tx_amount
            for transaction, action in zip(transactions, actions, strict=True)
            if transaction.tx_fraud == 1 and action is not Decision.APPROVE
        ),
        start=Decimal(0),
    )
    missed_fraud_loss = missed_fraud_amount * config.fraud_loss_multiplier
    review_cost = config.manual_review_cost * reviews
    decline_cost = config.false_decline_cost * false_declines
    total = missed_fraud_loss + review_cost + decline_cost
    row_count = len(transactions)
    legitimate_count = sum(transaction.tx_fraud == 0 for transaction in transactions)
    review_rate = reviews / row_count
    false_decline_rate = false_declines / legitimate_count if legitimate_count else 0.0
    return PolicyOperatingCost(
        approvals=approvals,
        reviews=reviews,
        declines=declines,
        false_declines=false_declines,
        missed_fraud_loss=missed_fraud_loss,
        manual_review_cost=review_cost,
        false_decline_cost=decline_cost,
        total=total,
        total_per_1000_transactions=(total * Decimal(1000)) / Decimal(row_count),
        fraud_amount_captured=fraud_amount_captured,
        review_rate=review_rate,
        false_decline_rate=false_decline_rate,
        within_review_capacity=review_rate <= config.max_manual_review_rate,
    )


def _decision_for_validated_score(
    risk_score: float, thresholds: DecisionThresholds
) -> Decision:
    if risk_score >= thresholds.decline_threshold:
        return Decision.DECLINE
    if risk_score >= thresholds.review_threshold:
        return Decision.REVIEW
    return Decision.APPROVE


def _reason_codes(decision: Decision) -> tuple[str, ...]:
    if decision is Decision.REVIEW:
        return (RISK_SCORE_REVIEW,)
    if decision is Decision.DECLINE:
        return (RISK_SCORE_DECLINE,)
    return ()


def _validate_risk_score(risk_score: float) -> None:
    if not math.isfinite(risk_score) or not 0 <= risk_score <= 1:
        raise ValueError("risk_score must be finite and within [0, 1]")


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
