"""Framework-free composition of provisional scoring and risk decisioning."""

from __future__ import annotations

from dataclasses import dataclass

from fraud_detection.decisioning.inference import (
    SelectedDecisionPolicyConfig,
    TransactionDecisionContext,
    decide_context_from_score,
)
from fraud_detection.decisioning.policy import Decision
from fraud_detection.decisioning.reasons import DecisionReasonConfig
from fraud_detection.models.artifact import LoadedProvisionalModel
from fraud_detection.models.model_input import ProvisionalModelInput
from fraud_detection.models.scoring import score_provisional_model


@dataclass(frozen=True, slots=True)
class ScoredTransactionDecisionResult:
    """One model score, policy decision, reasons, and artifact identity."""

    transaction_id: int
    risk_score: float
    decision: Decision
    reason_codes: tuple[str, ...]
    policy_version: str
    score_source_label: str
    artifact_version: str
    artifact_sha256: str
    xgboost_config_sha256: str


def score_and_decide_provisional_transaction(
    loaded_model: LoadedProvisionalModel,
    model_input: ProvisionalModelInput,
    policy_config: SelectedDecisionPolicyConfig,
    reason_config: DecisionReasonConfig,
) -> ScoredTransactionDecisionResult:
    """Score and decide one precomputed input without fitting or loading."""
    if loaded_model.metadata.score_source_label != policy_config.score_source_label:
        raise ValueError("artifact and policy score-source labels must match")

    model_score = score_provisional_model(loaded_model, model_input)
    if model_score.transaction_id != model_input.transaction_id:
        raise ValueError("model score transaction_id does not match the input")
    if model_score.score_source_label != policy_config.score_source_label:
        raise ValueError("model score and policy score-source labels must match")

    decision_result = decide_context_from_score(
        TransactionDecisionContext(
            transaction_id=model_input.transaction_id,
            transaction_amount=model_input.tx_amount,
            customer_tx_count_short_window=(model_input.customer_tx_count_short_window),
            customer_amount_mean_prior=model_input.customer_amount_mean_prior,
            customer_amount_deviation_from_mean_prior=(
                model_input.customer_amount_deviation_from_mean_prior
            ),
            customer_seconds_since_previous=(
                model_input.customer_seconds_since_previous
            ),
        ),
        model_score.risk_score,
        policy_config,
        reason_config,
    )
    if decision_result.transaction_id != model_score.transaction_id:
        raise ValueError("decision transaction_id does not match the model score")
    if decision_result.risk_score != model_score.risk_score:
        raise ValueError("decision risk_score does not match the model score")

    return ScoredTransactionDecisionResult(
        transaction_id=decision_result.transaction_id,
        risk_score=decision_result.risk_score,
        decision=decision_result.decision,
        reason_codes=decision_result.reason_codes,
        policy_version=decision_result.policy_version,
        score_source_label=model_score.score_source_label,
        artifact_version=model_score.artifact_version,
        artifact_sha256=model_score.artifact_sha256,
        xgboost_config_sha256=model_score.xgboost_config_sha256,
    )
