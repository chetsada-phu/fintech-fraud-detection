"""Tests for framework-free provisional score-to-decision composition."""

from dataclasses import fields
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import pytest
from sklearn.pipeline import Pipeline

import fraud_detection.decisioning.inference as inference_module
import fraud_detection.decisioning.reasons as reasons_module
import fraud_detection.decisioning.scored_inference as scored_inference_module
import fraud_detection.models.artifact as artifact_module
from fraud_detection.decisioning.inference import (
    SelectedDecisionPolicyConfig,
    TransactionDecisionContext,
    TransactionDecisionResult,
)
from fraud_detection.decisioning.policy import Decision
from fraud_detection.decisioning.reasons import (
    HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
    HIGH_TRANSACTION_AMOUNT,
    LIMITED_CUSTOMER_HISTORY,
    UNUSUAL_TRANSACTION_VELOCITY,
    DecisionReasonConfig,
)
from fraud_detection.decisioning.scored_inference import (
    ScoredTransactionDecisionResult,
    score_and_decide_provisional_transaction,
)
from fraud_detection.models.artifact import (
    ARTIFACT_VERSION,
    LoadedProvisionalModel,
    ProvisionalModelMetadata,
    TrainingProvenance,
)
from fraud_detection.models.model_input import (
    XGBOOST_FEATURE_CONTRACT,
    ProvisionalModelInput,
)
from fraud_detection.models.scoring import ProvisionalModelScore

SCORE_SOURCE_LABEL = "provisional test score source"


def test_scored_decision_delegates_exact_context_and_combines_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The use case must preserve exact scorer, decision, and metadata outputs."""
    loaded_model = _loaded_model(_PredictOnlyPipeline(0.72))
    model_input = _model_input()
    policy_config = _policy_config()
    reason_config = _reason_config()
    observed: dict[str, object] = {}

    def score(model: object, input_record: object) -> ProvisionalModelScore:
        observed["score_arguments"] = (model, input_record)
        return _model_score(0.72)

    def decide(
        context: TransactionDecisionContext,
        risk_score: float,
        policy: SelectedDecisionPolicyConfig,
        reasons: DecisionReasonConfig,
    ) -> TransactionDecisionResult:
        observed["decision_arguments"] = (
            context,
            risk_score,
            policy,
            reasons,
        )
        return TransactionDecisionResult(
            transaction_id=42,
            risk_score=0.72,
            decision=Decision.REVIEW,
            reason_codes=(UNUSUAL_TRANSACTION_VELOCITY,),
            policy_version="fixture-v1",
        )

    monkeypatch.setattr(scored_inference_module, "score_provisional_model", score)
    monkeypatch.setattr(scored_inference_module, "decide_context_from_score", decide)

    result = score_and_decide_provisional_transaction(
        loaded_model,
        model_input,
        policy_config,
        reason_config,
    )

    assert observed["score_arguments"] == (loaded_model, model_input)
    assert observed["decision_arguments"] == (
        TransactionDecisionContext(
            transaction_id=42,
            transaction_amount=Decimal("300.00"),
            customer_tx_count_short_window=3,
            customer_amount_mean_prior=Decimal("100.00"),
            customer_amount_deviation_from_mean_prior=Decimal("200.00"),
            customer_seconds_since_previous=60,
        ),
        0.72,
        policy_config,
        reason_config,
    )
    assert result == ScoredTransactionDecisionResult(
        transaction_id=42,
        risk_score=0.72,
        decision=Decision.REVIEW,
        reason_codes=(UNUSUAL_TRANSACTION_VELOCITY,),
        policy_version="fixture-v1",
        score_source_label=SCORE_SOURCE_LABEL,
        artifact_version=ARTIFACT_VERSION,
        artifact_sha256="1" * 64,
        xgboost_config_sha256="2" * 64,
    )


def test_scored_decision_preserves_configured_reason_order() -> None:
    """Real decisioning must retain the deterministic risky-reason priority."""
    result = score_and_decide_provisional_transaction(
        _loaded_model(_PredictOnlyPipeline(0.75)),
        _model_input(),
        _policy_config(),
        _reason_config(),
    )

    assert result.decision is Decision.DECLINE
    assert result.reason_codes == (
        HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
        HIGH_TRANSACTION_AMOUNT,
        UNUSUAL_TRANSACTION_VELOCITY,
    )


def test_score_source_mismatch_fails_before_prediction() -> None:
    """An artifact cannot be paired with a policy for another score source."""
    pipeline = _PredictOnlyPipeline(0.75)
    policy_config = SelectedDecisionPolicyConfig(
        policy_version="fixture-v1",
        selection_split="validation",
        score_source_label="different score source",
        review_threshold=Decimal("0.25"),
        decline_threshold=Decimal("0.75"),
    )

    with pytest.raises(ValueError, match="score-source labels must match"):
        score_and_decide_provisional_transaction(
            _loaded_model(pipeline),
            _model_input(),
            policy_config,
            _reason_config(),
        )

    assert pipeline.prediction_calls == 0


def test_use_case_is_label_free_and_never_fits_or_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composition must depend only on supplied immutable runtime objects."""

    def unexpected_call(*args: object, **kwargs: object) -> None:
        pytest.fail(f"unexpected fit or load call: {args}, {kwargs}")

    monkeypatch.setattr(
        inference_module,
        "load_selected_decision_policy_config",
        unexpected_call,
    )
    monkeypatch.setattr(
        reasons_module,
        "load_decision_reason_config",
        unexpected_call,
    )
    monkeypatch.setattr(artifact_module.joblib, "load", unexpected_call)
    pipeline = _PredictOnlyPipeline(0.10)

    result = score_and_decide_provisional_transaction(
        _loaded_model(pipeline),
        _model_input(),
        _policy_config(),
        _reason_config(),
    )

    label_fields = {"tx_fraud", "tx_fraud_scenario"}
    assert {field.name for field in fields(ProvisionalModelInput)}.isdisjoint(
        label_fields
    )
    assert {field.name for field in fields(ScoredTransactionDecisionResult)}.isdisjoint(
        label_fields
    )
    assert result.decision is Decision.APPROVE
    assert pipeline.prediction_calls == 1


class _Probabilities:
    shape = (1, 2)

    def __init__(self, risk_score: float) -> None:
        self.risk_score = risk_score

    def __getitem__(self, key: tuple[int, int]) -> float:
        if key == (0, 1):
            return self.risk_score
        raise IndexError(key)


class _PredictOnlyPipeline:
    def __init__(self, risk_score: float) -> None:
        self.risk_score = risk_score
        self.prediction_calls = 0

    def predict_proba(self, values: object) -> _Probabilities:
        self.prediction_calls += 1
        return _Probabilities(self.risk_score)

    def fit(self, *args: object, **kwargs: object) -> None:
        pytest.fail(f"pipeline.fit was called: {args}, {kwargs}")


def _loaded_model(pipeline: _PredictOnlyPipeline) -> LoadedProvisionalModel:
    return LoadedProvisionalModel(
        pipeline=cast(Pipeline, pipeline),
        metadata=ProvisionalModelMetadata(
            artifact_version=ARTIFACT_VERSION,
            artifact_sha256="1" * 64,
            xgboost_config_sha256="2" * 64,
            score_source_label=SCORE_SOURCE_LABEL,
            feature_contract=XGBOOST_FEATURE_CONTRACT,
            training_provenance=TrainingProvenance(
                split="train",
                transaction_rows=30,
                fraud_rows=5,
                transactions_sha256="3" * 64,
                temporal_features_sha256="4" * 64,
            ),
        ),
    )


def _model_score(risk_score: float) -> ProvisionalModelScore:
    return ProvisionalModelScore(
        transaction_id=42,
        risk_score=risk_score,
        score_source_label=SCORE_SOURCE_LABEL,
        artifact_version=ARTIFACT_VERSION,
        artifact_sha256="1" * 64,
        xgboost_config_sha256="2" * 64,
    )


def _model_input() -> ProvisionalModelInput:
    return ProvisionalModelInput(
        transaction_id=42,
        tx_amount=Decimal("300.00"),
        tx_time_days=2,
        tx_datetime=datetime(2018, 4, 3, 6, tzinfo=UTC),
        customer_tx_count_short_window=3,
        customer_tx_count_long_window=7,
        customer_amount_mean_prior=Decimal("100.00"),
        customer_amount_deviation_from_mean_prior=Decimal("200.00"),
        customer_seconds_since_previous=60,
        customer_id=12,
        terminal_id=8,
    )


def _policy_config() -> SelectedDecisionPolicyConfig:
    return SelectedDecisionPolicyConfig(
        policy_version="fixture-v1",
        selection_split="validation",
        score_source_label=SCORE_SOURCE_LABEL,
        review_threshold=Decimal("0.25"),
        decline_threshold=Decimal("0.75"),
    )


def _reason_config() -> DecisionReasonConfig:
    return DecisionReasonConfig(
        high_transaction_amount_threshold=Decimal("220.00"),
        customer_amount_ratio_threshold=Decimal("3.00"),
        customer_short_window_count_threshold=3,
        priority_order=(
            HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
            HIGH_TRANSACTION_AMOUNT,
            UNUSUAL_TRANSACTION_VELOCITY,
            LIMITED_CUSTOMER_HISTORY,
        ),
        max_feature_reason_codes=3,
    )
