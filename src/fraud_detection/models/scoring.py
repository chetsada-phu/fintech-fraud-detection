"""Framework-free single-record scoring for a preloaded provisional model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fraud_detection.models.artifact import LoadedProvisionalModel
from fraud_detection.models.model_input import ProvisionalModelInput


@dataclass(frozen=True, slots=True)
class ProvisionalModelScore:
    """One risk score and the exact artifact identity that produced it."""

    transaction_id: int
    risk_score: float
    score_source_label: str
    artifact_version: str
    artifact_sha256: str
    xgboost_config_sha256: str


def score_provisional_model(
    loaded_model: LoadedProvisionalModel,
    model_input: ProvisionalModelInput,
) -> ProvisionalModelScore:
    """Score one precomputed record without loading or fitting a model."""
    probabilities = loaded_model.pipeline.predict_proba(
        (model_input.to_feature_values(),)
    )
    if probabilities.shape != (1, 2):
        raise ValueError("provisional pipeline must return two-class probabilities")
    risk_score = float(probabilities[0, 1])
    if not math.isfinite(risk_score) or not 0 <= risk_score <= 1:
        raise ValueError("provisional pipeline returned an invalid risk score")
    metadata = loaded_model.metadata
    return ProvisionalModelScore(
        transaction_id=model_input.transaction_id,
        risk_score=risk_score,
        score_source_label=metadata.score_source_label,
        artifact_version=metadata.artifact_version,
        artifact_sha256=metadata.artifact_sha256,
        xgboost_config_sha256=metadata.xgboost_config_sha256,
    )
