"""Integration tests for the artifact-backed scored-decision API."""

from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

import fraud_detection.serving.api as api_module
from fraud_detection.decisioning.reasons import (
    HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
    HIGH_TRANSACTION_AMOUNT,
    UNUSUAL_TRANSACTION_VELOCITY,
)
from fraud_detection.models.artifact import (
    ARTIFACT_VERSION,
    LoadedProvisionalModel,
    ProvisionalModelMetadata,
    TrainingProvenance,
)
from fraud_detection.models.model_input import XGBOOST_FEATURE_CONTRACT
from fraud_detection.serving.api import create_scored_app
from fraud_detection.serving.schemas import ScoredDecisionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_decision_policy.toml"
REASON_CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_reasons.toml"
SCORE_SOURCE_LABEL = "XGBoost engineering-only provisional score source"


def test_scored_decisions_endpoint_returns_stable_decision_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _PredictOnlyPipeline(0.93)
    client = _client(monkeypatch, pipeline)

    first = client.post("/scored-decisions", json=_payload())
    second = client.post("/scored-decisions", json=_payload())

    expected = {
        "transaction_id": 42,
        "risk_score": 0.93,
        "decision": "decline",
        "reason_codes": [
            HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
            HIGH_TRANSACTION_AMOUNT,
            UNUSUAL_TRANSACTION_VELOCITY,
        ],
        "policy_version": "validation-provisional-v1",
        "score_source_label": SCORE_SOURCE_LABEL,
        "artifact_version": ARTIFACT_VERSION,
        "artifact_sha256": "1" * 64,
        "xgboost_config_sha256": "2" * 64,
    }
    assert first.status_code == 200
    assert first.json() == expected
    assert second.json() == expected
    assert pipeline.prediction_calls == 2


def test_scored_healthcheck_reports_its_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _client(monkeypatch, _PredictOnlyPipeline(0.10)).get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "scored"}


def test_scored_app_preserves_supplied_score_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch, _PredictOnlyPipeline(0.10))

    response = client.post(
        "/decisions",
        json={
            "transaction_id": 42,
            "risk_score": 0.10,
            "transaction_amount": "100.00",
            "customer_tx_count_short_window": 0,
            "customer_amount_mean_prior": "100.00",
            "customer_amount_deviation_from_mean_prior": "0.00",
            "customer_seconds_since_previous": 60,
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "approve"


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "risk_score",
        "tx_fraud",
        "tx_fraud_scenario",
        "TX_FRAUD",
        "TX_FRAUD_SCENARIO",
    ),
)
def test_scored_request_forbids_supplied_scores_and_fraud_labels(
    monkeypatch: pytest.MonkeyPatch,
    forbidden_field: str,
) -> None:
    assert forbidden_field not in ScoredDecisionRequest.model_fields
    payload = _payload()
    payload[forbidden_field] = 1

    response = _client(monkeypatch, _PredictOnlyPipeline(0.10)).post(
        "/scored-decisions",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("tx_amount", "0.00"),
        ("customer_tx_count_short_window", 8),
        ("customer_amount_mean_prior", None),
        ("tx_datetime", "2018-04-03T06:00:00"),
    ),
)
def test_scored_request_rejects_impossible_or_partial_values(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
) -> None:
    payload = _payload()
    payload[field] = invalid_value

    response = _client(monkeypatch, _PredictOnlyPipeline(0.10)).post(
        "/scored-decisions",
        json=payload,
    )

    assert response.status_code == 422


def test_scored_app_loads_runtime_objects_once_and_never_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = {"policy": 0, "reasons": 0, "artifact": 0}
    original_policy_loader = api_module.load_selected_decision_policy_config
    original_reason_loader = api_module.load_decision_reason_config
    pipeline = _PredictOnlyPipeline(0.10)

    def load_policy(path: Path):
        loads["policy"] += 1
        return original_policy_loader(path)

    def load_reasons(path: Path):
        loads["reasons"] += 1
        return original_reason_loader(path)

    def load_artifact(*args: object, **kwargs: object) -> LoadedProvisionalModel:
        loads["artifact"] += 1
        return _loaded_model(pipeline)

    monkeypatch.setattr(api_module, "load_selected_decision_policy_config", load_policy)
    monkeypatch.setattr(api_module, "load_decision_reason_config", load_reasons)
    monkeypatch.setattr(api_module, "load_provisional_model_artifact", load_artifact)
    client = TestClient(
        create_scored_app(
            policy_config_path=POLICY_CONFIG_PATH,
            reason_config_path=REASON_CONFIG_PATH,
        )
    )

    client.post("/scored-decisions", json=_payload())
    client.post("/scored-decisions", json=_payload())

    assert loads == {"policy": 1, "reasons": 1, "artifact": 1}
    assert pipeline.prediction_calls == 2
    assert pipeline.fit_calls == 0


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
        self.fit_calls = 0

    def predict_proba(self, values: object) -> _Probabilities:
        self.prediction_calls += 1
        return _Probabilities(self.risk_score)

    def fit(self, *args: object, **kwargs: object) -> None:
        self.fit_calls += 1
        pytest.fail(f"pipeline.fit was called: {args}, {kwargs}")


def _client(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _PredictOnlyPipeline,
) -> TestClient:
    monkeypatch.setattr(
        api_module,
        "load_provisional_model_artifact",
        lambda *args, **kwargs: _loaded_model(pipeline),
    )
    return TestClient(
        create_scored_app(
            policy_config_path=POLICY_CONFIG_PATH,
            reason_config_path=REASON_CONFIG_PATH,
        )
    )


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


def _payload() -> dict[str, object]:
    return {
        "transaction_id": 42,
        "tx_amount": "300.00",
        "tx_time_days": 2,
        "tx_datetime": "2018-04-03T06:00:00Z",
        "customer_tx_count_short_window": 3,
        "customer_tx_count_long_window": 7,
        "customer_amount_mean_prior": "100.00",
        "customer_amount_deviation_from_mean_prior": "200.00",
        "customer_seconds_since_previous": 60,
        "customer_id": 12,
        "terminal_id": 8,
    }
