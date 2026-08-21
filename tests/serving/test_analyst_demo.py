"""Integration tests for the scored-app-only analyst demo."""

from html.parser import HTMLParser
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

import fraud_detection.serving.api as api_module
from fraud_detection.models.artifact import (
    ARTIFACT_VERSION,
    LoadedProvisionalModel,
    ProvisionalModelMetadata,
    TrainingProvenance,
)
from fraud_detection.models.model_input import XGBOOST_FEATURE_CONTRACT
from fraud_detection.serving.api import create_app, create_scored_app
from fraud_detection.serving.schemas import ScoredDecisionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_decision_policy.toml"
REASON_CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_reasons.toml"
SCORE_SOURCE_LABEL = "XGBoost engineering-only provisional score source"


def test_analyst_demo_is_available_only_on_scored_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_only = TestClient(create_app(POLICY_CONFIG_PATH, REASON_CONFIG_PATH))
    scored = _scored_client(monkeypatch)

    assert decision_only.get("/analyst-demo").status_code == 404
    response = scored.get("/analyst-demo")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_analyst_demo_uses_the_existing_scored_request_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _scored_client(monkeypatch).get("/analyst-demo")
    parser = _NamedInputParser()
    parser.feed(response.text)

    assert parser.names == set(ScoredDecisionRequest.model_fields)
    assert 'fetch("/scored-decisions"' in response.text
    assert 'method: "POST"' in response.text
    assert 'headers: { "Content-Type": "application/json" }' in response.text


def test_analyst_demo_inserts_api_values_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _scored_client(monkeypatch).get("/analyst-demo").text

    assert ".textContent" in page
    assert "document.createElement" in page
    assert ".innerHTML" not in page
    assert "insertAdjacentHTML" not in page
    assert "document.write" not in page
    assert "localStorage" not in page
    assert "sessionStorage" not in page


def test_analyst_demo_includes_responsive_and_accessible_basics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _scored_client(monkeypatch).get("/analyst-demo").text

    assert 'name="viewport"' in page
    assert 'aria-live="polite"' in page
    assert 'role="alert"' in page
    assert "@media (max-width: 900px)" in page
    assert "@media (prefers-reduced-motion: reduce)" in page


class _NamedInputParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.names: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "input":
            return
        attributes = dict(attrs)
        name = attributes.get("name")
        if name is not None:
            self.names.add(name)


class _Probabilities:
    shape = (1, 2)

    def __getitem__(self, key: tuple[int, int]) -> float:
        if key == (0, 1):
            return 0.93
        raise IndexError(key)


class _PredictOnlyPipeline:
    def predict_proba(self, values: object) -> _Probabilities:
        return _Probabilities()


def _scored_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        api_module,
        "load_provisional_model_artifact",
        lambda *args, **kwargs: LoadedProvisionalModel(
            pipeline=cast(Pipeline, _PredictOnlyPipeline()),
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
        ),
    )
    return TestClient(
        create_scored_app(
            policy_config_path=POLICY_CONFIG_PATH,
            reason_config_path=REASON_CONFIG_PATH,
        )
    )
