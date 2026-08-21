"""Integration tests for the label-free decision-only FastAPI endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fraud_detection.serving.api as api_module
from fraud_detection.decisioning.reasons import (
    HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
    HIGH_TRANSACTION_AMOUNT,
    UNUSUAL_TRANSACTION_VELOCITY,
)
from fraud_detection.serving.api import create_app
from fraud_detection.serving.schemas import DecisionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_decision_policy.toml"
REASON_CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_reasons.toml"


def test_decisions_endpoint_exposes_exact_empty_review_band() -> None:
    client = _client()

    approve_response = client.post(
        "/decisions",
        json=_payload(risk_score=0.9299, transaction_amount="100.00"),
    )
    decline_response = client.post(
        "/decisions",
        json=_payload(risk_score=0.93, transaction_amount="100.00"),
    )

    assert approve_response.status_code == 200
    assert approve_response.json() == {
        "transaction_id": 42,
        "risk_score": 0.9299,
        "decision": "approve",
        "reason_codes": [],
        "policy_version": "validation-provisional-v1",
    }
    assert decline_response.status_code == 200
    assert decline_response.json()["decision"] == "decline"
    assert "review" not in {
        approve_response.json()["decision"],
        decline_response.json()["decision"],
    }


def test_decision_only_healthcheck_reports_its_mode() -> None:
    response = _client().get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "decision-only"}


def test_decisions_endpoint_preserves_configured_reason_order() -> None:
    response = _client().post(
        "/decisions",
        json=_payload(
            risk_score=0.93,
            transaction_amount="300.00",
            customer_tx_count_short_window=3,
        ),
    )

    assert response.status_code == 200
    assert response.json()["reason_codes"] == [
        HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
        HIGH_TRANSACTION_AMOUNT,
        UNUSUAL_TRANSACTION_VELOCITY,
    ]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("risk_score", 1.01),
        ("transaction_amount", "0.00"),
        ("customer_tx_count_short_window", -1),
        ("customer_seconds_since_previous", -1),
    ),
)
def test_decisions_endpoint_rejects_impossible_values(
    field: str, invalid_value: object
) -> None:
    payload = _payload(risk_score=0.50, transaction_amount="100.00")
    payload[field] = invalid_value

    response = _client().post("/decisions", json=payload)

    assert response.status_code == 422


def test_decisions_endpoint_rejects_partial_history() -> None:
    payload = _payload(risk_score=0.50, transaction_amount="100.00")
    payload["customer_amount_mean_prior"] = None

    response = _client().post("/decisions", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "label_field",
    ("tx_fraud", "tx_fraud_scenario", "TX_FRAUD", "TX_FRAUD_SCENARIO"),
)
def test_fraud_labels_are_absent_and_forbidden(label_field: str) -> None:
    assert label_field not in DecisionRequest.model_fields
    payload = _payload(risk_score=0.50, transaction_amount="100.00")
    payload[label_field] = 1

    response = _client().post("/decisions", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_app_loads_each_fixed_configuration_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_loads = 0
    reason_loads = 0
    original_policy_loader = api_module.load_selected_decision_policy_config
    original_reason_loader = api_module.load_decision_reason_config

    def load_policy(path: Path):
        nonlocal policy_loads
        policy_loads += 1
        return original_policy_loader(path)

    def load_reasons(path: Path):
        nonlocal reason_loads
        reason_loads += 1
        return original_reason_loader(path)

    monkeypatch.setattr(
        api_module,
        "load_selected_decision_policy_config",
        load_policy,
    )
    monkeypatch.setattr(api_module, "load_decision_reason_config", load_reasons)
    client = TestClient(create_app(POLICY_CONFIG_PATH, REASON_CONFIG_PATH))

    client.post(
        "/decisions",
        json=_payload(risk_score=0.10, transaction_amount="100.00"),
    )
    client.post(
        "/decisions",
        json=_payload(risk_score=0.93, transaction_amount="300.00"),
    )

    assert policy_loads == 1
    assert reason_loads == 1


def _client() -> TestClient:
    return TestClient(create_app(POLICY_CONFIG_PATH, REASON_CONFIG_PATH))


def _payload(
    *,
    risk_score: float,
    transaction_amount: str,
    customer_tx_count_short_window: int = 0,
) -> dict[str, object]:
    return {
        "transaction_id": 42,
        "risk_score": risk_score,
        "transaction_amount": transaction_amount,
        "customer_tx_count_short_window": customer_tx_count_short_window,
        "customer_amount_mean_prior": "100.00",
        "customer_amount_deviation_from_mean_prior": "0.00",
        "customer_seconds_since_previous": 60,
    }
