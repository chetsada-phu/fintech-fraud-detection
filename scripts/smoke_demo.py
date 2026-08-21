"""Smoke-test the checked local demo without starting a network server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraud_detection.serving.api import create_scored_app
from fraud_detection.serving.latency import load_scored_api_latency_config
from fraud_detection.serving.latency_cli import DEFAULT_BENCHMARK_CONFIG_PATH


def main() -> int:
    """Check the health page, analyst page, and one real scored decision."""
    config = load_scored_api_latency_config(DEFAULT_BENCHMARK_CONFIG_PATH)
    application = create_scored_app()
    with TestClient(application) as client:
        health = client.get("/healthz")
        analyst_page = client.get("/analyst-demo")
        scored = client.post(config.endpoint, json=config.request_payload)
    if health.status_code != 200 or health.json() != {
        "status": "ok",
        "mode": "scored",
    }:
        raise RuntimeError("scored application health check failed")
    if analyst_page.status_code != 200 or "Score one payment" not in analyst_page.text:
        raise RuntimeError("analyst demo page check failed")
    if scored.status_code != 200:
        raise RuntimeError(f"scored request failed with HTTP {scored.status_code}")
    result = scored.json()
    print(
        "demo_smoke_ok "
        f"transaction_id={result['transaction_id']} "
        f"risk_score={result['risk_score']} "
        f"decision={result['decision']} "
        f"policy={result['policy_version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
