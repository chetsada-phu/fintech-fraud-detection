"""Tests for the reproducible local scored-API latency benchmark."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from fraud_detection.serving.latency import (
    BenchmarkEnvironment,
    RequestLatencySummary,
    ScoredApiLatencyConfig,
    ScoredApiLatencyReport,
    benchmark_scored_api,
    load_scored_api_latency_config,
    render_markdown,
    summarize_request_latencies,
    write_markdown_report,
)
from fraud_detection.serving.schemas import ScoredDecisionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "serving_latency.toml"
SCORE_SOURCE_LABEL = "XGBoost engineering-only provisional score source"


def test_versioned_benchmark_config_has_fixed_label_free_workload() -> None:
    config = load_scored_api_latency_config(CONFIG_PATH)

    assert config.benchmark_version == "local-in-process-v1"
    assert config.endpoint == "/scored-decisions"
    assert config.warmup_requests == 10
    assert config.measured_requests == 200
    assert config.request_payload == _request_payload()
    assert {"risk_score", "tx_fraud", "tx_fraud_scenario"}.isdisjoint(
        config.request_payload
    )


def test_nearest_rank_percentiles_are_exact_and_order_independent() -> None:
    durations_ns = tuple(value * 1_000_000 for value in range(20, 0, -1))

    summary = summarize_request_latencies(durations_ns)

    assert summary == RequestLatencySummary(
        samples=20,
        minimum_ms=1.0,
        p50_ms=10.0,
        p95_ms=19.0,
        maximum_ms=20.0,
    )


@pytest.mark.parametrize("durations", ((), (1, -1), (1, 2.5)))
def test_latency_summary_rejects_missing_or_invalid_samples(
    durations: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="latenc|at least"):
        summarize_request_latencies(durations)  # type: ignore[arg-type]


def test_benchmark_creates_one_app_warms_then_measures_requests() -> None:
    application = _fixture_app()
    factory_calls = 0
    clock_values = iter(_clock_values())

    def create_application() -> FastAPI:
        nonlocal factory_calls
        factory_calls += 1
        return application

    report = benchmark_scored_api(
        _config(warmup_requests=2, measured_requests=20),
        create_application,
        clock_ns=lambda: next(clock_values),
        environment=_environment(),
    )

    assert factory_calls == 1
    assert application.state.request_calls == 22
    assert report.startup_validation_ms == 5.0
    assert report.request_latency == RequestLatencySummary(
        samples=20,
        minimum_ms=1.0,
        p50_ms=10.0,
        p95_ms=19.0,
        maximum_ms=20.0,
    )


def test_report_format_is_byte_stable_for_fixed_measurements(tmp_path: Path) -> None:
    report = ScoredApiLatencyReport(
        config=_config(warmup_requests=2, measured_requests=20),
        startup_validation_ms=5.1254,
        request_latency=RequestLatencySummary(
            samples=20,
            minimum_ms=1.0,
            p50_ms=10.0,
            p95_ms=19.0,
            maximum_ms=20.0,
        ),
        environment=_environment(),
        policy_version="fixture-policy-v1",
        score_source_label=SCORE_SOURCE_LABEL,
        artifact_version="fixture-artifact-v1",
        artifact_sha256="1" * 64,
        xgboost_config_sha256="2" * 64,
    )
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"

    write_markdown_report(report, first_path)
    write_markdown_report(report, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_text(encoding="utf-8") == render_markdown(report)
    assert "| Application factory startup validation | 1 | 5.125 |" in (
        render_markdown(report)
    )
    assert "| Warmed request p50 | 20 | 10.000 |" in render_markdown(report)
    assert "| Warmed request p95 | 20 | 19.000 |" in render_markdown(report)
    assert "production latency claim" in render_markdown(report)


def _config(
    *,
    warmup_requests: int,
    measured_requests: int,
) -> ScoredApiLatencyConfig:
    return ScoredApiLatencyConfig(
        benchmark_version="fixture-v1",
        endpoint="/scored-decisions",
        warmup_requests=warmup_requests,
        measured_requests=measured_requests,
        request=ScoredDecisionRequest.model_validate(_request_payload()),
    )


def _fixture_app() -> FastAPI:
    application = FastAPI()
    application.state.request_calls = 0
    metadata = SimpleNamespace(
        artifact_sha256="1" * 64,
        artifact_version="fixture-artifact-v1",
        score_source_label=SCORE_SOURCE_LABEL,
        xgboost_config_sha256="2" * 64,
    )
    application.state.loaded_model = SimpleNamespace(metadata=metadata)
    application.state.policy_config = SimpleNamespace(
        policy_version="fixture-policy-v1"
    )

    @application.post("/scored-decisions")
    def scored_decision() -> dict[str, object]:
        application.state.request_calls += 1
        return {
            "artifact_sha256": metadata.artifact_sha256,
            "artifact_version": metadata.artifact_version,
            "policy_version": "fixture-policy-v1",
            "score_source_label": metadata.score_source_label,
            "xgboost_config_sha256": metadata.xgboost_config_sha256,
        }

    return application


def _request_payload() -> dict[str, object]:
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


def _clock_values() -> tuple[int, ...]:
    values = [0, 5_000_000]
    current = 5_000_000
    for duration_ms in range(1, 21):
        values.append(current)
        current += duration_ms * 1_000_000
        values.append(current)
    return tuple(values)


def _environment() -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        python_version="3.13.7",
        operating_system="FixtureOS 1",
        machine="fixture64",
    )
