"""Reproducible local latency measurement for the scored-decision API."""

from __future__ import annotations

import math
import os
import platform
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraud_detection.serving.schemas import ScoredDecisionRequest

EXPECTED_ENDPOINT = "/scored-decisions"
PERCENTILE_METHOD = "nearest rank"
TRANSPORT_DESCRIPTION = "FastAPI TestClient using in-process ASGI"
TIMER_DESCRIPTION = "time.perf_counter_ns"
_CONFIG_KEYS = {
    "benchmark_version",
    "endpoint",
    "measured_requests",
    "request",
    "warmup_requests",
}


@dataclass(frozen=True, slots=True)
class ScoredApiLatencyConfig:
    """Versioned request fixture and fixed local benchmark workload."""

    benchmark_version: str
    endpoint: str
    warmup_requests: int
    measured_requests: int
    request: ScoredDecisionRequest

    def __post_init__(self) -> None:
        if not self.benchmark_version.strip():
            raise ValueError("benchmark_version must not be empty")
        if self.endpoint != EXPECTED_ENDPOINT:
            raise ValueError(f"endpoint must be {EXPECTED_ENDPOINT!r}")
        if type(self.warmup_requests) is not int or self.warmup_requests < 1:
            raise ValueError("warmup_requests must be a positive integer")
        if type(self.measured_requests) is not int or self.measured_requests < 20:
            raise ValueError("measured_requests must be an integer of at least 20")
        if not isinstance(self.request, ScoredDecisionRequest):
            raise ValueError("request must be a ScoredDecisionRequest")

    @property
    def request_payload(self) -> dict[str, object]:
        """Return the validated JSON-compatible benchmark request."""
        return self.request.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class RequestLatencySummary:
    """Nearest-rank latency statistics in milliseconds."""

    samples: int
    minimum_ms: float
    p50_ms: float
    p95_ms: float
    maximum_ms: float

    def __post_init__(self) -> None:
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("latency samples must be a positive integer")
        values = (self.minimum_ms, self.p50_ms, self.p95_ms, self.maximum_ms)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("latency statistics must be finite and non-negative")
        if tuple(sorted(values)) != values:
            raise ValueError("latency statistics must be monotonically ordered")


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironment:
    """Non-sensitive local runtime boundary for interpreting measurements."""

    python_version: str
    operating_system: str
    machine: str


@dataclass(frozen=True, slots=True)
class ScoredApiLatencyReport:
    """One startup measurement and warmed request-latency distribution."""

    config: ScoredApiLatencyConfig
    startup_validation_ms: float
    request_latency: RequestLatencySummary
    environment: BenchmarkEnvironment
    policy_version: str
    score_source_label: str
    artifact_version: str
    artifact_sha256: str
    xgboost_config_sha256: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.startup_validation_ms):
            raise ValueError("startup_validation_ms must be finite")
        if self.startup_validation_ms < 0:
            raise ValueError("startup_validation_ms must be non-negative")
        if self.request_latency.samples != self.config.measured_requests:
            raise ValueError("request latency samples must match measured_requests")


def load_scored_api_latency_config(path: Path) -> ScoredApiLatencyConfig:
    """Load and strictly validate the versioned local benchmark contract."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    if set(document) != {"scored_api_latency"}:
        raise ValueError("configuration must contain only [scored_api_latency]")
    section = document["scored_api_latency"]
    if not isinstance(section, Mapping):
        raise ValueError("[scored_api_latency] must be a table")
    if set(section) != _CONFIG_KEYS:
        raise ValueError("scored_api_latency fields do not match the contract")
    request = section["request"]
    if not isinstance(request, Mapping):
        raise ValueError("scored_api_latency.request must be a table")
    try:
        return ScoredApiLatencyConfig(
            benchmark_version=_require_string(section, "benchmark_version"),
            endpoint=_require_string(section, "endpoint"),
            warmup_requests=_require_integer(section, "warmup_requests"),
            measured_requests=_require_integer(section, "measured_requests"),
            request=ScoredDecisionRequest.model_validate(request),
        )
    except ValidationError as error:
        raise ValueError(f"invalid scored API request fixture: {error}") from error


def summarize_request_latencies(
    durations_ns: Sequence[int],
) -> RequestLatencySummary:
    """Calculate nearest-rank p50/p95 and range from nanosecond samples."""
    if not durations_ns:
        raise ValueError("at least one request latency is required")
    if any(type(duration) is not int or duration < 0 for duration in durations_ns):
        raise ValueError("request latencies must be non-negative integer nanoseconds")
    ordered = tuple(sorted(durations_ns))
    return RequestLatencySummary(
        samples=len(ordered),
        minimum_ms=_nanoseconds_to_milliseconds(ordered[0]),
        p50_ms=_nanoseconds_to_milliseconds(_nearest_rank(ordered, 0.50)),
        p95_ms=_nanoseconds_to_milliseconds(_nearest_rank(ordered, 0.95)),
        maximum_ms=_nanoseconds_to_milliseconds(ordered[-1]),
    )


def benchmark_scored_api(
    config: ScoredApiLatencyConfig,
    create_application: Callable[[], FastAPI],
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    environment: BenchmarkEnvironment | None = None,
) -> ScoredApiLatencyReport:
    """Create one app, warm it, and time only successful request calls."""
    startup_start_ns = clock_ns()
    application = create_application()
    startup_validation_ns = clock_ns() - startup_start_ns
    if startup_validation_ns < 0:
        raise ValueError("benchmark clock moved backwards during startup")

    loaded_model = application.state.loaded_model
    if loaded_model is None:
        raise ValueError("scored API benchmark requires an artifact-backed app")
    policy_config = application.state.policy_config
    metadata = loaded_model.metadata
    expected_identity = {
        "artifact_sha256": metadata.artifact_sha256,
        "artifact_version": metadata.artifact_version,
        "policy_version": policy_config.policy_version,
        "score_source_label": metadata.score_source_label,
        "xgboost_config_sha256": metadata.xgboost_config_sha256,
    }
    request_payload = config.request_payload
    durations_ns: list[int] = []
    with TestClient(application) as client:
        for _ in range(config.warmup_requests):
            response = client.post(config.endpoint, json=request_payload)
            _validate_successful_response(
                response.status_code, response.json(), expected_identity
            )
        for _ in range(config.measured_requests):
            request_start_ns = clock_ns()
            response = client.post(config.endpoint, json=request_payload)
            duration_ns = clock_ns() - request_start_ns
            if duration_ns < 0:
                raise ValueError("benchmark clock moved backwards during a request")
            durations_ns.append(duration_ns)
            _validate_successful_response(
                response.status_code, response.json(), expected_identity
            )

    return ScoredApiLatencyReport(
        config=config,
        startup_validation_ms=_nanoseconds_to_milliseconds(startup_validation_ns),
        request_latency=summarize_request_latencies(durations_ns),
        environment=environment or current_benchmark_environment(),
        policy_version=policy_config.policy_version,
        score_source_label=metadata.score_source_label,
        artifact_version=metadata.artifact_version,
        artifact_sha256=metadata.artifact_sha256,
        xgboost_config_sha256=metadata.xgboost_config_sha256,
    )


def current_benchmark_environment() -> BenchmarkEnvironment:
    """Describe the local runtime without recording a hostname or user path."""
    return BenchmarkEnvironment(
        python_version=platform.python_version(),
        operating_system=f"{platform.system()} {platform.release()}",
        machine=platform.machine() or "unknown",
    )


def render_markdown(report: ScoredApiLatencyReport) -> str:
    """Render a stable report structure around machine-specific measurements."""
    config = report.config
    latency = report.request_latency
    environment = report.environment
    lines = [
        "# Local Scored-decision API Latency Report",
        "",
        "Generated by `fraud-benchmark-scored-api` from the fixed local",
        "benchmark contract. Numeric timings are measurements of this run, not",
        "deterministic outputs or production service-level objectives.",
        "",
        "## Benchmark Contract",
        "",
        f"- Benchmark version: `{config.benchmark_version}`.",
        f"- Endpoint: `POST {config.endpoint}`.",
        f"- Transport: {TRANSPORT_DESCRIPTION}.",
        f"- Timer: `{TIMER_DESCRIPTION}`.",
        f"- Percentile method: {PERCENTILE_METHOD}.",
        f"- Warm-up requests: {config.warmup_requests:,}.",
        f"- Measured requests: {config.measured_requests:,}.",
        "- Startup boundary: one `create_scored_app` factory call, including",
        "  configuration, artifact-integrity, and training-provenance validation.",
        "- Request boundary: warmed in-process `client.post(...)` wall time;",
        "  response JSON validation happens after each timed interval.",
        "",
        "## Runtime Identity",
        "",
        f"- Python: `{environment.python_version}`.",
        f"- Operating system: `{environment.operating_system}`.",
        f"- Machine architecture: `{environment.machine}`.",
        f"- Policy version: `{report.policy_version}`.",
        f"- Score source: `{report.score_source_label}`.",
        f"- Artifact version: `{report.artifact_version}`.",
        f"- Artifact SHA-256: `{report.artifact_sha256}`.",
        f"- XGBoost config SHA-256: `{report.xgboost_config_sha256}`.",
        "",
        "## Measurements",
        "",
        "| Boundary | Samples | Milliseconds |",
        "| --- | ---: | ---: |",
        f"| Application factory startup validation | 1 | "
        f"{report.startup_validation_ms:.3f} |",
        f"| Warmed request minimum | {latency.samples:,} | {latency.minimum_ms:.3f} |",
        f"| Warmed request p50 | {latency.samples:,} | {latency.p50_ms:.3f} |",
        f"| Warmed request p95 | {latency.samples:,} | {latency.p95_ms:.3f} |",
        f"| Warmed request maximum | {latency.samples:,} | {latency.maximum_ms:.3f} |",
        "",
        "## Interpretation Limits",
        "",
        "- This is a single-process local development measurement, not a",
        "  production latency claim, capacity test, or service-level objective.",
        "- In-process TestClient timing excludes TCP, Uvicorn, reverse proxies,",
        "  network delay, process launch, Python module imports, and concurrency.",
        "- The fixed request uses precomputed past-only features. The benchmark",
        "  does not measure live temporal-state calculation or persistence.",
        "- Timings vary with hardware, operating-system scheduling, Python and",
        "  dependency builds, background load, and power-management state.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report(report: ScoredApiLatencyReport, output_path: Path) -> None:
    """Atomically write one machine-specific report in a stable format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _nearest_rank(ordered_values: Sequence[int], percentile: float) -> int:
    rank = math.ceil(percentile * len(ordered_values))
    return ordered_values[rank - 1]


def _nanoseconds_to_milliseconds(value: int) -> float:
    return value / 1_000_000


def _validate_successful_response(
    status_code: int,
    document: object,
    expected_identity: Mapping[str, str],
) -> None:
    if status_code != 200:
        raise ValueError(f"benchmark request returned HTTP {status_code}")
    if not isinstance(document, Mapping):
        raise ValueError("benchmark response must be a JSON object")
    for key, expected_value in expected_identity.items():
        if document.get(key) != expected_value:
            raise ValueError(f"benchmark response {key} does not match runtime")


def _require_string(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_integer(section: Mapping[str, object], key: str) -> int:
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value
