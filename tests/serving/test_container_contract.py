"""Static contract tests for the bounded local container image."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
RUNBOOK = PROJECT_ROOT / "docs" / "container_runbook.md"


def test_container_runs_the_scored_factory_as_a_non_root_user() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.13.7-slim-bookworm AS runtime" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "fraud_detection.serving.api:create_scored_app" in dockerfile
    assert '"--factory"' in dockerfile
    assert '"--host", "0.0.0.0"' in dockerfile
    assert '"--port", "8000"' in dockerfile


def test_container_healthcheck_uses_the_local_health_endpoint() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:8000/healthz" in dockerfile
    assert "urllib.request" in dockerfile


def test_build_context_excludes_data_models_and_local_state() -> None:
    patterns = tuple(
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    assert patterns[0] == "*"
    assert "!src/**" in patterns
    assert "!configs/**" in patterns
    assert not any(pattern.startswith("!data") for pattern in patterns)
    assert not any(pattern.startswith("!models") for pattern in patterns)
    assert not any(pattern.startswith("!.git") for pattern in patterns)
    assert not any(pattern.startswith("!.env") for pattern in patterns)


def test_container_does_not_copy_runtime_data_or_model_artifacts() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY data" not in dockerfile
    assert "COPY models" not in dockerfile
    assert "COPY . " not in dockerfile
    assert "COPY configs ./configs" in dockerfile


def test_runbook_mounts_only_the_four_required_runtime_files() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    required_mounts = (
        "models/xgboost_engineering_provisional.joblib",
        "models/xgboost_engineering_provisional.metadata.json",
        "data/processed/train.csv",
        "data/processed/features/train_temporal_features.csv",
    )

    assert "docker build --tag fintech-fraud-demo:local ." in runbook
    assert runbook.count("--mount type=bind") == 4
    assert runbook.count(",readonly") == 4
    for relative_path in required_mounts:
        assert f'src="$(pwd)/{relative_path}"' in runbook
