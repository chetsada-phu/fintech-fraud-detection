"""Tests for the bounded provisional XGBoost artifact lifecycle."""

import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import fraud_detection.models.artifact as artifact_module
from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.decisioning.inference import (
    load_selected_decision_policy_config,
)
from fraud_detection.features.temporal import (
    FEATURE_FILENAMES,
    TEMPORAL_FEATURE_COLUMNS,
    TemporalFeatureConfig,
    build_temporal_features,
)
from fraud_detection.models.artifact import (
    build_provisional_model_artifact,
    load_provisional_model_artifact,
    load_training_rows_with_provenance,
    persist_provisional_model_artifact,
)
from fraud_detection.models.artifact_cli import main
from fraud_detection.models.model_input import XGBOOST_FEATURE_CONTRACT
from fraud_detection.models.xgboost_model import (
    fit_xgboost_baseline,
    load_xgboost_config,
    predict_xgboost_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XGBOOST_CONFIG_PATH = PROJECT_ROOT / "configs" / "xgboost_baseline.toml"
POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_decision_policy.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)
TEMPORAL_CONFIG = TemporalFeatureConfig(3_600, 86_400, 4)


def test_round_trip_preserves_scores_and_validates_all_current_sources(
    tmp_path: Path,
) -> None:
    """A checked artifact should score exactly like its in-memory pipeline."""
    input_directory, feature_directory = _write_training_sources(tmp_path)
    rows, provenance = load_training_rows_with_provenance(
        input_directory,
        feature_directory,
    )
    pipeline = fit_xgboost_baseline(rows, load_xgboost_config(XGBOOST_CONFIG_PATH))
    expected_scores = predict_xgboost_scores(pipeline, rows)
    artifact_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.metadata.json"
    score_source_label = _score_source_label()

    persist_provisional_model_artifact(
        pipeline,
        artifact_path,
        metadata_path,
        xgboost_config_sha256=_sha256(XGBOOST_CONFIG_PATH),
        score_source_label=score_source_label,
        training_provenance=provenance,
    )
    loaded = load_provisional_model_artifact(
        artifact_path,
        metadata_path,
        XGBOOST_CONFIG_PATH,
        score_source_label,
        input_directory,
        feature_directory,
    )

    assert predict_xgboost_scores(loaded.pipeline, rows) == pytest.approx(
        expected_scores,
        abs=0.0,
    )
    assert loaded.metadata.feature_contract == XGBOOST_FEATURE_CONTRACT
    assert loaded.metadata.training_provenance == provenance


def test_metadata_is_byte_stable_for_the_same_pipeline(tmp_path: Path) -> None:
    """Metadata must omit timestamps, random IDs, and host-specific paths."""
    input_directory, feature_directory = _write_training_sources(tmp_path)
    rows, provenance = load_training_rows_with_provenance(
        input_directory,
        feature_directory,
    )
    pipeline = fit_xgboost_baseline(rows, load_xgboost_config(XGBOOST_CONFIG_PATH))
    first_artifact = tmp_path / "first.joblib"
    first_metadata = tmp_path / "first.json"
    second_artifact = tmp_path / "second.joblib"
    second_metadata = tmp_path / "second.json"
    arguments = {
        "xgboost_config_sha256": _sha256(XGBOOST_CONFIG_PATH),
        "score_source_label": _score_source_label(),
        "training_provenance": provenance,
    }

    persist_provisional_model_artifact(
        pipeline,
        first_artifact,
        first_metadata,
        **arguments,
    )
    persist_provisional_model_artifact(
        pipeline,
        second_artifact,
        second_metadata,
        **arguments,
    )

    assert first_artifact.read_bytes() == second_artifact.read_bytes()
    assert first_metadata.read_bytes() == second_metadata.read_bytes()


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (
        ("artifact_version", "artifact_version"),
        ("artifact_sha256", "checksum"),
        ("xgboost_config_sha256", "configuration hash"),
        ("score_source_label", "score-source label"),
        ("feature_contract", "feature_contract"),
        ("training_provenance", "training provenance"),
    ),
)
def test_metadata_mismatch_is_rejected_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    """Every frozen metadata boundary must fail before joblib can execute."""
    input_directory, feature_directory = _write_training_sources(tmp_path)
    artifact_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.json"
    score_source_label = _score_source_label()
    build_provisional_model_artifact(
        input_directory,
        feature_directory,
        XGBOOST_CONFIG_PATH,
        score_source_label,
        artifact_path,
        metadata_path,
    )
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mismatch == "artifact_version":
        document[mismatch] = "unexpected-v0"
    elif mismatch == "score_source_label":
        document[mismatch] = "different score source"
    elif mismatch == "feature_contract":
        document[mismatch] = document[mismatch][:-1]
    elif mismatch == "training_provenance":
        document[mismatch]["transactions_sha256"] = "0" * 64
    else:
        document[mismatch] = "0" * 64
    metadata_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def unexpected_load(path: Path) -> object:
        pytest.fail(f"joblib.load ran for mismatched metadata: {path}")

    monkeypatch.setattr(artifact_module.joblib, "load", unexpected_load)

    with pytest.raises(ValueError, match=message):
        load_provisional_model_artifact(
            artifact_path,
            metadata_path,
            XGBOOST_CONFIG_PATH,
            score_source_label,
            input_directory,
            feature_directory,
        )


def test_cli_builds_with_only_training_files(tmp_path: Path) -> None:
    """The offline command must not require validation or test data."""
    input_directory, feature_directory = _write_training_sources(tmp_path)
    artifact_path = tmp_path / "models" / "model.joblib"
    metadata_path = tmp_path / "models" / "model.metadata.json"

    exit_code = main(
        [
            "--input-directory",
            str(input_directory),
            "--feature-directory",
            str(feature_directory),
            "--xgboost-config",
            str(XGBOOST_CONFIG_PATH),
            "--policy-config",
            str(POLICY_CONFIG_PATH),
            "--artifact",
            str(artifact_path),
            "--metadata",
            str(metadata_path),
        ]
    )

    assert exit_code == 0
    assert artifact_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["training_provenance"]["split"] == "train"
    assert metadata["training_provenance"]["transaction_rows"] == 18
    assert not (input_directory / "validation.csv").exists()
    assert not (input_directory / "test.csv").exists()


def _write_training_sources(tmp_path: Path) -> tuple[Path, Path]:
    input_directory = tmp_path / "processed"
    feature_directory = tmp_path / "features"
    input_directory.mkdir()
    feature_directory.mkdir()
    transactions = _transactions()
    write_transactions_csv(transactions, input_directory / "train.csv")
    features = build_temporal_features(transactions, TEMPORAL_CONFIG)
    feature_path = feature_directory / FEATURE_FILENAMES["train"]
    with feature_path.open("w", encoding="utf-8", newline="") as feature_file:
        writer = csv.DictWriter(feature_file, fieldnames=TEMPORAL_FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(feature.to_csv_row() for feature in features)
    return input_directory, feature_directory


def _transactions() -> tuple[Transaction, ...]:
    fraud_ids = {3, 9, 15}
    return tuple(
        Transaction(
            transaction_id=index,
            tx_datetime=START + timedelta(hours=index * 2),
            customer_id=index % 5,
            terminal_id=index % 4,
            tx_amount=(
                Decimal("260.00") if index in fraud_ids else Decimal("30.00") + index
            ),
            tx_time_seconds=index * 2 * 3_600,
            tx_time_days=(index * 2 * 3_600) // SECONDS_PER_DAY,
            tx_fraud=int(index in fraud_ids),
            tx_fraud_scenario=int(index in fraud_ids),
        )
        for index in range(18)
    )


def _score_source_label() -> str:
    return load_selected_decision_policy_config(POLICY_CONFIG_PATH).score_source_label


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
