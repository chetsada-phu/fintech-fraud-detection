"""Trusted-local artifact lifecycle for the provisional XGBoost pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
from sklearn.pipeline import Pipeline

from fraud_detection.data.eda import load_training_transactions
from fraud_detection.data.splitter import SPLIT_FILENAMES
from fraud_detection.features.matrix import (
    JoinedFeatureRow,
    join_transaction_features,
    load_temporal_feature_csv,
)
from fraud_detection.features.temporal import FEATURE_FILENAMES
from fraud_detection.models.model_input import XGBOOST_FEATURE_CONTRACT
from fraud_detection.models.xgboost_model import (
    fit_xgboost_baseline,
    load_xgboost_config,
)

ARTIFACT_VERSION: Final = "xgboost-engineering-provisional-v1"
TRAINING_SPLIT: Final = "train"
_METADATA_KEYS: Final = {
    "artifact_sha256",
    "artifact_version",
    "feature_contract",
    "score_source_label",
    "training_provenance",
    "xgboost_config_sha256",
}
_PROVENANCE_KEYS: Final = {
    "fraud_rows",
    "split",
    "temporal_features_sha256",
    "transaction_rows",
    "transactions_sha256",
}


@dataclass(frozen=True, slots=True)
class TrainingProvenance:
    """Deterministic identity and counts for the exact training inputs."""

    split: str
    transaction_rows: int
    fraud_rows: int
    transactions_sha256: str
    temporal_features_sha256: str

    def __post_init__(self) -> None:
        if self.split != TRAINING_SPLIT:
            raise ValueError("training provenance split must be 'train'")
        if type(self.transaction_rows) is not int or self.transaction_rows <= 0:
            raise ValueError("training transaction_rows must be a positive integer")
        if (
            type(self.fraud_rows) is not int
            or self.fraud_rows < 0
            or self.fraud_rows > self.transaction_rows
        ):
            raise ValueError("training fraud_rows must be within the row count")
        _validate_sha256(self.transactions_sha256, "training transactions")
        _validate_sha256(
            self.temporal_features_sha256,
            "training temporal features",
        )

    def to_document(self) -> dict[str, object]:
        """Return the stable JSON-compatible representation."""
        return {
            "fraud_rows": self.fraud_rows,
            "split": self.split,
            "temporal_features_sha256": self.temporal_features_sha256,
            "transaction_rows": self.transaction_rows,
            "transactions_sha256": self.transactions_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProvisionalModelMetadata:
    """Validated metadata required before trusted-local deserialization."""

    artifact_version: str
    artifact_sha256: str
    xgboost_config_sha256: str
    score_source_label: str
    feature_contract: tuple[str, ...]
    training_provenance: TrainingProvenance

    def __post_init__(self) -> None:
        if self.artifact_version != ARTIFACT_VERSION:
            raise ValueError(f"artifact_version must be {ARTIFACT_VERSION!r}")
        _validate_sha256(self.artifact_sha256, "model artifact")
        _validate_sha256(self.xgboost_config_sha256, "XGBoost configuration")
        if not self.score_source_label.strip():
            raise ValueError("score_source_label must not be empty")
        if self.feature_contract != XGBOOST_FEATURE_CONTRACT:
            raise ValueError(
                "feature_contract does not match the frozen XGBoost inputs"
            )

    def to_document(self) -> dict[str, object]:
        """Return the stable JSON-compatible representation."""
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_version": self.artifact_version,
            "feature_contract": list(self.feature_contract),
            "score_source_label": self.score_source_label,
            "training_provenance": self.training_provenance.to_document(),
            "xgboost_config_sha256": self.xgboost_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class LoadedProvisionalModel:
    """One validated pipeline and its checked metadata."""

    pipeline: Pipeline
    metadata: ProvisionalModelMetadata


def build_provisional_model_artifact(
    input_directory: Path,
    feature_directory: Path,
    xgboost_config_path: Path,
    score_source_label: str,
    artifact_path: Path,
    metadata_path: Path,
) -> ProvisionalModelMetadata:
    """Fit from training only and atomically persist the provisional pipeline."""
    rows, provenance = load_training_rows_with_provenance(
        input_directory,
        feature_directory,
    )
    config_bytes = xgboost_config_path.read_bytes()
    config = load_xgboost_config(xgboost_config_path)
    if xgboost_config_path.read_bytes() != config_bytes:
        raise ValueError("XGBoost configuration changed during artifact build")
    pipeline = fit_xgboost_baseline(rows, config)
    return persist_provisional_model_artifact(
        pipeline,
        artifact_path,
        metadata_path,
        xgboost_config_sha256=_sha256_bytes(config_bytes),
        score_source_label=score_source_label,
        training_provenance=provenance,
    )


def persist_provisional_model_artifact(
    pipeline: Pipeline,
    artifact_path: Path,
    metadata_path: Path,
    *,
    xgboost_config_sha256: str,
    score_source_label: str,
    training_provenance: TrainingProvenance,
) -> ProvisionalModelMetadata:
    """Atomically replace an artifact and deterministic metadata sidecar."""
    _validate_pipeline(pipeline)
    if artifact_path.resolve() == metadata_path.resolve():
        raise ValueError("artifact and metadata paths must be different")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = _temporary_sibling(artifact_path)
    temporary_metadata = _temporary_sibling(metadata_path)
    try:
        joblib.dump(pipeline, temporary_artifact, compress=3, protocol=5)
        metadata = ProvisionalModelMetadata(
            artifact_version=ARTIFACT_VERSION,
            artifact_sha256=_sha256_file(temporary_artifact),
            xgboost_config_sha256=xgboost_config_sha256,
            score_source_label=score_source_label,
            feature_contract=XGBOOST_FEATURE_CONTRACT,
            training_provenance=training_provenance,
        )
        temporary_metadata.write_text(
            render_provisional_model_metadata(metadata),
            encoding="utf-8",
        )
        os.replace(temporary_artifact, artifact_path)
        os.replace(temporary_metadata, metadata_path)
        return metadata
    finally:
        temporary_artifact.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def load_provisional_model_artifact(
    artifact_path: Path,
    metadata_path: Path,
    xgboost_config_path: Path,
    expected_score_source_label: str,
    input_directory: Path,
    feature_directory: Path,
) -> LoadedProvisionalModel:
    """Validate current sources and then deserialize a trusted local artifact.

    Joblib artifacts can execute code while loading. The checksum here detects
    accidental change; it is not an authenticity mechanism for untrusted files.
    """
    metadata = load_provisional_model_metadata(metadata_path)
    expected_config_sha256 = _sha256_file(xgboost_config_path)
    if metadata.xgboost_config_sha256 != expected_config_sha256:
        raise ValueError("artifact XGBoost configuration hash does not match")
    if metadata.score_source_label != expected_score_source_label:
        raise ValueError("artifact score-source label does not match")
    _, expected_provenance = load_training_rows_with_provenance(
        input_directory,
        feature_directory,
    )
    if metadata.training_provenance != expected_provenance:
        raise ValueError("artifact training provenance does not match")
    if metadata.artifact_sha256 != _sha256_file(artifact_path):
        raise ValueError("artifact checksum does not match metadata")

    pipeline = joblib.load(artifact_path)
    _validate_pipeline(pipeline)
    return LoadedProvisionalModel(pipeline=pipeline, metadata=metadata)


def load_training_rows_with_provenance(
    input_directory: Path,
    feature_directory: Path,
) -> tuple[tuple[JoinedFeatureRow, ...], TrainingProvenance]:
    """Load only the two training files and bind them to deterministic hashes."""
    transaction_path = input_directory / SPLIT_FILENAMES[TRAINING_SPLIT]
    temporal_feature_path = feature_directory / FEATURE_FILENAMES[TRAINING_SPLIT]
    transaction_bytes = transaction_path.read_bytes()
    temporal_feature_bytes = temporal_feature_path.read_bytes()
    transactions = load_training_transactions(transaction_path)
    temporal_features = load_temporal_feature_csv(temporal_feature_path)
    rows = join_transaction_features(transactions, temporal_features)
    if transaction_path.read_bytes() != transaction_bytes:
        raise ValueError("training transactions changed while loading")
    if temporal_feature_path.read_bytes() != temporal_feature_bytes:
        raise ValueError("training temporal features changed while loading")
    return rows, TrainingProvenance(
        split=TRAINING_SPLIT,
        transaction_rows=len(rows),
        fraud_rows=sum(row.transaction.tx_fraud for row in rows),
        transactions_sha256=_sha256_bytes(transaction_bytes),
        temporal_features_sha256=_sha256_bytes(temporal_feature_bytes),
    )


def render_provisional_model_metadata(metadata: ProvisionalModelMetadata) -> str:
    """Render byte-stable metadata without clocks or host-specific paths."""
    return json.dumps(metadata.to_document(), indent=2, sort_keys=True) + "\n"


def load_provisional_model_metadata(path: Path) -> ProvisionalModelMetadata:
    """Load and strictly validate one deterministic metadata sidecar."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid artifact metadata JSON: {error.msg}") from error
    if not isinstance(document, Mapping):
        raise ValueError("artifact metadata must be a JSON object")
    _require_exact_keys(document, _METADATA_KEYS, "artifact metadata")
    provenance_document = document["training_provenance"]
    if not isinstance(provenance_document, Mapping):
        raise ValueError("training_provenance must be a JSON object")
    _require_exact_keys(
        provenance_document,
        _PROVENANCE_KEYS,
        "training_provenance",
    )
    feature_contract = document["feature_contract"]
    if not isinstance(feature_contract, list) or not all(
        isinstance(value, str) for value in feature_contract
    ):
        raise ValueError("feature_contract must be a list of strings")
    return ProvisionalModelMetadata(
        artifact_version=_require_string(document, "artifact_version"),
        artifact_sha256=_require_string(document, "artifact_sha256"),
        xgboost_config_sha256=_require_string(
            document,
            "xgboost_config_sha256",
        ),
        score_source_label=_require_string(document, "score_source_label"),
        feature_contract=tuple(feature_contract),
        training_provenance=TrainingProvenance(
            split=_require_string(provenance_document, "split"),
            transaction_rows=_require_integer(
                provenance_document,
                "transaction_rows",
            ),
            fraud_rows=_require_integer(provenance_document, "fraud_rows"),
            transactions_sha256=_require_string(
                provenance_document,
                "transactions_sha256",
            ),
            temporal_features_sha256=_require_string(
                provenance_document,
                "temporal_features_sha256",
            ),
        ),
    )


def _validate_pipeline(pipeline: object) -> None:
    if not isinstance(pipeline, Pipeline):
        raise ValueError("artifact must contain a scikit-learn Pipeline")
    if tuple(pipeline.named_steps) != ("preprocessing", "classifier"):
        raise ValueError("artifact pipeline steps do not match the frozen baseline")


def _temporary_sibling(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} SHA-256 must be 64 lowercase hexadecimal characters")


def _require_exact_keys(
    document: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(document) != expected:
        raise ValueError(f"{name} fields do not match the artifact contract")


def _require_string(document: Mapping[str, object], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_integer(document: Mapping[str, object], key: str) -> int:
    value = document[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value
