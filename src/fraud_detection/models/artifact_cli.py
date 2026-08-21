"""CLI for the bounded offline provisional-model artifact build."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.decisioning.inference import (
    load_selected_decision_policy_config,
)
from fraud_detection.models.artifact import build_provisional_model_artifact

DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_FEATURE_DIRECTORY = Path("data/processed/features")
DEFAULT_XGBOOST_CONFIG = Path("configs/xgboost_baseline.toml")
DEFAULT_POLICY_CONFIG = Path("configs/selected_decision_policy.toml")
DEFAULT_ARTIFACT_PATH = Path("models/xgboost_engineering_provisional.joblib")
DEFAULT_METADATA_PATH = Path("models/xgboost_engineering_provisional.metadata.json")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the training-only artifact-build parser."""
    parser = argparse.ArgumentParser(
        description="Fit and persist the training-only provisional XGBoost model."
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
    )
    parser.add_argument(
        "--feature-directory",
        type=Path,
        default=DEFAULT_FEATURE_DIRECTORY,
    )
    parser.add_argument(
        "--xgboost-config",
        type=Path,
        default=DEFAULT_XGBOOST_CONFIG,
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=DEFAULT_POLICY_CONFIG,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one artifact from training without reading held-out splits."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    try:
        policy = load_selected_decision_policy_config(arguments.policy_config)
        metadata = build_provisional_model_artifact(
            arguments.input_directory,
            arguments.feature_directory,
            arguments.xgboost_config,
            policy.score_source_label,
            arguments.artifact,
            arguments.metadata,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("provisional_model_artifact_failed error=%s", error)
        return 1

    LOGGER.info(
        "provisional_model_artifact_complete artifact=%s metadata=%s "
        "train_rows=%d config_sha256=%s",
        arguments.artifact,
        arguments.metadata,
        metadata.training_provenance.transaction_rows,
        metadata.xgboost_config_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
