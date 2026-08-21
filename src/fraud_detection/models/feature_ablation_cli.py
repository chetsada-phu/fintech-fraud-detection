"""CLI for validation-only XGBoost feature-group ablation."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.decisioning.costs import load_business_cost_config
from fraud_detection.models.feature_ablation import (
    evaluate_feature_ablation,
    load_feature_ablation_config,
    write_markdown_report,
)
from fraud_detection.models.xgboost_model import load_xgboost_config

DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_FEATURE_DIRECTORY = Path("data/processed/features")
DEFAULT_XGBOOST_CONFIG = Path("configs/xgboost_baseline.toml")
DEFAULT_ABLATION_CONFIG = Path("configs/xgboost_feature_ablation.toml")
DEFAULT_COST_CONFIG = Path("configs/business_costs.toml")
DEFAULT_OUTPUT = Path("docs/xgboost_feature_ablation_report.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the validation-only feature-ablation parser."""
    parser = argparse.ArgumentParser(
        description="Compare fixed XGBoost feature groups on validation only."
    )
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument(
        "--feature-directory", type=Path, default=DEFAULT_FEATURE_DIRECTORY
    )
    parser.add_argument("--xgboost-config", type=Path, default=DEFAULT_XGBOOST_CONFIG)
    parser.add_argument("--ablation-config", type=Path, default=DEFAULT_ABLATION_CONFIG)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Fit fixed variants on train and score validation only."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    try:
        xgboost_config_bytes = arguments.xgboost_config.read_bytes()
        report = evaluate_feature_ablation(
            arguments.input_directory,
            arguments.feature_directory,
            load_xgboost_config(arguments.xgboost_config),
            hashlib.sha256(xgboost_config_bytes).hexdigest(),
            load_feature_ablation_config(arguments.ablation_config),
            load_business_cost_config(arguments.cost_config),
        )
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("xgboost_feature_ablation_failed error=%s", error)
        return 1

    LOGGER.info(
        "xgboost_feature_ablation_complete output=%s train_rows=%d "
        "validation_rows=%d variants=%d",
        arguments.output,
        report.train_rows,
        report.validation_rows,
        len(report.results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
