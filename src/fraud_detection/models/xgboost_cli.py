"""CLI for fixed XGBoost validation and one-time test comparison."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.decisioning.costs import load_business_cost_config
from fraud_detection.models.logistic import load_logistic_config
from fraud_detection.models.main_model_evaluation import (
    evaluate_main_model,
    write_markdown_report,
)
from fraud_detection.models.rules import load_rule_config
from fraud_detection.models.validation_diagnostics import (
    load_validation_diagnostics_config,
)
from fraud_detection.models.xgboost_model import load_xgboost_config

DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_FEATURE_DIRECTORY = Path("data/processed/features")
DEFAULT_XGBOOST_CONFIG = Path("configs/xgboost_baseline.toml")
DEFAULT_LOGISTIC_CONFIG = Path("configs/logistic_baseline.toml")
DEFAULT_RULE_CONFIG = Path("configs/rule_baseline.toml")
DEFAULT_COST_CONFIG = Path("configs/business_costs.toml")
DEFAULT_DIAGNOSTICS_CONFIG = Path("configs/validation_diagnostics.toml")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the main-model evaluation parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate frozen XGBoost on one chronological held-out split."
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="held-out split to score (default: validation)",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"processed split directory (default: {DEFAULT_INPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--feature-directory",
        type=Path,
        default=DEFAULT_FEATURE_DIRECTORY,
        help=f"temporal feature directory (default: {DEFAULT_FEATURE_DIRECTORY})",
    )
    parser.add_argument("--xgboost-config", type=Path, default=DEFAULT_XGBOOST_CONFIG)
    parser.add_argument("--logistic-config", type=Path, default=DEFAULT_LOGISTIC_CONFIG)
    parser.add_argument("--rule-config", type=Path, default=DEFAULT_RULE_CONFIG)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument(
        "--diagnostics-config", type=Path, default=DEFAULT_DIAGNOSTICS_CONFIG
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="report path (default: docs/xgboost_<split>_report.md)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Fit train-only models and score one explicit held-out split."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    output_path = arguments.output or Path(f"docs/xgboost_{arguments.split}_report.md")
    try:
        report = evaluate_main_model(
            arguments.input_directory,
            arguments.feature_directory,
            arguments.split,
            load_xgboost_config(arguments.xgboost_config),
            load_logistic_config(arguments.logistic_config),
            load_rule_config(arguments.rule_config),
            load_business_cost_config(arguments.cost_config),
            load_validation_diagnostics_config(arguments.diagnostics_config),
        )
        write_markdown_report(report, output_path)
    except (OSError, ValueError) as error:
        LOGGER.error("xgboost_evaluation_failed error=%s", error)
        return 1

    LOGGER.info(
        "xgboost_evaluation_complete split=%s output=%s train_rows=%d held_out_rows=%d",
        arguments.split,
        output_path,
        report.train_rows,
        report.metrics[0].rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
