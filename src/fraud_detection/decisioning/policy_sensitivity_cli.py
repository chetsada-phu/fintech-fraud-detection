"""CLI for validation-only decision-policy business-cost sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.decisioning.costs import load_business_cost_config
from fraud_detection.decisioning.policy import load_decision_policy_config
from fraud_detection.decisioning.policy_sensitivity import (
    load_policy_sensitivity_config,
)
from fraud_detection.decisioning.policy_sensitivity_evaluation import (
    evaluate_policy_sensitivity,
    write_markdown_report,
)
from fraud_detection.models.xgboost_model import load_xgboost_config

DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_FEATURE_DIRECTORY = Path("data/processed/features")
DEFAULT_XGBOOST_CONFIG = Path("configs/xgboost_baseline.toml")
DEFAULT_POLICY_CONFIG = Path("configs/decision_policy.toml")
DEFAULT_SENSITIVITY_CONFIG = Path("configs/decision_policy_sensitivity.toml")
DEFAULT_COST_CONFIG = Path("configs/business_costs.toml")
DEFAULT_OUTPUT = Path("docs/decision_policy_sensitivity_report.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the validation-only cost-sensitivity parser."""
    parser = argparse.ArgumentParser(
        description="Analyze decision-policy cost sensitivity on validation only."
    )
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument(
        "--feature-directory", type=Path, default=DEFAULT_FEATURE_DIRECTORY
    )
    parser.add_argument("--xgboost-config", type=Path, default=DEFAULT_XGBOOST_CONFIG)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument(
        "--sensitivity-config", type=Path, default=DEFAULT_SENSITIVITY_CONFIG
    )
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score validation once and write the ordered scenario report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    try:
        xgboost_config_bytes = arguments.xgboost_config.read_bytes()
        report = evaluate_policy_sensitivity(
            arguments.input_directory,
            arguments.feature_directory,
            load_xgboost_config(arguments.xgboost_config),
            hashlib.sha256(xgboost_config_bytes).hexdigest(),
            load_decision_policy_config(arguments.policy_config),
            load_policy_sensitivity_config(arguments.sensitivity_config),
            load_business_cost_config(arguments.cost_config),
        )
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("policy_sensitivity_failed error=%s", error)
        return 1

    LOGGER.info(
        "policy_sensitivity_complete output=%s validation_rows=%d scenarios=%d",
        arguments.output,
        report.validation_rows,
        len(report.analysis.results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
