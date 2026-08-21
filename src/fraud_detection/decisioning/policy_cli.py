"""CLI for validation-only provisional decision-policy selection."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.decisioning.costs import load_business_cost_config
from fraud_detection.decisioning.policy import load_decision_policy_config
from fraud_detection.decisioning.policy_evaluation import (
    evaluate_decision_policy,
    write_markdown_report,
)
from fraud_detection.decisioning.reasons import load_decision_reason_config
from fraud_detection.models.xgboost_model import load_xgboost_config

DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_FEATURE_DIRECTORY = Path("data/processed/features")
DEFAULT_XGBOOST_CONFIG = Path("configs/xgboost_baseline.toml")
DEFAULT_POLICY_CONFIG = Path("configs/decision_policy.toml")
DEFAULT_COST_CONFIG = Path("configs/business_costs.toml")
DEFAULT_REASON_CONFIG = Path("configs/decision_reasons.toml")
DEFAULT_OUTPUT = Path("docs/decision_policy_report.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the validation-only decision-policy parser."""
    parser = argparse.ArgumentParser(
        description="Select provisional three-way thresholds on validation only."
    )
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument(
        "--feature-directory", type=Path, default=DEFAULT_FEATURE_DIRECTORY
    )
    parser.add_argument("--xgboost-config", type=Path, default=DEFAULT_XGBOOST_CONFIG)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument("--reason-config", type=Path, default=DEFAULT_REASON_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Fit on train, score validation, and write the provisional policy report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    try:
        xgboost_config_bytes = arguments.xgboost_config.read_bytes()
        report = evaluate_decision_policy(
            arguments.input_directory,
            arguments.feature_directory,
            load_xgboost_config(arguments.xgboost_config),
            hashlib.sha256(xgboost_config_bytes).hexdigest(),
            load_decision_policy_config(arguments.policy_config),
            load_business_cost_config(arguments.cost_config),
            load_decision_reason_config(arguments.reason_config),
        )
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("decision_policy_selection_failed error=%s", error)
        return 1

    LOGGER.info(
        "decision_policy_selection_complete output=%s validation_rows=%d "
        "review_threshold=%.2f decline_threshold=%.2f",
        arguments.output,
        report.validation_rows,
        report.selection.thresholds.review_threshold,
        report.selection.thresholds.decline_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
