"""CLI for reproducible rule-versus-logistic baseline comparison."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.decisioning.costs import load_business_cost_config
from fraud_detection.models.baseline_comparison import (
    compare_baselines,
    write_markdown_report,
)
from fraud_detection.models.logistic import load_logistic_config
from fraud_detection.models.rules import load_rule_config

DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_LOGISTIC_CONFIG_PATH = Path("configs/logistic_baseline.toml")
DEFAULT_RULE_CONFIG_PATH = Path("configs/rule_baseline.toml")
DEFAULT_COST_CONFIG_PATH = Path("configs/business_costs.toml")
DEFAULT_OUTPUT_PATH = Path("docs/baseline_comparison.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the baseline-comparison command parser."""
    parser = argparse.ArgumentParser(
        description="Fit train-only Logistic Regression and compare held-out baselines."
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"processed split directory (default: {DEFAULT_INPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--logistic-config",
        type=Path,
        default=DEFAULT_LOGISTIC_CONFIG_PATH,
        help=f"Logistic TOML path (default: {DEFAULT_LOGISTIC_CONFIG_PATH})",
    )
    parser.add_argument(
        "--rule-config",
        type=Path,
        default=DEFAULT_RULE_CONFIG_PATH,
        help=f"rule TOML path (default: {DEFAULT_RULE_CONFIG_PATH})",
    )
    parser.add_argument(
        "--cost-config",
        type=Path,
        default=DEFAULT_COST_CONFIG_PATH,
        help=f"business-cost TOML path (default: {DEFAULT_COST_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Markdown report path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the Logistic baseline and write held-out comparison metrics."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)

    try:
        report = compare_baselines(
            arguments.input_directory,
            load_logistic_config(arguments.logistic_config),
            load_rule_config(arguments.rule_config),
            load_business_cost_config(arguments.cost_config),
        )
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("baseline_comparison_failed error=%s", error)
        return 1

    LOGGER.info(
        "baseline_comparison_complete input=%s output=%s train_rows=%d "
        "validation_rows=%d test_rows=%d",
        arguments.input_directory,
        arguments.output,
        report.train_rows,
        report.validation[0].rows,
        report.test[0].rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
