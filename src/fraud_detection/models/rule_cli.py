"""Command-line interface for the reproducible Phase 2 rule baseline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.models.rule_evaluation import (
    evaluate_rule_baseline,
    write_markdown_report,
)
from fraud_detection.models.rules import load_rule_config

DEFAULT_CONFIG_PATH = Path("configs/rule_baseline.toml")
DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_OUTPUT_PATH = Path("docs/rule_baseline_report.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the rule-evaluation command parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate transparent rules on chronological held-out splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"rule TOML path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"processed split directory (default: {DEFAULT_INPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Markdown report path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate rules and write the deterministic held-out report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)

    try:
        config = load_rule_config(arguments.config)
        report = evaluate_rule_baseline(arguments.input_directory, config)
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("rule_evaluation_failed error=%s", error)
        return 1

    LOGGER.info(
        "rule_evaluation_complete input=%s output=%s "
        "validation_rows=%d validation_flagged=%d test_rows=%d test_flagged=%d",
        arguments.input_directory,
        arguments.output,
        report.validation.rows,
        report.validation.flagged,
        report.test.rows,
        report.test.flagged,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
