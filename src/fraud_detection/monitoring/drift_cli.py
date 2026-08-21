"""CLI for the deterministic label-free input drift report."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.monitoring.drift import (
    build_input_drift_report,
    load_input_drift_config,
    write_markdown_report,
)

DEFAULT_CONFIG_PATH = Path("configs/input_drift.toml")
DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_FEATURE_DIRECTORY = Path("data/processed/features")
DEFAULT_OUTPUT_PATH = Path("docs/input_drift_report.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the input drift command parser."""
    parser = argparse.ArgumentParser(
        description="Compare label-free train and validation model inputs with PSI."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument(
        "--feature-directory", type=Path, default=DEFAULT_FEATURE_DIRECTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one checked input drift report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    try:
        report = build_input_drift_report(
            load_input_drift_config(arguments.config),
            arguments.input_directory,
            arguments.feature_directory,
        )
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("input_drift_failed error=%s", error)
        return 1
    LOGGER.info(
        "input_drift_complete output=%s reference_rows=%d comparison_rows=%d",
        arguments.output,
        report.reference_rows,
        report.comparison_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
