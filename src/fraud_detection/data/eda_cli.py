"""Command-line interface for reproducible focused transaction EDA."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.data.eda import profile_processed_splits, write_markdown_report

DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_OUTPUT_PATH = Path("docs/eda_report.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the focused-EDA command parser."""
    parser = argparse.ArgumentParser(
        description="Validate chronological splits and write focused EDA Markdown."
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
    """Profile validated chronological splits and write their report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)

    try:
        report = profile_processed_splits(arguments.input_directory)
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("data_profile_failed error=%s", error)
        return 1

    LOGGER.info(
        "data_profile_complete input=%s output=%s rows=%d frauds=%d fraud_rate=%.6f",
        arguments.input_directory,
        arguments.output,
        report.overall.rows,
        report.overall.frauds,
        report.overall.frauds / report.overall.rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
