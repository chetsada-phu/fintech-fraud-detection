"""Command-line interface for chronological transaction-data splitting."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.data.splitter import (
    load_split_config,
    split_transactions_csv,
)

DEFAULT_CONFIG_PATH = Path("configs/data_split.toml")
DEFAULT_INPUT_PATH = Path("data/raw/transactions.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("data/processed")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the chronological-splitting command parser."""
    parser = argparse.ArgumentParser(
        description="Validate and chronologically split a transaction CSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"split TOML path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"validated raw CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"split output directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write deterministic train, validation, and test CSVs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)

    try:
        config = load_split_config(arguments.config)
        result = split_transactions_csv(
            arguments.input, arguments.output_directory, config
        )
    except (OSError, ValueError) as error:
        LOGGER.error("data_split_failed error=%s", error)
        return 1

    LOGGER.info(
        "data_split_complete input=%s train=%s train_rows=%d "
        "validation=%s validation_rows=%d test=%s test_rows=%d "
        "train_end=%s validation_start=%s validation_end=%s test_start=%s",
        arguments.input,
        result.train_path,
        result.train_rows,
        result.validation_path,
        result.validation_rows,
        result.test_path,
        result.test_rows,
        result.train_end.isoformat(),
        result.validation_start.isoformat(),
        result.validation_end.isoformat(),
        result.test_start.isoformat(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
