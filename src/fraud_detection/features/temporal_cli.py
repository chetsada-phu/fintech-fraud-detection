"""CLI for reproducible past-only customer feature generation."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.features.temporal import (
    build_processed_temporal_features,
    load_temporal_feature_config,
)

DEFAULT_CONFIG_PATH = Path("configs/temporal_features.toml")
DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_OUTPUT_DIRECTORY = Path("data/processed/features")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the temporal-feature command parser."""
    parser = argparse.ArgumentParser(
        description="Build leakage-safe customer features across chronological splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"feature TOML path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"processed split directory (default: {DEFAULT_INPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"feature output directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build deterministic temporal feature files for all three splits."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)

    try:
        result = build_processed_temporal_features(
            arguments.input_directory,
            arguments.output_directory,
            load_temporal_feature_config(arguments.config),
        )
    except (OSError, ValueError) as error:
        LOGGER.error("temporal_feature_build_failed error=%s", error)
        return 1

    LOGGER.info(
        "temporal_feature_build_complete input=%s output=%s "
        "train_rows=%d validation_rows=%d test_rows=%d",
        arguments.input_directory,
        arguments.output_directory,
        result.train_rows,
        result.validation_rows,
        result.test_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
