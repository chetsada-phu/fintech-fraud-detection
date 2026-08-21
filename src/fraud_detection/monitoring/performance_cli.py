"""CLI for the delayed-label performance monitoring example."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.monitoring.performance import (
    load_decision_outcomes,
    load_performance_monitoring_config,
    summarize_delayed_outcomes,
    write_markdown_report,
)

DEFAULT_CONFIG_PATH = Path("configs/performance_monitoring.toml")
DEFAULT_INPUT_PATH = Path("examples/delayed_outcomes.csv")
DEFAULT_OUTPUT_PATH = Path("docs/performance_monitoring_example.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the delayed-label monitoring parser."""
    parser = argparse.ArgumentParser(
        description="Summarize a versioned delayed-label monitoring example."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one deterministic delayed-label example report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    try:
        config = load_performance_monitoring_config(arguments.config)
        summary = summarize_delayed_outcomes(
            load_decision_outcomes(arguments.input), config
        )
        write_markdown_report(config, summary, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("performance_monitoring_failed error=%s", error)
        return 1
    LOGGER.info(
        "performance_monitoring_complete output=%s labeled=%d pending=%d",
        arguments.output,
        summary.labeled_rows,
        summary.pending_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
