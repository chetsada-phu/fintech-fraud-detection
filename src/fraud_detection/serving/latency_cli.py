"""CLI for the local scored-decision API latency benchmark."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.serving.api import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_FEATURE_DIRECTORY,
    DEFAULT_INPUT_DIRECTORY,
    DEFAULT_METADATA_PATH,
    DEFAULT_POLICY_CONFIG_PATH,
    DEFAULT_REASON_CONFIG_PATH,
    DEFAULT_XGBOOST_CONFIG_PATH,
    create_scored_app,
)
from fraud_detection.serving.latency import (
    benchmark_scored_api,
    load_scored_api_latency_config,
    write_markdown_report,
)

DEFAULT_BENCHMARK_CONFIG_PATH = Path("configs/serving_latency.toml")
DEFAULT_OUTPUT_PATH = Path("docs/serving_latency_report.md")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the local scored-API benchmark parser."""
    parser = argparse.ArgumentParser(
        description="Measure warmed local scored-decision API p50 and p95 latency."
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG_PATH,
    )
    parser.add_argument(
        "--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG_PATH
    )
    parser.add_argument(
        "--reason-config", type=Path, default=DEFAULT_REASON_CONFIG_PATH
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--xgboost-config",
        type=Path,
        default=DEFAULT_XGBOOST_CONFIG_PATH,
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
    )
    parser.add_argument(
        "--feature-directory",
        type=Path,
        default=DEFAULT_FEATURE_DIRECTORY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one checked local benchmark and atomically write its report."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    arguments = build_parser().parse_args(argv)
    try:
        config = load_scored_api_latency_config(arguments.benchmark_config)
        report = benchmark_scored_api(
            config,
            lambda: create_scored_app(
                policy_config_path=arguments.policy_config,
                reason_config_path=arguments.reason_config,
                artifact_path=arguments.artifact,
                metadata_path=arguments.metadata,
                xgboost_config_path=arguments.xgboost_config,
                input_directory=arguments.input_directory,
                feature_directory=arguments.feature_directory,
            ),
        )
        write_markdown_report(report, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("scored_api_latency_failed error=%s", error)
        return 1

    LOGGER.info(
        "scored_api_latency_complete output=%s samples=%d p50_ms=%.3f p95_ms=%.3f",
        arguments.output,
        report.request_latency.samples,
        report.request_latency.p50_ms,
        report.request_latency.p95_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
