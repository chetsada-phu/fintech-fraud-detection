"""Command-line interface for deterministic raw data generation."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.data.generator import (
    generate_transactions,
    load_simulation_config,
    write_transactions_csv,
)

DEFAULT_CONFIG_PATH = Path("configs/data_generation.toml")
DEFAULT_OUTPUT_PATH = Path("data/raw/transactions.csv")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the data-generation command parser."""
    parser = argparse.ArgumentParser(
        description="Generate a deterministic, validated ULB-style transaction CSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"simulation TOML path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"generated CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and atomically write a validated transaction dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)

    try:
        config = load_simulation_config(arguments.config)
        transactions = generate_transactions(config)
        write_transactions_csv(transactions, arguments.output)
    except (OSError, ValueError) as error:
        LOGGER.error("data_generation_failed error=%s", error)
        return 1

    scenario_counts = Counter(
        transaction.tx_fraud_scenario for transaction in transactions
    )
    fraud_count = len(transactions) - scenario_counts[0]
    digest = _sha256(arguments.output)
    LOGGER.info(
        "data_generation_complete output=%s rows=%d frauds=%d fraud_rate=%.6f "
        "scenario_1=%d scenario_2=%d scenario_3=%d seed=%d sha256=%s",
        arguments.output,
        len(transactions),
        fraud_count,
        fraud_count / len(transactions),
        scenario_counts[1],
        scenario_counts[2],
        scenario_counts[3],
        config.seed,
        digest,
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
