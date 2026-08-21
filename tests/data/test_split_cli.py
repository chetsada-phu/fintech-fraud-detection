"""Integration test for the installed chronological-splitting command."""

import csv
from pathlib import Path

from fraud_detection.data.generator import (
    generate_transactions,
    load_simulation_config,
    write_transactions_csv,
)
from fraud_detection.data.split_cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_generation.toml"
SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_split.toml"


def test_cli_writes_three_complete_chronological_splits(tmp_path: Path) -> None:
    """The CLI should consume the validated raw schema and write every row once."""
    transactions = generate_transactions(load_simulation_config(GENERATION_CONFIG_PATH))
    input_path = tmp_path / "transactions.csv"
    output_directory = tmp_path / "processed"
    write_transactions_csv(transactions, input_path)

    exit_code = main(
        [
            "--config",
            str(SPLIT_CONFIG_PATH),
            "--input",
            str(input_path),
            "--output-directory",
            str(output_directory),
        ]
    )

    assert exit_code == 0
    row_counts = []
    for filename in ("train.csv", "validation.csv", "test.csv"):
        with (output_directory / filename).open(
            encoding="utf-8", newline=""
        ) as data_file:
            row_counts.append(sum(1 for _ in csv.DictReader(data_file)))
    assert all(row_counts)
    assert sum(row_counts) == len(transactions)
