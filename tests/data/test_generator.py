"""Tests for deterministic transaction generation and CSV output."""

import csv
from pathlib import Path

from fraud_detection.data.generator import (
    generate_transactions,
    load_simulation_config,
    write_transactions_csv,
)
from fraud_detection.data.schema import CSV_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_generation.toml"


def test_default_generation_is_deterministic_and_imbalanced() -> None:
    """The same versioned seed must produce the same minority-fraud dataset."""
    config = load_simulation_config(CONFIG_PATH)

    first_run = generate_transactions(config)
    second_run = generate_transactions(config)

    assert first_run == second_run
    fraud_count = sum(transaction.tx_fraud for transaction in first_run)
    assert 0 < fraud_count < len(first_run) * 0.05
    assert [transaction.transaction_id for transaction in first_run] == list(
        range(len(first_run))
    )


def test_csv_output_is_byte_stable(tmp_path: Path) -> None:
    """Writing identical records twice must produce identical CSV bytes."""
    config = load_simulation_config(CONFIG_PATH)
    transactions = generate_transactions(config)
    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"

    write_transactions_csv(transactions, first_output)
    write_transactions_csv(transactions, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    with first_output.open(encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        assert sum(1 for _ in reader) == len(transactions)
