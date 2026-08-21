"""Integration test for the installed data-generation command."""

import csv
from pathlib import Path

from fraud_detection.data.cli import main
from fraud_detection.data.schema import CSV_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_generation.toml"


def test_cli_generates_a_labeled_csv(tmp_path: Path) -> None:
    """The CLI should generate a non-empty file with the public schema."""
    output_path = tmp_path / "transactions.csv"

    exit_code = main(["--config", str(CONFIG_PATH), "--output", str(output_path)])

    assert exit_code == 0
    with output_path.open(encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
    assert rows
    assert {row["TX_FRAUD"] for row in rows} == {"0", "1"}
