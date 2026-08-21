"""Integration test for the train-only baseline comparison workflow."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.models.comparison_cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cli_writes_reproducible_rule_logistic_comparison(tmp_path: Path) -> None:
    """The CLI should fit train only and report both held-out splits."""
    processed_directory = _write_processed_splits(tmp_path)
    output_path = tmp_path / "comparison.md"

    exit_code = main(
        [
            "--input-directory",
            str(processed_directory),
            "--logistic-config",
            str(PROJECT_ROOT / "configs" / "logistic_baseline.toml"),
            "--rule-config",
            str(PROJECT_ROOT / "configs" / "rule_baseline.toml"),
            "--cost-config",
            str(PROJECT_ROOT / "configs" / "business_costs.toml"),
            "--output",
            str(output_path),
        ]
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert markdown.startswith("# Phase 2 Baseline Comparison\n")
    assert "training split (6 rows," in markdown
    assert "| Validation | Rules |" in markdown
    assert "| Validation | Logistic Regression |" in markdown
    assert "| Test | Logistic Regression |" in markdown
    assert "Accuracy is omitted" in markdown
    assert "No held-out hyperparameter search" in markdown


def _write_processed_splits(tmp_path: Path) -> Path:
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    transactions = _transactions()
    write_transactions_csv(transactions[:6], processed_directory / "train.csv")
    write_transactions_csv(transactions[6:9], processed_directory / "validation.csv")
    write_transactions_csv(transactions[9:], processed_directory / "test.csv")
    return processed_directory


def _transactions() -> tuple[Transaction, ...]:
    fraud_scenarios = {2: 1, 5: 2, 7: 3, 10: 1}
    return tuple(
        Transaction(
            transaction_id=transaction_id,
            tx_datetime=datetime(2018, 4, 1, tzinfo=UTC)
            + timedelta(hours=transaction_id * 7),
            customer_id=transaction_id % 4,
            terminal_id=transaction_id % 3,
            tx_amount=(
                Decimal("250.00")
                if transaction_id in fraud_scenarios
                else Decimal("50.00") + transaction_id
            ),
            tx_time_seconds=transaction_id * 7 * 3_600,
            tx_time_days=(transaction_id * 7 * 3_600) // SECONDS_PER_DAY,
            tx_fraud=int(transaction_id in fraud_scenarios),
            tx_fraud_scenario=fraud_scenarios.get(transaction_id, 0),
        )
        for transaction_id in range(12)
    )
