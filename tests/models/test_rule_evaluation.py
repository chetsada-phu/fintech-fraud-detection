"""Integration tests for chronological rule-baseline evaluation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.models.rule_cli import main
from fraud_detection.models.rule_evaluation import (
    evaluate_rule_baseline,
    render_markdown,
)
from fraud_detection.models.rules import load_rule_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "rule_baseline.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_evaluation_reports_held_out_metrics_without_accuracy(tmp_path: Path) -> None:
    """Validation and test metrics should expose imbalanced-data trade-offs."""
    processed_directory = _write_processed_splits(tmp_path)

    report = evaluate_rule_baseline(processed_directory, load_rule_config(CONFIG_PATH))

    assert report.validation.rows == 3
    assert report.validation.frauds == 1
    assert report.validation.flagged == 2
    assert report.validation.precision == pytest.approx(0.5)
    assert report.validation.recall == pytest.approx(1.0)
    assert report.validation.false_positive_rate == pytest.approx(0.5)
    assert report.test.rows == 3
    assert report.test.frauds == 1
    assert report.test.flagged == 1
    assert report.test.precision == pytest.approx(0.0)
    assert report.test.recall == pytest.approx(0.0)
    markdown = render_markdown(report)
    assert "Accuracy is intentionally omitted" in markdown
    assert "| Validation | 3 | 1 | 2 |" in markdown
    assert "| Test | 3 | 1 | 1 |" in markdown


def test_cli_writes_byte_stable_rule_report(tmp_path: Path) -> None:
    """The installed workflow should write the same report for the same splits."""
    processed_directory = _write_processed_splits(tmp_path)
    first_output = tmp_path / "first.md"
    second_output = tmp_path / "second.md"

    first_exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--input-directory",
            str(processed_directory),
            "--output",
            str(first_output),
        ]
    )
    second_exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--input-directory",
            str(processed_directory),
            "--output",
            str(second_output),
        ]
    )

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_output.read_text(encoding="utf-8").startswith(
        "# Phase 2 Rule Baseline\n"
    )


def _write_processed_splits(tmp_path: Path) -> Path:
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    transactions = _transactions()
    write_transactions_csv(transactions[:3], processed_directory / "train.csv")
    write_transactions_csv(transactions[3:6], processed_directory / "validation.csv")
    write_transactions_csv(transactions[6:], processed_directory / "test.csv")
    return processed_directory


def _transactions() -> tuple[Transaction, ...]:
    seconds = (
        10 * 3_600,
        11 * 3_600,
        12 * 3_600,
        SECONDS_PER_DAY + (23 * 3_600),
        (2 * SECONDS_PER_DAY) + (12 * 3_600),
        (2 * SECONDS_PER_DAY) + (13 * 3_600),
        (3 * SECONDS_PER_DAY) + (12 * 3_600),
        (3 * SECONDS_PER_DAY) + (13 * 3_600),
        (3 * SECONDS_PER_DAY) + (14 * 3_600),
    )
    fraud_scenarios = {4: 1, 6: 2}
    amounts = {
        4: Decimal("230.00"),
        7: Decimal("230.00"),
    }
    return tuple(
        Transaction(
            transaction_id=transaction_id,
            tx_datetime=START + timedelta(seconds=tx_seconds),
            customer_id=transaction_id % 3,
            terminal_id=transaction_id % 2,
            tx_amount=amounts.get(transaction_id, Decimal("100.00")),
            tx_time_seconds=tx_seconds,
            tx_time_days=tx_seconds // SECONDS_PER_DAY,
            tx_fraud=int(transaction_id in fraud_scenarios),
            tx_fraud_scenario=fraud_scenarios.get(transaction_id, 0),
        )
        for transaction_id, tx_seconds in enumerate(seconds)
    )
