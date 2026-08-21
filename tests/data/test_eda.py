"""Tests for deterministic focused EDA of chronological splits."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.eda import (
    profile_processed_splits,
    render_markdown,
    write_markdown_report,
)
from fraud_detection.data.eda_cli import main
from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.data.splitter import SplitConfig, split_transactions_csv

START = datetime(2018, 4, 1, tzinfo=UTC)


def test_profile_reports_measured_split_and_overall_metrics(tmp_path: Path) -> None:
    """Profiles should measure class balance, amounts, and entity coverage."""
    processed_directory = _create_processed_splits(tmp_path)

    report = profile_processed_splits(processed_directory)

    assert [profile.rows for profile in report.splits] == [6, 3, 3]
    assert report.overall.rows == 12
    assert report.overall.frauds == 3
    assert report.overall.amount_min == Decimal("10.00")
    assert report.overall.amount_median == Decimal("15.50")
    assert report.overall.amount_mean == Decimal("15.50")
    assert report.overall.amount_p95 == Decimal("21.00")
    assert report.overall.amount_max == Decimal("21.00")
    assert report.overall.customers == 3
    assert report.overall.terminals == 2
    assert report.overall.scenario_counts == (9, 1, 1, 1)


def test_profile_rejects_timestamp_overlap_between_splits(tmp_path: Path) -> None:
    """Equal timestamps may not appear on opposite sides of a split boundary."""
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    transactions = _transactions(seconds=(0, 1, 1, 2, 3, 4))
    write_transactions_csv(transactions[:2], processed_directory / "train.csv")
    write_transactions_csv(transactions[2:4], processed_directory / "validation.csv")
    write_transactions_csv(transactions[4:], processed_directory / "test.csv")

    with pytest.raises(ValueError, match="strictly ordered"):
        profile_processed_splits(processed_directory)


def test_markdown_report_is_byte_stable(tmp_path: Path) -> None:
    """Identical profiles should produce identical portfolio report bytes."""
    report = profile_processed_splits(_create_processed_splits(tmp_path))
    first_output = tmp_path / "first.md"
    second_output = tmp_path / "second.md"

    write_markdown_report(report, first_output)
    write_markdown_report(report, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert render_markdown(report) == first_output.read_text(encoding="utf-8")
    assert "| Overall | 12 |" in render_markdown(report)
    assert "These descriptive measurements do not establish model performance" in (
        render_markdown(report)
    )
    assert "Fraud-scenario support is uneven" in render_markdown(report)


def test_cli_writes_focused_eda_report(tmp_path: Path) -> None:
    """The installed command should write a non-empty deterministic report."""
    processed_directory = _create_processed_splits(tmp_path)
    output_path = tmp_path / "eda.md"

    exit_code = main(
        [
            "--input-directory",
            str(processed_directory),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8").startswith("# Phase 1 Focused EDA\n")


def _create_processed_splits(tmp_path: Path) -> Path:
    input_path = tmp_path / "raw.csv"
    output_directory = tmp_path / "processed"
    write_transactions_csv(_transactions(), input_path)
    split_transactions_csv(input_path, output_directory, SplitConfig(0.50, 0.25, 0.25))
    return output_directory


def _transactions(
    seconds: tuple[int, ...] = tuple(range(12)),
) -> tuple[Transaction, ...]:
    transactions = []
    fraud_scenarios = {2: 1, 7: 2, 11: 3}
    for transaction_id, tx_seconds in enumerate(seconds):
        scenario = fraud_scenarios.get(transaction_id, 0)
        transactions.append(
            Transaction(
                transaction_id=transaction_id,
                tx_datetime=START + timedelta(seconds=tx_seconds),
                customer_id=transaction_id % 3,
                terminal_id=transaction_id % 2,
                tx_amount=Decimal("10.00") + transaction_id,
                tx_time_seconds=tx_seconds,
                tx_time_days=tx_seconds // SECONDS_PER_DAY,
                tx_fraud=int(scenario != 0),
                tx_fraud_scenario=scenario,
            )
        )
    return tuple(transactions)
