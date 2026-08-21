"""Tests for deterministic label-free PSI monitoring."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import CSV_COLUMNS
from fraud_detection.features.temporal import (
    TEMPORAL_FEATURE_COLUMNS,
    TemporalFeatureRow,
)
from fraud_detection.models.model_input import ProvisionalModelInput
from fraud_detection.monitoring.drift import (
    InputDriftConfig,
    analyze_input_drift,
    build_input_drift_report,
    load_input_drift_config,
    render_markdown,
    write_markdown_report,
)
from fraud_detection.monitoring.drift_cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "input_drift.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_versioned_config_fixes_sources_bins_and_features() -> None:
    config = load_input_drift_config(CONFIG_PATH)

    assert config == InputDriftConfig(
        monitor_version="label-free-psi-v1",
        reference_split="train",
        comparison_split="validation",
        quantile_bin_count=5,
        smoothing_epsilon=0.000001,
        moderate_psi_threshold=0.10,
        high_psi_threshold=0.25,
        features=(
            "tx_amount",
            "customer_tx_count_short_window",
            "customer_tx_count_long_window",
            "customer_amount_deviation_from_mean_prior",
            "customer_seconds_since_previous",
        ),
    )


def test_reference_quantiles_fix_boundaries_and_missing_bin() -> None:
    reference = tuple(_input(index, float(index + 1)) for index in range(5))
    comparison = tuple(
        _input(10 + index, value, missing=index == 4, day=1)
        for index, value in enumerate((1.0, 1.0, 5.0, 6.0, 7.0))
    )

    result = analyze_input_drift(reference, comparison, _config())[0]

    assert result.boundaries == (1.0, 2.0, 3.0, 4.0)
    assert tuple(bin_.reference_rows for bin_ in result.bins) == (1, 1, 1, 1, 1, 0)
    assert tuple(bin_.comparison_rows for bin_ in result.bins) == (2, 0, 0, 0, 2, 1)
    assert result.bins[-1].label == "Missing"
    assert math_is_finite(result.psi)


def test_zero_count_smoothing_keeps_psi_finite_and_identical_inputs_are_zero() -> None:
    reference = tuple(_input(index, value) for index, value in enumerate((1, 2, 3)))
    comparison = tuple(
        _input(10 + index, value, day=1) for index, value in enumerate((1, 2, 3))
    )

    result = analyze_input_drift(reference, comparison, _config())[0]

    assert result.psi == pytest.approx(0.0)
    assert all(math_is_finite(bin_.psi_contribution) for bin_ in result.bins)


def test_report_loader_ignores_labels_and_does_not_require_test_split(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "processed"
    feature_directory = input_directory / "features"
    feature_directory.mkdir(parents=True)
    _write_split(input_directory, feature_directory, "train", 0, START, "secret-a")
    _write_split(
        input_directory,
        feature_directory,
        "validation",
        1,
        START + timedelta(days=1),
        "secret-b",
    )

    report = build_input_drift_report(_config(), input_directory, feature_directory)

    assert report.reference_rows == 1
    assert report.comparison_rows == 1
    assert not (input_directory / "test.csv").exists()
    assert "secret-a" not in render_markdown(report)
    assert "secret-b" not in render_markdown(report)


def test_report_and_cli_outputs_are_reproducible(tmp_path: Path) -> None:
    input_directory = tmp_path / "processed"
    feature_directory = input_directory / "features"
    feature_directory.mkdir(parents=True)
    _write_split(input_directory, feature_directory, "train", 0, START, "ignored")
    _write_split(
        input_directory,
        feature_directory,
        "validation",
        1,
        START + timedelta(days=1),
        "ignored",
    )
    report = build_input_drift_report(_config(), input_directory, feature_directory)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    write_markdown_report(report, first)
    write_markdown_report(report, second)

    assert first.read_bytes() == second.read_bytes()
    cli_output = tmp_path / "cli.md"
    assert (
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--input-directory",
                str(input_directory),
                "--feature-directory",
                str(feature_directory),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    assert cli_output.read_text(encoding="utf-8").startswith(
        "# Label-free input drift report\n"
    )


def test_invalid_boundaries_and_source_order_fail_closed() -> None:
    with pytest.raises(ValueError, match="reference must end"):
        analyze_input_drift(
            (_input(0, 1.0, day=2),),
            (_input(1, 2.0, day=1),),
            _config(),
        )
    with pytest.raises(ValueError, match="reference feature"):
        analyze_input_drift(
            (_input(0, 1.0, missing=True),),
            (_input(1, 2.0, missing=True, day=1),),
            _config(feature="customer_seconds_since_previous"),
        )


def _config(*, feature: str = "customer_seconds_since_previous") -> InputDriftConfig:
    return InputDriftConfig(
        monitor_version="fixture-v1",
        reference_split="train",
        comparison_split="validation",
        quantile_bin_count=5,
        smoothing_epsilon=0.000001,
        moderate_psi_threshold=0.10,
        high_psi_threshold=0.25,
        features=(feature,),
    )


def _input(
    transaction_id: int,
    value: float,
    *,
    missing: bool = False,
    day: int = 0,
) -> ProvisionalModelInput:
    history_value = None if missing else Decimal(str(value))
    history_seconds = None if missing else int(value)
    return ProvisionalModelInput(
        transaction_id=transaction_id,
        tx_amount=Decimal(str(value)),
        tx_time_days=day,
        tx_datetime=START + timedelta(days=day, seconds=transaction_id),
        customer_tx_count_short_window=0,
        customer_tx_count_long_window=0,
        customer_amount_mean_prior=history_value,
        customer_amount_deviation_from_mean_prior=history_value,
        customer_seconds_since_previous=history_seconds,
        customer_id=transaction_id,
        terminal_id=transaction_id,
    )


def _write_split(
    input_directory: Path,
    feature_directory: Path,
    split: str,
    transaction_id: int,
    timestamp: datetime,
    label_value: str,
) -> None:
    transaction_path = input_directory / f"{split}.csv"
    with transaction_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "TRANSACTION_ID": transaction_id,
                "TX_DATETIME": timestamp.isoformat(),
                "CUSTOMER_ID": 2,
                "TERMINAL_ID": 3,
                "TX_AMOUNT": "20.00",
                "TX_TIME_SECONDS": 0,
                "TX_TIME_DAYS": 0,
                "TX_FRAUD": label_value,
                "TX_FRAUD_SCENARIO": label_value,
            }
        )
    temporal = TemporalFeatureRow(
        transaction_id=transaction_id,
        customer_tx_count_short_window=1,
        customer_tx_count_long_window=2,
        customer_amount_mean_prior=Decimal("15.00"),
        customer_amount_deviation_from_mean_prior=Decimal("5.00"),
        customer_seconds_since_previous=60,
        terminal_tx_count_short_window=1,
        terminal_tx_count_long_window=2,
        terminal_amount_mean_prior=Decimal("15.00"),
        terminal_amount_deviation_from_mean_prior=Decimal("5.00"),
        terminal_seconds_since_previous=60,
    )
    feature_path = feature_directory / f"{split}_temporal_features.csv"
    with feature_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TEMPORAL_FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerow(temporal.to_csv_row())


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
