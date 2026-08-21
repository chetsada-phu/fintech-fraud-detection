"""Tests for validation-only XGBoost score diagnostics."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.schema import Transaction
from fraud_detection.features.matrix import JoinedFeatureRow
from fraud_detection.features.temporal import TemporalFeatureRow
from fraud_detection.models.validation_diagnostics import (
    ValidationDiagnosticsConfig,
    analyze_validation_scores,
    load_validation_diagnostics_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "validation_diagnostics.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_versioned_validation_diagnostic_settings_are_fixed() -> None:
    assert load_validation_diagnostics_config(CONFIG_PATH) == (
        ValidationDiagnosticsConfig(
            calibration_bin_count=5,
            amount_band_upper_bounds=(
                Decimal("50.0"),
                Decimal("100.0"),
                Decimal("220.0"),
            ),
        )
    )


def test_calibration_and_segments_use_only_supplied_validation_rows() -> None:
    rows = (
        _row(0, "20.00", 0, has_history=False),
        _row(1, "75.00", 0, has_history=True),
        _row(2, "150.00", 1, has_history=True),
        _row(3, "300.00", 1, has_history=False),
    )
    diagnostics = analyze_validation_scores(
        rows,
        (0.1, 0.2, 0.8, 1.0),
        0.5,
        load_validation_diagnostics_config(CONFIG_PATH),
    )

    assert diagnostics.brier_score == pytest.approx(0.0225)
    assert diagnostics.expected_calibration_error == pytest.approx(0.125)
    assert tuple(
        calibration_bin.rows for calibration_bin in diagnostics.calibration_bins
    ) == (
        1,
        1,
        0,
        0,
        2,
    )
    assert diagnostics.calibration_bins[-1].includes_upper_bound is True
    assert diagnostics.calibration_bins[-1].mean_score == pytest.approx(0.9)
    assert tuple(segment.rows for segment in diagnostics.amount_segments) == (
        1,
        1,
        1,
        1,
    )
    assert tuple(segment.frauds for segment in diagnostics.amount_segments) == (
        0,
        0,
        1,
        1,
    )
    assert tuple(segment.segment_name for segment in diagnostics.amount_segments) == (
        "Amount <= 50.00",
        "50.00 < amount <= 100.00",
        "100.00 < amount <= 220.00",
        "Amount > 220.00",
    )
    assert tuple(segment.segment_name for segment in diagnostics.history_segments) == (
        "Missing prior history",
        "Prior history available",
    )
    assert tuple(segment.rows for segment in diagnostics.history_segments) == (2, 2)
    assert tuple(
        segment.false_negatives for segment in diagnostics.history_segments
    ) == (
        0,
        0,
    )


def test_diagnostics_reject_partial_prior_history_and_invalid_scores() -> None:
    row = _row(0, "20.00", 0, has_history=True)
    partial_history = JoinedFeatureRow(
        transaction=row.transaction,
        temporal=TemporalFeatureRow(
            transaction_id=0,
            customer_tx_count_short_window=0,
            customer_tx_count_long_window=1,
            customer_amount_mean_prior=Decimal("10.00"),
            customer_amount_deviation_from_mean_prior=None,
            customer_seconds_since_previous=60,
            terminal_tx_count_short_window=0,
            terminal_tx_count_long_window=1,
            terminal_amount_mean_prior=Decimal("10.00"),
            terminal_amount_deviation_from_mean_prior=Decimal("10.00"),
            terminal_seconds_since_previous=60,
        ),
    )
    config = load_validation_diagnostics_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="missing together"):
        analyze_validation_scores((partial_history,), (0.2,), 0.5, config)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        analyze_validation_scores((row,), (1.1,), 0.5, config)


def _row(
    transaction_id: int,
    amount: str,
    fraud: int,
    *,
    has_history: bool,
) -> JoinedFeatureRow:
    transaction = Transaction(
        transaction_id=transaction_id,
        tx_datetime=START + timedelta(minutes=transaction_id),
        customer_id=transaction_id,
        terminal_id=transaction_id,
        tx_amount=Decimal(amount),
        tx_time_seconds=transaction_id * 60,
        tx_time_days=0,
        tx_fraud=fraud,
        tx_fraud_scenario=fraud,
    )
    temporal = TemporalFeatureRow(
        transaction_id=transaction_id,
        customer_tx_count_short_window=int(has_history),
        customer_tx_count_long_window=int(has_history),
        customer_amount_mean_prior=(Decimal("40.00") if has_history else None),
        customer_amount_deviation_from_mean_prior=(
            Decimal(amount) - Decimal("40.00") if has_history else None
        ),
        customer_seconds_since_previous=60 if has_history else None,
        terminal_tx_count_short_window=int(has_history),
        terminal_tx_count_long_window=int(has_history),
        terminal_amount_mean_prior=(Decimal("40.00") if has_history else None),
        terminal_amount_deviation_from_mean_prior=(
            Decimal(amount) - Decimal("40.00") if has_history else None
        ),
        terminal_seconds_since_previous=60 if has_history else None,
    )
    return JoinedFeatureRow(transaction=transaction, temporal=temporal)
