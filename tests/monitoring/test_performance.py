"""Tests for offline delayed-label performance monitoring."""

from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.monitoring.performance import (
    DecisionOutcome,
    PerformanceMonitoringConfig,
    load_decision_outcomes,
    load_performance_monitoring_config,
    render_markdown,
    summarize_delayed_outcomes,
    write_markdown_report,
)
from fraud_detection.monitoring.performance_cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "performance_monitoring.toml"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "delayed_outcomes.csv"


def test_versioned_example_config_is_fixed() -> None:
    assert load_performance_monitoring_config(CONFIG_PATH) == (
        PerformanceMonitoringConfig(
            monitor_version="delayed-label-example-v1",
            source_label="illustrative synthetic decision outcomes",
            risky_decisions=("review", "decline"),
        )
    )


def test_metrics_use_only_mature_labels() -> None:
    config = load_performance_monitoring_config(CONFIG_PATH)
    summary = summarize_delayed_outcomes(load_decision_outcomes(EXAMPLE_PATH), config)

    assert summary.total_rows == 10
    assert summary.labeled_rows == 8
    assert summary.pending_rows == 2
    assert summary.frauds == 3
    assert summary.risky_rows == 4
    assert summary.declines == 2
    assert summary.frauds_intercepted == 2
    assert summary.legitimate_declines == 1
    assert summary.label_coverage == pytest.approx(0.8)
    assert summary.fraud_recall == pytest.approx(2 / 3)
    assert summary.false_decline_rate == pytest.approx(1 / 5)
    assert summary.fraud_amount_capture == pytest.approx(455 / 503)
    assert summary.brier_score == pytest.approx(0.309475)


def test_pending_only_window_reports_coverage_without_fabricating_metrics() -> None:
    summary = summarize_delayed_outcomes(
        (_outcome(1, 0.5, "review", None),),
        load_performance_monitoring_config(CONFIG_PATH),
    )

    assert summary.label_coverage == 0.0
    assert summary.pending_rows == 1
    assert summary.fraud_recall is None
    assert summary.false_decline_rate is None
    assert summary.fraud_amount_capture is None
    assert summary.brier_score is None


def test_invalid_decisions_scores_and_labels_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _outcome(1, 0.5, "hold", 0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _outcome(1, 1.1, "approve", 0)
    with pytest.raises(ValueError, match="0, 1, or pending"):
        _outcome(1, 0.5, "approve", 2)


def test_report_and_cli_are_byte_stable(tmp_path: Path) -> None:
    config = load_performance_monitoring_config(CONFIG_PATH)
    summary = summarize_delayed_outcomes(load_decision_outcomes(EXAMPLE_PATH), config)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    write_markdown_report(config, summary, first)
    write_markdown_report(config, summary, second)

    assert first.read_bytes() == second.read_bytes()
    assert "illustrative and are not model evaluation" in render_markdown(
        config, summary
    )
    cli_output = tmp_path / "cli.md"
    assert (
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--input",
                str(EXAMPLE_PATH),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    assert cli_output.read_bytes() == first.read_bytes()


def _outcome(
    transaction_id: int,
    risk_score: float,
    decision: str,
    fraud_label: int | None,
) -> DecisionOutcome:
    return DecisionOutcome(
        transaction_id=transaction_id,
        risk_score=risk_score,
        decision=decision,
        tx_amount=Decimal("100.00"),
        fraud_label=fraud_label,
    )
