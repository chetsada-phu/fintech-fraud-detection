"""Offline monitoring for decisions after delayed labels arrive."""

from __future__ import annotations

import csv
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

DECISIONS: Final = ("approve", "review", "decline")
OUTCOME_COLUMNS: Final = (
    "TRANSACTION_ID",
    "RISK_SCORE",
    "DECISION",
    "TX_AMOUNT",
    "FRAUD_LABEL",
)


@dataclass(frozen=True, slots=True)
class PerformanceMonitoringConfig:
    """Version and boundary for one delayed-label example."""

    monitor_version: str
    source_label: str
    risky_decisions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.monitor_version.strip() or not self.source_label.strip():
            raise ValueError("monitor version and source label must not be empty")
        if not self.risky_decisions or len(set(self.risky_decisions)) != len(
            self.risky_decisions
        ):
            raise ValueError("risky_decisions must be non-empty and unique")
        if not set(self.risky_decisions).issubset(DECISIONS):
            raise ValueError("risky_decisions contain an unsupported decision")


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """One decision with an optional delayed fraud label."""

    transaction_id: int
    risk_score: float
    decision: str
    tx_amount: Decimal
    fraud_label: int | None

    def __post_init__(self) -> None:
        if type(self.transaction_id) is not int or self.transaction_id < 0:
            raise ValueError("transaction_id must be a non-negative integer")
        if not math.isfinite(self.risk_score) or not 0 <= self.risk_score <= 1:
            raise ValueError("risk_score must be within [0, 1]")
        if self.decision not in DECISIONS:
            raise ValueError("decision is unsupported")
        if not self.tx_amount.is_finite() or self.tx_amount <= 0:
            raise ValueError("tx_amount must be finite and positive")
        if self.fraud_label not in {None, 0, 1}:
            raise ValueError("fraud_label must be 0, 1, or pending")


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Metrics over labeled rows plus explicit label coverage."""

    total_rows: int
    labeled_rows: int
    pending_rows: int
    frauds: int
    risky_rows: int
    declines: int
    frauds_intercepted: int
    legitimate_declines: int
    label_coverage: float
    fraud_recall: float | None
    false_decline_rate: float | None
    fraud_amount_capture: float | None
    brier_score: float | None


def load_performance_monitoring_config(path: Path) -> PerformanceMonitoringConfig:
    """Load the strict delayed-label example contract."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    if set(document) != {"performance_monitoring"}:
        raise ValueError("configuration must contain [performance_monitoring]")
    section = document["performance_monitoring"]
    expected = {"monitor_version", "source_label", "risky_decisions"}
    if not isinstance(section, Mapping) or set(section) != expected:
        raise ValueError("performance_monitoring fields do not match the contract")
    decisions = section["risky_decisions"]
    if not isinstance(decisions, list) or not all(
        isinstance(decision, str) for decision in decisions
    ):
        raise ValueError("risky_decisions must be an array of strings")
    return PerformanceMonitoringConfig(
        monitor_version=_require_string(section, "monitor_version"),
        source_label=_require_string(section, "source_label"),
        risky_decisions=tuple(decisions),
    )


def load_decision_outcomes(path: Path) -> tuple[DecisionOutcome, ...]:
    """Load synthetic or post-event outcomes from a strict CSV contract."""
    with path.open(encoding="utf-8", newline="") as outcome_file:
        reader = csv.DictReader(outcome_file)
        if tuple(reader.fieldnames or ()) != OUTCOME_COLUMNS:
            raise ValueError("outcome columns do not match the monitoring contract")
        try:
            outcomes = tuple(_outcome_from_row(row) for row in reader)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid delayed outcome: {error}") from error
    if not outcomes:
        raise ValueError("at least one outcome row is required")
    ids = tuple(outcome.transaction_id for outcome in outcomes)
    if len(set(ids)) != len(ids):
        raise ValueError("transaction IDs must be unique")
    return outcomes


def summarize_delayed_outcomes(
    outcomes: Sequence[DecisionOutcome], config: PerformanceMonitoringConfig
) -> PerformanceSummary:
    """Measure only rows whose delayed labels are available."""
    if not outcomes:
        raise ValueError("at least one outcome row is required")
    labeled = tuple(row for row in outcomes if row.fraud_label is not None)
    pending_rows = len(outcomes) - len(labeled)
    if not labeled:
        return PerformanceSummary(
            total_rows=len(outcomes),
            labeled_rows=0,
            pending_rows=pending_rows,
            frauds=0,
            risky_rows=0,
            declines=0,
            frauds_intercepted=0,
            legitimate_declines=0,
            label_coverage=0.0,
            fraud_recall=None,
            false_decline_rate=None,
            fraud_amount_capture=None,
            brier_score=None,
        )
    risky = tuple(row for row in labeled if row.decision in config.risky_decisions)
    frauds = tuple(row for row in labeled if row.fraud_label == 1)
    legitimate = tuple(row for row in labeled if row.fraud_label == 0)
    intercepted = tuple(row for row in frauds if row.decision in config.risky_decisions)
    legitimate_declines = sum(
        row.fraud_label == 0 and row.decision == "decline" for row in labeled
    )
    total_fraud_amount = sum((row.tx_amount for row in frauds), Decimal(0))
    intercepted_amount = sum((row.tx_amount for row in intercepted), Decimal(0))
    return PerformanceSummary(
        total_rows=len(outcomes),
        labeled_rows=len(labeled),
        pending_rows=pending_rows,
        frauds=len(frauds),
        risky_rows=len(risky),
        declines=sum(row.decision == "decline" for row in labeled),
        frauds_intercepted=len(intercepted),
        legitimate_declines=legitimate_declines,
        label_coverage=len(labeled) / len(outcomes),
        fraud_recall=(len(intercepted) / len(frauds) if frauds else None),
        false_decline_rate=(
            legitimate_declines / len(legitimate) if legitimate else None
        ),
        fraud_amount_capture=(
            float(intercepted_amount / total_fraud_amount)
            if total_fraud_amount
            else None
        ),
        brier_score=(
            sum((row.risk_score - int(row.fraud_label)) ** 2 for row in labeled)
            / len(labeled)
        ),
    )


def render_markdown(
    config: PerformanceMonitoringConfig, summary: PerformanceSummary
) -> str:
    """Render the delayed-label monitoring example."""
    lines = [
        "# Delayed-label performance monitoring example",
        "",
        "Generated by `fraud-monitor-performance-example` from a small versioned",
        "outcome fixture. The rows are illustrative and are not model evaluation",
        "results from the project dataset.",
        "",
        "## Contract",
        "",
        f"- Monitor version: `{config.monitor_version}`.",
        f"- Source: {config.source_label}.",
        f"- Risky decisions: {', '.join(config.risky_decisions)}.",
        "- Pending outcomes count toward label coverage but not performance metrics.",
        "- This offline boundary is separate from the label-free scoring API.",
        "",
        "## Example window",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Received decisions | {summary.total_rows:,} |",
        f"| Labeled decisions | {summary.labeled_rows:,} |",
        f"| Pending labels | {summary.pending_rows:,} |",
        f"| Label coverage | {_format_percent(summary.label_coverage)} |",
        f"| Labeled frauds | {summary.frauds:,} |",
        f"| Risky decisions | {summary.risky_rows:,} |",
        f"| Declines | {summary.declines:,} |",
        f"| Frauds intercepted | {summary.frauds_intercepted:,} |",
        f"| Fraud recall | {_format_optional_percent(summary.fraud_recall)} |",
        f"| False-decline rate | {_format_optional_percent(summary.false_decline_rate)} |",
        f"| Fraud amount capture | {_format_optional_percent(summary.fraud_amount_capture)} |",
        f"| Brier score | {_format_optional_number(summary.brier_score)} |",
        "",
        "## Interpretation limits",
        "",
        "The fixture proves metric and delayed-label handling, not current model",
        "quality. A real monitoring window must record label maturity, source",
        "provenance, segment support, and the policy and artifact versions that",
        "produced each decision. Sparse or delayed labels can make early metrics",
        "misleading, so this command does not trigger alerts or retraining.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report(
    config: PerformanceMonitoringConfig,
    summary: PerformanceSummary,
    output_path: Path,
) -> None:
    """Write the example report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(config, summary), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _outcome_from_row(row: Mapping[str, str]) -> DecisionOutcome:
    raw_label = row["FRAUD_LABEL"]
    return DecisionOutcome(
        transaction_id=int(row["TRANSACTION_ID"]),
        risk_score=float(row["RISK_SCORE"]),
        decision=row["DECISION"],
        tx_amount=Decimal(row["TX_AMOUNT"]),
        fraud_label=None if raw_label == "" else int(raw_label),
    )


def _require_string(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_optional_percent(value: float | None) -> str:
    return "Not available" if value is None else _format_percent(value)


def _format_optional_number(value: float | None) -> str:
    return "Not available" if value is None else f"{value:.6f}"
