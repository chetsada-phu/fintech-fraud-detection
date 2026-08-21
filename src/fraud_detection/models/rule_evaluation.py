"""Reproducible validation and test evaluation for the rule baseline."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from fraud_detection.data.eda import ProcessedSplits, load_processed_splits
from fraud_detection.data.schema import Transaction
from fraud_detection.models.rules import (
    REASON_CODE_ORDER,
    RuleConfig,
    apply_rules,
)


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    """Measured binary-classification facts for one chronological split."""

    split_name: str
    rows: int
    frauds: int
    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    fraud_amount_total: Decimal
    fraud_amount_flagged: Decimal
    scenario_counts: tuple[int, int, int, int]
    reason_counts: tuple[tuple[str, int], ...]

    @property
    def flag_rate(self) -> float:
        """Return the fraction sent to the rule flag."""
        return self.flagged / self.rows

    @property
    def precision(self) -> float | None:
        """Return fraud precision among flags, if at least one row was flagged."""
        if self.flagged == 0:
            return None
        return self.true_positives / self.flagged

    @property
    def recall(self) -> float | None:
        """Return the captured fraud-label fraction, if fraud labels exist."""
        if self.frauds == 0:
            return None
        return self.true_positives / self.frauds

    @property
    def false_positive_rate(self) -> float | None:
        """Return the flagged fraction among legitimate transactions."""
        legitimate_rows = self.false_positives + self.true_negatives
        if legitimate_rows == 0:
            return None
        return self.false_positives / legitimate_rows

    @property
    def fraud_amount_capture(self) -> float | None:
        """Return the captured fraction of labeled-fraud transaction amount."""
        if self.fraud_amount_total == 0:
            return None
        return float(self.fraud_amount_flagged / self.fraud_amount_total)


@dataclass(frozen=True, slots=True)
class RuleBaselineReport:
    """Rule configuration and held-out chronological evaluation results."""

    config: RuleConfig
    validation: RuleMetrics
    test: RuleMetrics


def evaluate_rule_baseline(
    input_directory: Path, config: RuleConfig
) -> RuleBaselineReport:
    """Validate all splits, then score validation and test without fitting."""
    splits = load_processed_splits(input_directory)
    return evaluate_loaded_splits(splits, config)


def evaluate_loaded_splits(
    splits: ProcessedSplits, config: RuleConfig
) -> RuleBaselineReport:
    """Evaluate already-validated chronological splits."""
    return RuleBaselineReport(
        config=config,
        validation=_evaluate_transactions("validation", splits.validation, config),
        test=_evaluate_transactions("test", splits.test, config),
    )


def render_markdown(report: RuleBaselineReport) -> str:
    """Render deterministic, portfolio-safe rule evaluation Markdown."""
    config = report.config
    metrics = (report.validation, report.test)
    lines = [
        "# Phase 2 Rule Baseline",
        "",
        "Generated reproducibly by `fraud-evaluate-rules` from the validated",
        "chronological splits under `data/processed/`. Rules read decision-time",
        "fields only; fraud labels are used after scoring for evaluation.",
        "",
        "## Versioned Rules",
        "",
        f"- `{REASON_CODE_ORDER[0]}`: flag `TX_AMOUNT` strictly greater than "
        f"{config.high_amount_threshold:.2f}.",
        f"- `{REASON_CODE_ORDER[1]}`: flag UTC hours at or after "
        f"{config.unusual_hour_start_utc:02d}:00 and before "
        f"{config.unusual_hour_end_utc:02d}:00.",
        "- A transaction is flagged when either rule fires. This binary flag is a",
        "  baseline, not a final approve/review/decline policy.",
        "",
        "## Chronological Evaluation",
        "",
        "Accuracy is intentionally omitted because fraud is highly imbalanced.",
        "",
        "| Dataset | Rows | Frauds | Flagged | Flag rate | Precision | Recall | "
        "False-positive rate | Fraud amount captured |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_metrics_row(item) for item in metrics)
    lines.extend(
        [
            "",
            "## Reason-Code Counts",
            "",
            "Counts are rule activations, so a transaction can contribute to both",
            "reason codes.",
            "",
            "| Dataset | Reason code | Activations |",
            "| --- | --- | ---: |",
        ]
    )
    for item in metrics:
        lines.extend(
            f"| {item.split_name.capitalize()} | `{reason_code}` | {count:,} |"
            for reason_code, count in item.reason_counts
        )
    lines.extend(
        [
            "",
            "## Fraud-Scenario Support",
            "",
            "Scenario labels are simulation-only evaluation details and never rule",
            "inputs.",
            "",
            "| Dataset | Scenario 0 | Scenario 1 | Scenario 2 | Scenario 3 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {item.split_name.capitalize()} | {item.scenario_counts[0]:,} | "
        f"{item.scenario_counts[1]:,} | {item.scenario_counts[2]:,} | "
        f"{item.scenario_counts[3]:,} |"
        for item in metrics
    )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- The amount threshold mirrors a synthetic-generator assumption; its",
            "  measured performance is not evidence for a real payment system.",
            "- The overnight UTC rule is an illustrative heuristic, not a learned",
            "  customer-specific behavior pattern.",
            "- Fraud counts and scenario support are small and uneven across splits.",
            "- Threshold selection, manual-review capacity, and three-way decisions",
            "  remain future decisioning work.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: RuleBaselineReport, output_path: Path) -> None:
    """Write the deterministic rule evaluation report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _evaluate_transactions(
    split_name: str,
    transactions: Sequence[Transaction],
    config: RuleConfig,
) -> RuleMetrics:
    results = tuple(apply_rules(transaction, config) for transaction in transactions)
    true_positives = sum(
        result.flagged and transaction.tx_fraud == 1
        for transaction, result in zip(transactions, results, strict=True)
    )
    false_positives = sum(
        result.flagged and transaction.tx_fraud == 0
        for transaction, result in zip(transactions, results, strict=True)
    )
    false_negatives = sum(
        not result.flagged and transaction.tx_fraud == 1
        for transaction, result in zip(transactions, results, strict=True)
    )
    true_negatives = sum(
        not result.flagged and transaction.tx_fraud == 0
        for transaction, result in zip(transactions, results, strict=True)
    )
    fraud_amount_total = sum(
        (transaction.tx_amount for transaction in transactions if transaction.tx_fraud),
        start=Decimal(0),
    )
    fraud_amount_flagged = sum(
        (
            transaction.tx_amount
            for transaction, result in zip(transactions, results, strict=True)
            if transaction.tx_fraud and result.flagged
        ),
        start=Decimal(0),
    )
    return RuleMetrics(
        split_name=split_name,
        rows=len(transactions),
        frauds=true_positives + false_negatives,
        flagged=true_positives + false_positives,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        fraud_amount_total=fraud_amount_total,
        fraud_amount_flagged=fraud_amount_flagged,
        scenario_counts=tuple(
            sum(
                transaction.tx_fraud_scenario == scenario
                for transaction in transactions
            )
            for scenario in range(4)
        ),
        reason_counts=tuple(
            (
                reason_code,
                sum(reason_code in result.reason_codes for result in results),
            )
            for reason_code in REASON_CODE_ORDER
        ),
    )


def _metrics_row(metrics: RuleMetrics) -> str:
    return (
        f"| {metrics.split_name.capitalize()} | {metrics.rows:,} | "
        f"{metrics.frauds:,} | {metrics.flagged:,} | "
        f"{_format_percent(metrics.flag_rate)} | "
        f"{_format_percent(metrics.precision)} | "
        f"{_format_percent(metrics.recall)} | "
        f"{_format_percent(metrics.false_positive_rate)} | "
        f"{_format_percent(metrics.fraud_amount_capture)} |"
    )


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"
