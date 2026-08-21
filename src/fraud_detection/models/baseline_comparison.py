"""Train-only Logistic Regression and rule-baseline comparison."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fraud_detection.data.eda import ProcessedSplits, load_processed_splits
from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.costs import (
    BusinessCostConfig,
    OperatingCost,
    calculate_operating_cost,
)
from fraud_detection.models.logistic import (
    LogisticConfig,
    fit_logistic_baseline,
    predict_fraud_scores,
)
from fraud_detection.models.rules import RuleConfig, apply_rules


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    """Held-out ranking, classification, amount, and cost measurements."""

    model_name: str
    split_name: str
    threshold: float
    rows: int
    frauds: int
    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    average_precision: float | None
    roc_auc: float | None
    fraud_amount_total: Decimal
    fraud_amount_flagged: Decimal
    operating_cost: OperatingCost

    @property
    def flag_rate(self) -> float:
        """Return the fraction sent to manual review by the binary baseline."""
        return self.flagged / self.rows

    @property
    def precision(self) -> float | None:
        """Return fraud precision among flags, if any rows were flagged."""
        if self.flagged == 0:
            return None
        return self.true_positives / self.flagged

    @property
    def recall(self) -> float | None:
        """Return fraud-label recall, if the split contains fraud."""
        if self.frauds == 0:
            return None
        return self.true_positives / self.frauds

    @property
    def false_positive_rate(self) -> float | None:
        """Return the flag rate among legitimate transactions."""
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
class BaselineComparisonReport:
    """Fixed configurations and chronological baseline measurements."""

    train_rows: int
    train_frauds: int
    logistic_config: LogisticConfig
    rule_config: RuleConfig
    cost_config: BusinessCostConfig
    validation: tuple[BaselineMetrics, BaselineMetrics]
    test: tuple[BaselineMetrics, BaselineMetrics]


def compare_baselines(
    input_directory: Path,
    logistic_config: LogisticConfig,
    rule_config: RuleConfig,
    cost_config: BusinessCostConfig,
) -> BaselineComparisonReport:
    """Load chronological data, fit train only, and evaluate held-out splits."""
    splits = load_processed_splits(input_directory)
    return compare_loaded_splits(splits, logistic_config, rule_config, cost_config)


def compare_loaded_splits(
    splits: ProcessedSplits,
    logistic_config: LogisticConfig,
    rule_config: RuleConfig,
    cost_config: BusinessCostConfig,
) -> BaselineComparisonReport:
    """Fit the ML baseline on train and compare fixed held-out behavior."""
    pipeline = fit_logistic_baseline(splits.train, logistic_config)
    validation = _evaluate_split(
        "validation",
        splits.validation,
        pipeline,
        logistic_config,
        rule_config,
        cost_config,
    )
    test = _evaluate_split(
        "test",
        splits.test,
        pipeline,
        logistic_config,
        rule_config,
        cost_config,
    )
    return BaselineComparisonReport(
        train_rows=len(splits.train),
        train_frauds=sum(transaction.tx_fraud for transaction in splits.train),
        logistic_config=logistic_config,
        rule_config=rule_config,
        cost_config=cost_config,
        validation=validation,
        test=test,
    )


def render_markdown(report: BaselineComparisonReport) -> str:
    """Render a deterministic, caveated rule-versus-logistic report."""
    logistic = report.logistic_config
    costs = report.cost_config
    all_metrics = (*report.validation, *report.test)
    lines = [
        "# Phase 2 Baseline Comparison",
        "",
        "Generated reproducibly by `fraud-compare-baselines`. Logistic Regression",
        f"fits only the chronological training split ({report.train_rows:,} rows,",
        f"{report.train_frauds:,} fraud labels). Validation and test labels are used",
        "only after scoring. No held-out hyperparameter search is performed.",
        "",
        "## Logistic Regression Configuration",
        "",
        "- Decision-time inputs: transaction amount, elapsed day, cyclical UTC",
        "  hour, customer ID, and terminal ID.",
        "- Numeric inputs are standardized using training data only; IDs are",
        "  one-hot encoded with unseen held-out values ignored.",
        f"- Solver `{logistic.solver}`, class weight `{logistic.class_weight}`, "
        f"C={logistic.regularization_c:g}, maximum iterations "
        f"{logistic.max_iterations:,}, random state {logistic.random_state}.",
        f"- Fixed binary flag threshold: {logistic.flag_threshold:.2f}.",
        "",
        "## Simulated Business Assumptions",
        "",
        f"- Missed fraud loss: transaction amount x {costs.fraud_loss_multiplier:.2f}.",
        f"- Manual review cost: {costs.manual_review_cost:.2f} per flag.",
        f"- Maximum manual-review rate: {costs.max_manual_review_rate * 100:.2f}%.",
        f"- Reserved false-decline cost: {costs.false_decline_cost:.2f}; not used",
        "  until a three-way decision policy exists.",
        "- Costs are illustrative portfolio assumptions, not bank economics.",
        "",
        "## Held-out Comparison",
        "",
        "Average precision is the primary ranking metric. Accuracy is omitted.",
        "The rule baseline's binary flags are treated as scores of zero or one.",
        "",
        "| Dataset | Baseline | AP | ROC-AUC | Flag rate | Precision | Recall | "
        "FPR | Fraud amount captured | Cost per 1,000 | Within capacity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(metrics_markdown_row(metrics) for metrics in all_metrics)
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- The Logistic Regression probabilities are not calibrated; balanced",
            "  class weights intentionally change the fitted class emphasis.",
            "- Customer and terminal IDs are synthetic categories, not portable",
            "  behavioral risk features.",
            "- Fraud counts and scenario support are small and uneven across splits.",
            "- The fixed 0.50 threshold is a baseline, not a business-optimized",
            "  approve/review/decline policy.",
            "- Test results are a one-time report and must not be used to revise this",
            "  configuration. Future choices should use training and validation only.",
            "- Neither baseline is automatically promoted from this comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: BaselineComparisonReport, output_path: Path) -> None:
    """Write the deterministic baseline comparison atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _evaluate_split(
    split_name: str,
    transactions: Sequence[Transaction],
    pipeline: Pipeline,
    logistic_config: LogisticConfig,
    rule_config: RuleConfig,
    cost_config: BusinessCostConfig,
) -> tuple[BaselineMetrics, BaselineMetrics]:
    rule_scores = tuple(
        float(apply_rules(transaction, rule_config).flagged)
        for transaction in transactions
    )
    logistic_scores = predict_fraud_scores(pipeline, transactions)
    return (
        evaluate_scores(
            "Rules", split_name, transactions, rule_scores, 0.5, cost_config
        ),
        evaluate_scores(
            "Logistic Regression",
            split_name,
            transactions,
            logistic_scores,
            logistic_config.flag_threshold,
            cost_config,
        ),
    )


def evaluate_scores(
    model_name: str,
    split_name: str,
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    threshold: float,
    cost_config: BusinessCostConfig,
) -> BaselineMetrics:
    if len(transactions) != len(scores):
        raise ValueError("transactions and scores must have equal lengths")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("baseline scores must be finite and within [0, 1]")

    labels = tuple(transaction.tx_fraud for transaction in transactions)
    flags = tuple(score >= threshold for score in scores)
    true_positives = sum(
        flagged and label == 1 for flagged, label in zip(flags, labels, strict=True)
    )
    false_positives = sum(
        flagged and label == 0 for flagged, label in zip(flags, labels, strict=True)
    )
    false_negatives = sum(
        not flagged and label == 1 for flagged, label in zip(flags, labels, strict=True)
    )
    true_negatives = sum(
        not flagged and label == 0 for flagged, label in zip(flags, labels, strict=True)
    )
    fraud_amount_total = sum(
        (transaction.tx_amount for transaction in transactions if transaction.tx_fraud),
        start=Decimal(0),
    )
    fraud_amount_flagged = sum(
        (
            transaction.tx_amount
            for transaction, flagged in zip(transactions, flags, strict=True)
            if transaction.tx_fraud and flagged
        ),
        start=Decimal(0),
    )
    frauds = sum(labels)
    has_both_classes = 0 < frauds < len(labels)
    return BaselineMetrics(
        model_name=model_name,
        split_name=split_name,
        threshold=threshold,
        rows=len(transactions),
        frauds=frauds,
        flagged=sum(flags),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        average_precision=(
            float(average_precision_score(labels, scores)) if frauds else None
        ),
        roc_auc=(float(roc_auc_score(labels, scores)) if has_both_classes else None),
        fraud_amount_total=fraud_amount_total,
        fraud_amount_flagged=fraud_amount_flagged,
        operating_cost=calculate_operating_cost(transactions, flags, cost_config),
    )


def metrics_markdown_row(metrics: BaselineMetrics) -> str:
    return (
        f"| {metrics.split_name.capitalize()} | {metrics.model_name} | "
        f"{_format_rate(metrics.average_precision)} | "
        f"{_format_rate(metrics.roc_auc)} | "
        f"{_format_percent(metrics.flag_rate)} | "
        f"{_format_percent(metrics.precision)} | "
        f"{_format_percent(metrics.recall)} | "
        f"{_format_percent(metrics.false_positive_rate)} | "
        f"{_format_percent(metrics.fraud_amount_capture)} | "
        f"{metrics.operating_cost.total_per_1000_transactions:,.2f} | "
        f"{'yes' if metrics.operating_cost.within_review_capacity else 'no'} |"
    )


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"
