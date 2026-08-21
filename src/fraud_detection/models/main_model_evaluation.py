"""Held-out comparison for the fixed temporal-feature XGBoost baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fraud_detection.data.eda import ProcessedSplits, load_processed_splits
from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.costs import BusinessCostConfig
from fraud_detection.features.matrix import (
    JoinedFeatureRow,
    JoinedFeatureSplits,
    load_joined_feature_splits,
)
from fraud_detection.models.baseline_comparison import (
    BaselineMetrics,
    evaluate_scores,
    metrics_markdown_row,
)
from fraud_detection.models.logistic import (
    LogisticConfig,
    fit_logistic_baseline,
    predict_fraud_scores,
)
from fraud_detection.models.rules import RuleConfig, apply_rules
from fraud_detection.models.validation_diagnostics import (
    SegmentMetrics,
    ValidationDiagnostics,
    ValidationDiagnosticsConfig,
    analyze_validation_scores,
)
from fraud_detection.models.xgboost_model import (
    XGBoostConfig,
    fit_xgboost_baseline,
    predict_xgboost_scores,
)


@dataclass(frozen=True, slots=True)
class MainModelReport:
    """One held-out split compared under frozen model configurations."""

    split_name: str
    train_rows: int
    train_frauds: int
    xgboost_config: XGBoostConfig
    metrics: tuple[BaselineMetrics, BaselineMetrics, BaselineMetrics]
    validation_diagnostics: ValidationDiagnostics | None


def evaluate_main_model(
    input_directory: Path,
    feature_directory: Path,
    split_name: str,
    xgboost_config: XGBoostConfig,
    logistic_config: LogisticConfig,
    rule_config: RuleConfig,
    cost_config: BusinessCostConfig,
    diagnostics_config: ValidationDiagnosticsConfig,
) -> MainModelReport:
    """Fit train-only baselines and score exactly one named held-out split."""
    transaction_splits = load_processed_splits(input_directory)
    joined_splits = load_joined_feature_splits(transaction_splits, feature_directory)
    return evaluate_loaded_main_model(
        transaction_splits,
        joined_splits,
        split_name,
        xgboost_config,
        logistic_config,
        rule_config,
        cost_config,
        diagnostics_config,
    )


def evaluate_loaded_main_model(
    transaction_splits: ProcessedSplits,
    joined_splits: JoinedFeatureSplits,
    split_name: str,
    xgboost_config: XGBoostConfig,
    logistic_config: LogisticConfig,
    rule_config: RuleConfig,
    cost_config: BusinessCostConfig,
    diagnostics_config: ValidationDiagnosticsConfig,
) -> MainModelReport:
    """Fit on train and evaluate validation or test without tuning."""
    transactions = _select_transaction_split(transaction_splits, split_name)
    joined_rows = _select_joined_split(joined_splits, split_name)
    logistic_pipeline = fit_logistic_baseline(transaction_splits.train, logistic_config)
    xgboost_pipeline = fit_xgboost_baseline(joined_splits.train, xgboost_config)
    rule_scores = tuple(
        float(apply_rules(transaction, rule_config).flagged)
        for transaction in transactions
    )
    logistic_scores = predict_fraud_scores(logistic_pipeline, transactions)
    xgboost_scores = predict_xgboost_scores(xgboost_pipeline, joined_rows)
    validation_diagnostics = (
        analyze_validation_scores(
            joined_rows,
            xgboost_scores,
            xgboost_config.flag_threshold,
            diagnostics_config,
        )
        if split_name == "validation"
        else None
    )
    return MainModelReport(
        split_name=split_name,
        train_rows=len(transaction_splits.train),
        train_frauds=sum(
            transaction.tx_fraud for transaction in transaction_splits.train
        ),
        xgboost_config=xgboost_config,
        metrics=(
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
            evaluate_scores(
                "XGBoost",
                split_name,
                transactions,
                xgboost_scores,
                xgboost_config.flag_threshold,
                cost_config,
            ),
        ),
        validation_diagnostics=validation_diagnostics,
    )


def render_markdown(report: MainModelReport) -> str:
    """Render deterministic metrics for one explicitly selected split."""
    config = report.xgboost_config
    class_ratio = (report.train_rows - report.train_frauds) / report.train_frauds
    split_display = report.split_name.capitalize()
    test_note = (
        "This is the one-time test report produced after freezing the configuration."
        if report.split_name == "test"
        else "The test split is not scored by this validation report."
    )
    lines = [
        f"# Phase 3 XGBoost {split_display} Report",
        "",
        "Generated reproducibly by `fraud-evaluate-xgboost`. All models fit only",
        f"the chronological training split ({report.train_rows:,} rows,",
        f"{report.train_frauds:,} fraud labels). {test_note}",
        "",
        "## Frozen XGBoost Configuration",
        "",
        "- Inputs: Phase 2 decision-time fields plus the validated past-only",
        "  customer temporal features. Transaction ID and fraud labels are excluded.",
        "- Missing prior history remains missing and is handled natively by XGBoost.",
        "- Customer and terminal IDs are one-hot encoded using training data only.",
        f"- Trees {config.n_estimators}, maximum depth {config.max_depth}, learning "
        f"rate {config.learning_rate:g}, minimum child weight "
        f"{config.min_child_weight:g}.",
        f"- Row sample {config.subsample:.2f}, column sample "
        f"{config.column_sample_by_tree:.2f}, L1 {config.regularization_alpha:g}, "
        f"L2 {config.regularization_lambda:g}.",
        f"- Training-only class ratio weight: {class_ratio:.2f}; random state "
        f"{config.random_state}; one worker for deterministic fitting.",
        f"- Fixed binary flag threshold: {config.flag_threshold:.2f}.",
        "",
        f"## {split_display} Comparison",
        "",
        "Average precision is the primary ranking metric. Accuracy is omitted.",
        "",
        "| Dataset | Baseline | AP | ROC-AUC | Flag rate | Precision | Recall | "
        "FPR | Fraud amount captured | Cost per 1,000 | Within capacity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(metrics_markdown_row(metrics) for metrics in report.metrics)
    if report.validation_diagnostics is not None:
        lines.extend(_validation_diagnostics_markdown(report))
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- This is one fixed baseline, not a hyperparameter search or a claim of",
            "  model promotion.",
            "- The synthetic sample has few and uneven fraud scenarios, so measured",
            "  differences can be unstable.",
            "- The fixed threshold remains a baseline and is not optimized for the",
            "  simulated manual-review constraint.",
            *(
                (
                    "- Calibration and segment results describe validation only; no",
                    "  calibrator or threshold was fitted from these labels.",
                )
                if report.validation_diagnostics is not None
                else (
                    "- Validation-only calibration and segment diagnostics are",
                    "  intentionally not computed from the one-time test scores.",
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _validation_diagnostics_markdown(report: MainModelReport) -> list[str]:
    diagnostics = report.validation_diagnostics
    if diagnostics is None:
        return []
    bin_count = diagnostics.config.calibration_bin_count
    lines = [
        "",
        "## Validation-only Calibration Diagnostics",
        "",
        "These diagnostics use the frozen validation scores after scoring. They do",
        "not fit a calibrator or change the model configuration or threshold.",
        "Because the validation split is highly imbalanced, the Brier score can be",
        "dominated by legitimate transactions and must be read with the reliability",
        "table and ranking metrics.",
        "",
        f"- Brier score: {diagnostics.brier_score:.4f}.",
        "- Expected calibration error: "
        f"{diagnostics.expected_calibration_error:.4f} across {bin_count} fixed "
        "equal-width bins.",
        "",
        "| Score interval | Rows | Frauds | Mean score | Observed fraud rate | "
        "Absolute gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for calibration_bin in diagnostics.calibration_bins:
        closing = "]" if calibration_bin.includes_upper_bound else ")"
        lines.append(
            f"| [{calibration_bin.lower_bound:.2f}, "
            f"{calibration_bin.upper_bound:.2f}{closing} | "
            f"{calibration_bin.rows:,} | {calibration_bin.frauds:,} | "
            f"{_format_rate(calibration_bin.mean_score)} | "
            f"{_format_percent(calibration_bin.observed_fraud_rate)} | "
            f"{_format_rate(calibration_bin.absolute_gap)} |"
        )
    lines.extend(
        [
            "",
            "## Validation Segment Error Analysis",
            "",
            "Amount bands are fixed in `configs/validation_diagnostics.toml`; they",
            "are not derived from validation or test outcomes. Errors use the frozen",
            f"XGBoost threshold of {report.xgboost_config.flag_threshold:.2f}.",
            "",
            "### Transaction amount",
            "",
            *_segment_table(diagnostics.amount_segments),
            "",
            "### Prior customer history",
            "",
            "Prior history is available only when the validated prior mean, amount",
            "deviation, and seconds-since-previous fields are present together.",
            "",
            *_segment_table(diagnostics.history_segments),
            *(
                (
                    "",
                    "No validation rows lack prior history, so behavior for that",
                    "segment cannot be assessed from this split.",
                )
                if diagnostics.history_segments[0].rows == 0
                else ()
            ),
            "",
            "## Provisional Challenger Decision",
            "",
            "**Decision: retain XGBoost as a provisional challenger; do not promote",
            "it as the selected main model.**",
            "",
            *_challenger_rationale(report),
        ]
    )
    return lines


def _segment_table(segments: tuple[SegmentMetrics, ...]) -> list[str]:
    lines = [
        "| Segment | Rows | Frauds | Fraud rate | Mean score | Flagged | "
        "Flag rate | TP | FP | FN | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    for segment in segments:
        lines.append(
            f"| {segment.segment_name} | {segment.rows:,} | {segment.frauds:,} | "
            f"{_format_percent(segment.fraud_rate)} | "
            f"{_format_rate(segment.mean_score)} | {segment.flagged:,} | "
            f"{_format_percent(segment.flag_rate)} | {segment.true_positives:,} | "
            f"{segment.false_positives:,} | {segment.false_negatives:,} | "
            f"{_format_percent(segment.precision)} | "
            f"{_format_percent(segment.recall)} |"
        )
    return lines


def _challenger_rationale(report: MainModelReport) -> list[str]:
    logistic_metrics = report.metrics[1]
    xgboost_metrics = report.metrics[2]
    if (
        logistic_metrics.average_precision is None
        or xgboost_metrics.average_precision is None
    ):
        average_precision_summary = (
            "- Average-precision evidence is unavailable for a reliable comparison."
        )
    else:
        difference = (
            xgboost_metrics.average_precision - logistic_metrics.average_precision
        )
        average_precision_summary = (
            f"- Validation AP is {xgboost_metrics.average_precision:.4f} versus "
            f"{logistic_metrics.average_precision:.4f} for Logistic Regression "
            f"(difference {difference:+.4f}), which is not a material advantage on "
            "this small split."
        )
    return [
        average_precision_summary,
        f"- Validation ROC-AUC is {_format_rate(xgboost_metrics.roc_auc)} and recall "
        f"at the frozen threshold is {_format_percent(xgboost_metrics.recall)}; "
        "ranking and capture evidence remain weak.",
        "- Calibration and segment tables are descriptive and have sparse fraud",
        "  support. They do not justify fitting a calibrator or promoting the model",
        "  from the same validation labels.",
        "- The one-time test result is retained as a frozen report and is not used",
        "  to revise this decision or any configuration.",
    ]


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def write_markdown_report(report: MainModelReport, output_path: Path) -> None:
    """Write one held-out report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _select_transaction_split(
    splits: ProcessedSplits, split_name: str
) -> tuple[Transaction, ...]:
    if split_name == "validation":
        return splits.validation
    if split_name == "test":
        return splits.test
    raise ValueError("split_name must be 'validation' or 'test'")


def _select_joined_split(
    splits: JoinedFeatureSplits, split_name: str
) -> tuple[JoinedFeatureRow, ...]:
    if split_name == "validation":
        return splits.validation
    if split_name == "test":
        return splits.test
    raise ValueError("split_name must be 'validation' or 'test'")
