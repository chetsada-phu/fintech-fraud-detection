"""Reproducible validation-only report for decision-policy cost sensitivity."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fraud_detection.data.eda import load_processed_splits
from fraud_detection.decisioning.costs import BusinessCostConfig
from fraud_detection.decisioning.policy import DecisionPolicyConfig
from fraud_detection.decisioning.policy_sensitivity import (
    PolicySensitivityAnalysis,
    PolicySensitivityConfig,
    PolicySensitivityResult,
    analyze_policy_cost_sensitivity,
)
from fraud_detection.features.matrix import load_joined_feature_splits
from fraud_detection.models.xgboost_model import (
    XGBoostConfig,
    fit_xgboost_baseline,
    predict_xgboost_scores,
)


@dataclass(frozen=True, slots=True)
class PolicySensitivityReport:
    """Shared-score validation sensitivity across versioned cost scenarios."""

    train_rows: int
    train_frauds: int
    validation_rows: int
    validation_frauds: int
    xgboost_config_hash: str
    policy_config: DecisionPolicyConfig
    sensitivity_config: PolicySensitivityConfig
    base_cost_config: BusinessCostConfig
    analysis: PolicySensitivityAnalysis


def evaluate_policy_sensitivity(
    input_directory: Path,
    feature_directory: Path,
    xgboost_config: XGBoostConfig,
    xgboost_config_hash: str,
    policy_config: DecisionPolicyConfig,
    sensitivity_config: PolicySensitivityConfig,
    base_cost_config: BusinessCostConfig,
) -> PolicySensitivityReport:
    """Fit and score once, then evaluate isolated validation cost scenarios."""
    if policy_config.selection_split != "validation":
        raise ValueError("decision-policy selection must use validation only")
    if sensitivity_config.evaluation_split != "validation":
        raise ValueError("policy sensitivity must evaluate validation only")
    if len(xgboost_config_hash) != 64 or any(
        character not in "0123456789abcdef" for character in xgboost_config_hash
    ):
        raise ValueError("xgboost_config_hash must be a lowercase SHA-256 digest")

    transaction_splits = load_processed_splits(input_directory)
    joined_splits = load_joined_feature_splits(transaction_splits, feature_directory)
    pipeline = fit_xgboost_baseline(joined_splits.train, xgboost_config)
    validation_scores = predict_xgboost_scores(pipeline, joined_splits.validation)
    analysis = analyze_policy_cost_sensitivity(
        transaction_splits.validation,
        validation_scores,
        policy_config,
        sensitivity_config,
        base_cost_config,
        split_name="validation",
    )
    return PolicySensitivityReport(
        train_rows=len(joined_splits.train),
        train_frauds=sum(row.transaction.tx_fraud for row in joined_splits.train),
        validation_rows=len(joined_splits.validation),
        validation_frauds=sum(
            row.transaction.tx_fraud for row in joined_splits.validation
        ),
        xgboost_config_hash=xgboost_config_hash,
        policy_config=policy_config,
        sensitivity_config=sensitivity_config,
        base_cost_config=base_cost_config,
        analysis=analysis,
    )


def render_markdown(report: PolicySensitivityReport) -> str:
    """Render deterministic threshold and action-mix sensitivity."""
    results = report.analysis.results
    base_result = results[0]
    threshold_pairs = {
        (
            result.selection.thresholds.review_threshold,
            result.selection.thresholds.decline_threshold,
        )
        for result in results
    }
    action_mixes = {
        (
            result.selection.operating_cost.approvals,
            result.selection.operating_cost.reviews,
            result.selection.operating_cost.declines,
        )
        for result in results
    }
    lines = [
        "# Phase 4 Decision-policy Cost Sensitivity",
        "",
        "Generated reproducibly by `fraud-analyze-policy-sensitivity`. The frozen",
        f"XGBoost pipeline fits training ({report.train_rows:,} rows, "
        f"{report.train_frauds:,} fraud labels), then produces one shared score set",
        f"for chronological validation ({report.validation_rows:,} rows, "
        f"{report.validation_frauds:,} fraud labels). Every scenario reuses those",
        "scores. The one-time test split is not scored by this command.",
        "",
        "## Score-source and Isolation Contract",
        "",
        f"- Score source: **{report.policy_config.score_source_label}**.",
        "- XGBoost remains unpromoted and supports engineering analysis only.",
        f"- Frozen XGBoost config SHA-256: `{report.xgboost_config_hash}`.",
        "- Scenario order is fixed in `configs/decision_policy_sensitivity.toml`;",
        f"  `{report.sensitivity_config.base_scenario_key}` is evaluated first.",
        "- Every scenario creates a separate frozen `BusinessCostConfig` from the",
        "  base assumptions. The base configuration is not mutated.",
        f"- Threshold selection uses validation only and the unchanged "
        f"{report.policy_config.threshold_grid_step} grid.",
        "",
        "## Validation Sensitivity",
        "",
        "All monetary values are simulated portfolio assumptions. Cost values",
        "should be interpreted within each scenario, not as production economics.",
        "",
        "| Scenario | Fraud loss x | Review cost | False-decline cost | Capacity | "
        "Review threshold | Decline threshold | Approve | Review | Decline | "
        "Review rate | False declines | Captured amount | Cost per 1,000 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    lines.extend(_scenario_markdown_row(result) for result in results)
    lines.extend(
        [
            "",
            "## Stability Summary",
            "",
            f"- Unique selected threshold pairs: {len(threshold_pairs)} across "
            f"{len(results)} scenarios.",
            f"- Unique approve/review/decline mixes: {len(action_mixes)} across "
            f"{len(results)} scenarios.",
            "- Every selected policy satisfies its scenario-specific manual-review",
            "  capacity constraint.",
            "",
            f"- Base `{base_result.scenario.display_name}`: review threshold "
            f"{base_result.selection.thresholds.review_threshold:.2f}, decline "
            f"threshold {base_result.selection.thresholds.decline_threshold:.2f},",
            f"  and action mix {_action_mix(base_result)}.",
            "",
        ]
    )
    lines.extend(_scenario_change_line(base_result, result) for result in results[1:])
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Every scenario selects and reports thresholds on the same small",
            "  validation labels. This describes assumption sensitivity, not",
            "  unbiased performance or production robustness.",
            "- Lower cost in one scenario is not directly comparable with lower",
            "  cost in another because the simulated cost scales differ.",
            "- Review is modeled as intercepting fraud at one fixed review cost;",
            "  queue timing, reviewer error, and delayed outcomes are not modeled.",
            "- The score source remains an engineering-only provisional XGBoost",
            "  challenger. Test evidence remains frozen and is not reused here.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: PolicySensitivityReport, output_path: Path) -> None:
    """Write the cost-sensitivity report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _scenario_markdown_row(result: PolicySensitivityResult) -> str:
    costs = result.cost_config
    thresholds = result.selection.thresholds
    operating = result.selection.operating_cost
    return (
        f"| {result.scenario.display_name} | {costs.fraud_loss_multiplier:.2f} | "
        f"{costs.manual_review_cost:.2f} | {costs.false_decline_cost:.2f} | "
        f"{costs.max_manual_review_rate:.2%} | "
        f"{thresholds.review_threshold:.2f} | "
        f"{thresholds.decline_threshold:.2f} | {operating.approvals:,} | "
        f"{operating.reviews:,} | {operating.declines:,} | "
        f"{operating.review_rate:.2%} | {operating.false_declines:,} | "
        f"{operating.fraud_amount_captured:,.2f} | "
        f"{operating.total_per_1000_transactions:,.2f} |"
    )


def _scenario_change_line(
    base: PolicySensitivityResult, result: PolicySensitivityResult
) -> str:
    base_thresholds = base.selection.thresholds
    thresholds = result.selection.thresholds
    threshold_summary = (
        "thresholds unchanged"
        if thresholds == base_thresholds
        else (
            f"thresholds changed to {thresholds.review_threshold:.2f}/"
            f"{thresholds.decline_threshold:.2f}"
        )
    )
    action_summary = (
        "action mix unchanged"
        if _action_counts(result) == _action_counts(base)
        else f"action mix changed to {_action_mix(result)}"
    )
    return f"- `{result.scenario.display_name}`: {threshold_summary}; {action_summary}."


def _action_counts(result: PolicySensitivityResult) -> tuple[int, int, int]:
    operating = result.selection.operating_cost
    return operating.approvals, operating.reviews, operating.declines


def _action_mix(result: PolicySensitivityResult) -> str:
    approvals, reviews, declines = _action_counts(result)
    return f"{approvals:,}/{reviews:,}/{declines:,}"
