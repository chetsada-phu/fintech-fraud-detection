"""Validation-only evaluation for the provisional three-way decision policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fraud_detection.data.eda import load_processed_splits
from fraud_detection.decisioning.costs import BusinessCostConfig
from fraud_detection.decisioning.policy import (
    RISK_SCORE_DECLINE,
    RISK_SCORE_REVIEW,
    DecisionPolicyConfig,
    PolicySelection,
    select_policy_thresholds,
)
from fraud_detection.decisioning.reasons import (
    HIGH_AMOUNT_VS_CUSTOMER_BASELINE,
    HIGH_TRANSACTION_AMOUNT,
    LIMITED_CUSTOMER_HISTORY,
    UNUSUAL_TRANSACTION_VELOCITY,
    DecisionExplanationSummary,
    DecisionReasonConfig,
    explain_policy_decisions,
)
from fraud_detection.features.matrix import load_joined_feature_splits
from fraud_detection.models.xgboost_model import (
    XGBoostConfig,
    fit_xgboost_baseline,
    predict_xgboost_scores,
)


@dataclass(frozen=True, slots=True)
class DecisionPolicyReport:
    """Reproducible validation-only policy-selection result."""

    train_rows: int
    train_frauds: int
    validation_rows: int
    validation_frauds: int
    xgboost_config_hash: str
    policy_config: DecisionPolicyConfig
    cost_config: BusinessCostConfig
    selection: PolicySelection
    reason_config: DecisionReasonConfig
    explanations: DecisionExplanationSummary


def evaluate_decision_policy(
    input_directory: Path,
    feature_directory: Path,
    xgboost_config: XGBoostConfig,
    xgboost_config_hash: str,
    policy_config: DecisionPolicyConfig,
    cost_config: BusinessCostConfig,
    reason_config: DecisionReasonConfig,
) -> DecisionPolicyReport:
    """Fit on train and select three-way thresholds on validation only."""
    if policy_config.selection_split != "validation":
        raise ValueError("decision-policy evaluation must use validation only")
    if len(xgboost_config_hash) != 64 or any(
        character not in "0123456789abcdef" for character in xgboost_config_hash
    ):
        raise ValueError("xgboost_config_hash must be a lowercase SHA-256 digest")

    transaction_splits = load_processed_splits(input_directory)
    joined_splits = load_joined_feature_splits(transaction_splits, feature_directory)
    pipeline = fit_xgboost_baseline(joined_splits.train, xgboost_config)
    validation_scores = predict_xgboost_scores(pipeline, joined_splits.validation)
    selection = select_policy_thresholds(
        transaction_splits.validation,
        validation_scores,
        policy_config,
        cost_config,
        split_name="validation",
    )
    explanations = explain_policy_decisions(
        joined_splits.validation,
        selection.decisions,
        reason_config,
    )
    return DecisionPolicyReport(
        train_rows=len(joined_splits.train),
        train_frauds=sum(row.transaction.tx_fraud for row in joined_splits.train),
        validation_rows=len(joined_splits.validation),
        validation_frauds=sum(
            row.transaction.tx_fraud for row in joined_splits.validation
        ),
        xgboost_config_hash=xgboost_config_hash,
        policy_config=policy_config,
        cost_config=cost_config,
        selection=selection,
        reason_config=reason_config,
        explanations=explanations,
    )


def render_markdown(report: DecisionPolicyReport) -> str:
    """Render the deterministic provisional decision-policy report."""
    config = report.policy_config
    costs = report.cost_config
    selection = report.selection
    thresholds = selection.thresholds
    result = selection.operating_cost
    reason_config = report.reason_config
    explanations = report.explanations
    lines = [
        "# Phase 4 Provisional Decision Policy",
        "",
        "Generated reproducibly by `fraud-select-decision-policy`. The existing",
        f"XGBoost pipeline fits training ({report.train_rows:,} rows, "
        f"{report.train_frauds:,} fraud labels) and scores only chronological",
        f"validation ({report.validation_rows:,} rows, "
        f"{report.validation_frauds:,} fraud labels) for threshold selection.",
        "The one-time test split is not scored by this command.",
        "",
        "## Score-source Status",
        "",
        f"- Score source: **{config.score_source_label}**.",
        "- XGBoost remains unpromoted. These scores support decision-policy",
        "  engineering only and are not a model-performance claim.",
        f"- Frozen XGBoost config SHA-256: `{report.xgboost_config_hash}`.",
        "",
        "## Versioned Selection Contract",
        "",
        f"- Selection split: `{config.selection_split}` only.",
        f"- Fixed threshold grid step: {config.threshold_grid_step} across the",
        "  inclusive score range from 0.00 to 1.00.",
        f"- Candidate thresholds: {selection.threshold_candidate_count:,}; ordered",
        f"  review/decline pairs evaluated: {selection.evaluated_candidate_pairs:,}.",
        "- Primary objective: minimum simulated total operating cost while the",
        f"  validation review rate is at most {costs.max_manual_review_rate:.2%}.",
        "- Cost ties prefer fewer false declines, then fewer reviews, then fewer",
        "  declines, and finally higher thresholds.",
        "- Boundary semantics: score below the review threshold is `approve`;",
        "  score at or above review and below decline is `review`; score at or",
        "  above decline is `decline`. If thresholds are equal, decline takes",
        "  precedence and the review band is empty.",
        "",
        "## Selected Validation Policy",
        "",
        f"- Review threshold: **{thresholds.review_threshold:.2f}**.",
        f"- Decline threshold: **{thresholds.decline_threshold:.2f}**.",
        "",
        "| Decision | Rows | Rate | Score-band fallback |",
        "| --- | ---: | ---: | --- |",
        f"| Approve | {result.approvals:,} | "
        f"{result.approvals / report.validation_rows:.2%} | none |",
        f"| Review | {result.reviews:,} | {result.review_rate:.2%} | "
        f"`{RISK_SCORE_REVIEW}` |",
        f"| Decline | {result.declines:,} | "
        f"{result.declines / report.validation_rows:.2%} | "
        f"`{RISK_SCORE_DECLINE}` |",
        "",
        f"Manual-review capacity: **{'satisfied' if result.within_review_capacity else 'not satisfied'}** "
        f"({result.review_rate:.2%} observed versus {costs.max_manual_review_rate:.2%} maximum).",
        "",
        "## Deterministic Decision Reasons",
        "",
        "Feature reasons use only the current transaction and aligned strictly",
        "past customer features. Fraud labels, fraud scenarios, and future rows",
        "are not inputs. These reasons describe configured conditions, not model",
        "causality or feature attribution.",
        "",
        (
            "- Priority order: "
            f"{', '.join(f'`{code}`' for code in reason_config.priority_order)}."
        ),
        (
            f"- At most {reason_config.max_feature_reason_codes} feature reasons "
            "are emitted per risky decision."
        ),
        "- A score-band review or decline code is retained only when no feature",
        "  condition matches.",
        (
            f"- Risky decisions: {explanations.risky_decisions:,}; "
            "feature-explained: "
            f"{explanations.feature_explained_decisions:,}; score-band fallbacks: "
            f"{explanations.fallback_decisions:,}."
        ),
        "",
        "| Reason code | Exact configured condition | Activations |",
        "| --- | --- | ---: |",
    ]
    lines.extend(
        f"| `{reason_code}` | {_reason_description(reason_code, reason_config)} | "
        f"{count:,} |"
        for reason_code, count in explanations.reason_counts
    )
    lines.extend(
        [
            "",
            "## Simulated Validation Cost",
            "",
            "These values use portfolio assumptions, not bank economics.",
            "",
            f"- Missed-fraud loss: {result.missed_fraud_loss:,.2f}.",
            f"- Manual-review cost: {result.manual_review_cost:,.2f}.",
            f"- False-decline cost: {result.false_decline_cost:,.2f} across",
            f"  {result.false_declines:,} legitimate declines",
            f"  ({result.false_decline_rate:.2%} of legitimate validation rows).",
            f"- Total: {result.total:,.2f}; cost per 1,000 transactions:",
            f"  {result.total_per_1000_transactions:,.2f}.",
            "- Fraud transaction amount intercepted by review or decline:",
            f"  {result.fraud_amount_captured:,.2f}.",
            "",
            "## Interpretation Limits",
            "",
            "- Thresholds were selected against the same small validation labels used",
            "  to report this cost. They are provisional engineering settings, not an",
            "  unbiased performance estimate or a production policy.",
            "- The simulated calculation assumes every reviewed fraud is intercepted",
            "  and charges one review cost. Real review outcomes and queue timing are",
            "  not modeled.",
            "- Feature-derived reasons identify configured transaction patterns; they",
            "  do not prove why the model produced its score or establish causality.",
            "- The score source remains the versioned",
            f"  `{config.score_source_label}`. The frozen test report is not reused to",
            "  select or revise these thresholds.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: DecisionPolicyReport, output_path: Path) -> None:
    """Write the decision-policy report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _reason_description(reason_code: str, config: DecisionReasonConfig) -> str:
    descriptions = {
        HIGH_AMOUNT_VS_CUSTOMER_BASELINE: (
            "amount at least "
            f"{config.customer_amount_ratio_threshold:.2f}x prior customer mean"
        ),
        HIGH_TRANSACTION_AMOUNT: (
            f"amount strictly above {config.high_transaction_amount_threshold:.2f}"
        ),
        UNUSUAL_TRANSACTION_VELOCITY: (
            "at least "
            f"{config.customer_short_window_count_threshold} prior customer "
            "transactions in the short window"
        ),
        LIMITED_CUSTOMER_HISTORY: "prior customer-history fields are all missing",
        RISK_SCORE_REVIEW: "review score band; used only as fallback",
        RISK_SCORE_DECLINE: "decline score band; used only as fallback",
    }
    return descriptions[reason_code]
