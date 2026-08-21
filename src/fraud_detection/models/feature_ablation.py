"""Validation-only feature-group ablation for the frozen XGBoost challenger."""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fraud_detection.data.eda import ProcessedSplits, load_processed_splits
from fraud_detection.decisioning.costs import BusinessCostConfig
from fraud_detection.features.matrix import (
    JoinedFeatureSplits,
    load_joined_feature_splits,
)
from fraud_detection.models.baseline_comparison import BaselineMetrics, evaluate_scores
from fraud_detection.models.xgboost_model import (
    XGBoostConfig,
    fit_xgboost_feature_variant,
    predict_xgboost_feature_variant_scores,
)


@dataclass(frozen=True, slots=True)
class FeatureVariant:
    """One explicit combination of the four XGBoost feature groups."""

    key: str
    display_name: str
    include_temporal_features: bool
    include_synthetic_ids: bool
    include_terminal_temporal_features: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError(
                "variant key must contain letters, numbers, or underscores"
            )
        if not self.display_name.strip():
            raise ValueError("variant display_name must not be empty")
        if type(self.include_temporal_features) is not bool:
            raise ValueError("include_temporal_features must be a boolean")
        if type(self.include_synthetic_ids) is not bool:
            raise ValueError("include_synthetic_ids must be a boolean")
        if type(self.include_terminal_temporal_features) is not bool:
            raise ValueError("include_terminal_temporal_features must be a boolean")

    @property
    def input_summary(self) -> str:
        """Return a concise, stable feature-group description."""
        groups = ["Base"]
        if self.include_temporal_features:
            groups.append("customer temporal")
        if self.include_terminal_temporal_features:
            groups.append("terminal temporal")
        if self.include_synthetic_ids:
            groups.append("synthetic IDs")
        return " + ".join(groups)


@dataclass(frozen=True, slots=True)
class FeatureAblationConfig:
    """Versioned validation-only variants and decision screen."""

    evaluation_split: str
    baseline_variant_key: str
    minimum_ap_improvement: float
    variants: tuple[FeatureVariant, ...]

    def __post_init__(self) -> None:
        if self.evaluation_split != "validation":
            raise ValueError("evaluation_split must be 'validation'")
        if (
            not math.isfinite(self.minimum_ap_improvement)
            or not 0 <= self.minimum_ap_improvement <= 1
        ):
            raise ValueError("minimum_ap_improvement must be within [0, 1]")
        if len(self.variants) < 2:
            raise ValueError("at least two feature variants are required")
        keys = tuple(variant.key for variant in self.variants)
        if len(set(keys)) != len(keys):
            raise ValueError("feature variant keys must be unique")
        feature_combinations = tuple(
            (
                variant.include_temporal_features,
                variant.include_synthetic_ids,
                variant.include_terminal_temporal_features,
            )
            for variant in self.variants
        )
        if len(set(feature_combinations)) != len(feature_combinations):
            raise ValueError("feature variants must use unique feature combinations")
        if self.baseline_variant_key not in keys:
            raise ValueError("baseline_variant_key must match one configured variant")
        baseline = self.variants[keys.index(self.baseline_variant_key)]
        if not (
            baseline.include_temporal_features
            and baseline.include_synthetic_ids
            and not baseline.include_terminal_temporal_features
        ):
            raise ValueError(
                "the frozen baseline must include customer temporal features and "
                "IDs without terminal temporal features"
            )


@dataclass(frozen=True, slots=True)
class FeatureAblationResult:
    """Validation metrics for one fixed feature combination."""

    variant: FeatureVariant
    metrics: BaselineMetrics


@dataclass(frozen=True, slots=True)
class FeatureDirectionDecision:
    """Versioned AP screen comparing the best simpler variant to the baseline."""

    candidate: FeatureAblationResult
    baseline: FeatureAblationResult
    ap_improvement: float | None
    revision_recommended: bool


@dataclass(frozen=True, slots=True)
class FeatureAblationReport:
    """Reproducible validation-only feature-group comparison."""

    train_rows: int
    train_frauds: int
    validation_rows: int
    validation_frauds: int
    xgboost_config: XGBoostConfig
    xgboost_config_hash: str
    ablation_config: FeatureAblationConfig
    results: tuple[FeatureAblationResult, ...]
    decision: FeatureDirectionDecision


def load_feature_ablation_config(path: Path) -> FeatureAblationConfig:
    """Load and type-check the versioned feature-ablation contract."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("xgboost_feature_ablation")
    if not isinstance(section, dict):
        raise ValueError(
            "configuration must contain an [xgboost_feature_ablation] table"
        )
    try:
        raw_variants = section["variants"]
        if not isinstance(raw_variants, list):
            raise ValueError("variants must be an array of tables")
        variants = tuple(_load_variant(raw_variant) for raw_variant in raw_variants)
        return FeatureAblationConfig(
            evaluation_split=_require_str(section, "evaluation_split"),
            baseline_variant_key=_require_str(section, "baseline_variant_key"),
            minimum_ap_improvement=_require_number(section, "minimum_ap_improvement"),
            variants=variants,
        )
    except KeyError as error:
        raise ValueError(
            f"missing XGBoost feature-ablation setting: {error.args[0]}"
        ) from error
    except ValueError as error:
        raise ValueError(f"invalid XGBoost feature-ablation config: {error}") from error


def evaluate_feature_ablation(
    input_directory: Path,
    feature_directory: Path,
    xgboost_config: XGBoostConfig,
    xgboost_config_hash: str,
    ablation_config: FeatureAblationConfig,
    cost_config: BusinessCostConfig,
) -> FeatureAblationReport:
    """Fit fixed variants on train and score the chronological validation split."""
    transaction_splits = load_processed_splits(input_directory)
    joined_splits = load_joined_feature_splits(transaction_splits, feature_directory)
    return evaluate_loaded_feature_ablation(
        transaction_splits,
        joined_splits,
        xgboost_config,
        xgboost_config_hash,
        ablation_config,
        cost_config,
    )


def evaluate_loaded_feature_ablation(
    transaction_splits: ProcessedSplits,
    joined_splits: JoinedFeatureSplits,
    xgboost_config: XGBoostConfig,
    xgboost_config_hash: str,
    ablation_config: FeatureAblationConfig,
    cost_config: BusinessCostConfig,
) -> FeatureAblationReport:
    """Evaluate only validation labels under the fixed feature contract."""
    if ablation_config.evaluation_split != "validation":
        raise ValueError("feature ablation must evaluate validation only")
    if len(xgboost_config_hash) != 64 or any(
        character not in "0123456789abcdef" for character in xgboost_config_hash
    ):
        raise ValueError("xgboost_config_hash must be a lowercase SHA-256 digest")

    results = []
    for variant in ablation_config.variants:
        pipeline = fit_xgboost_feature_variant(
            joined_splits.train,
            xgboost_config,
            include_temporal_features=variant.include_temporal_features,
            include_synthetic_ids=variant.include_synthetic_ids,
            include_terminal_temporal_features=(
                variant.include_terminal_temporal_features
            ),
        )
        scores = predict_xgboost_feature_variant_scores(
            pipeline,
            joined_splits.validation,
            include_temporal_features=variant.include_temporal_features,
            include_synthetic_ids=variant.include_synthetic_ids,
            include_terminal_temporal_features=(
                variant.include_terminal_temporal_features
            ),
        )
        results.append(
            FeatureAblationResult(
                variant=variant,
                metrics=evaluate_scores(
                    variant.display_name,
                    "validation",
                    transaction_splits.validation,
                    scores,
                    xgboost_config.flag_threshold,
                    cost_config,
                ),
            )
        )
    result_tuple = tuple(results)
    return FeatureAblationReport(
        train_rows=len(joined_splits.train),
        train_frauds=sum(row.transaction.tx_fraud for row in joined_splits.train),
        validation_rows=len(joined_splits.validation),
        validation_frauds=sum(
            row.transaction.tx_fraud for row in joined_splits.validation
        ),
        xgboost_config=xgboost_config,
        xgboost_config_hash=xgboost_config_hash,
        ablation_config=ablation_config,
        results=result_tuple,
        decision=_make_feature_decision(result_tuple, ablation_config),
    )


def render_markdown(report: FeatureAblationReport) -> str:
    """Render the deterministic validation-only feature comparison."""
    config = report.ablation_config
    model = report.xgboost_config
    lines = [
        "# Phase 3 XGBoost Feature Ablation",
        "",
        "Generated reproducibly by `fraud-analyze-xgboost-features`. Every variant",
        f"fits the same chronological training split ({report.train_rows:,} rows,",
        f"{report.train_frauds:,} fraud labels) and scores only validation",
        f"({report.validation_rows:,} rows, {report.validation_frauds:,} fraud labels).",
        "The one-time test split is not scored by this command.",
        "",
        "## Fixed Experiment Contract",
        "",
        f"- Frozen XGBoost config SHA-256: `{report.xgboost_config_hash}`.",
        f"- Trees {model.n_estimators}, depth {model.max_depth}, learning rate "
        f"{model.learning_rate:g}, random state {model.random_state}, and fixed "
        f"threshold {model.flag_threshold:.2f} for every variant.",
        "- Base features: transaction amount, elapsed day, and cyclical UTC hour.",
        "- Temporal features: one-hour and 24-hour customer counts, prior amount",
        "  mean and deviation, and seconds since the previous customer transaction.",
        "- Terminal temporal features mirror the customer history contract for",
        "  each terminal using strictly earlier transactions only.",
        "- Synthetic IDs: one-hot encoded customer and terminal identifiers.",
        "- Only feature inclusion changes; no hyperparameter, threshold, or",
        "  calibration search is performed.",
        "",
        "## Validation Comparison",
        "",
        "| Variant | Inputs | Rows | Frauds | AP | ROC-AUC | Flagged | Flag rate | "
        "TP | FP | FN | Precision | Recall | FPR | Cost per 1,000 | Within capacity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report.results:
        metrics = result.metrics
        lines.append(
            f"| {result.variant.display_name} | {result.variant.input_summary} | "
            f"{metrics.rows:,} | {metrics.frauds:,} | "
            f"{_format_rate(metrics.average_precision)} | "
            f"{_format_rate(metrics.roc_auc)} | {metrics.flagged:,} | "
            f"{_format_percent(metrics.flag_rate)} | {metrics.true_positives:,} | "
            f"{metrics.false_positives:,} | {metrics.false_negatives:,} | "
            f"{_format_percent(metrics.precision)} | "
            f"{_format_percent(metrics.recall)} | "
            f"{_format_percent(metrics.false_positive_rate)} | "
            f"{metrics.operating_cost.total_per_1000_transactions:,.2f} | "
            f"{'yes' if metrics.operating_cost.within_review_capacity else 'no'} |"
        )
    decision = report.decision
    candidate_ap = decision.candidate.metrics.average_precision
    baseline_ap = decision.baseline.metrics.average_precision
    lines.extend(
        [
            "",
            "## Feature Direction Decision",
            "",
            f"The versioned screen requires at least +{config.minimum_ap_improvement:.4f}",
            "absolute validation AP over the frozen full baseline before recommending",
            "a bounded feature revision.",
            "",
            *(
                (
                    "**Decision: a bounded feature revision is justified. Carry "
                    f"`{decision.candidate.variant.display_name}` forward as a new "
                    "validation-selected challenger; do not replace the frozen "
                    "baseline or score test.**",
                )
                if decision.revision_recommended
                else (
                    "**Decision: this ablation does not justify a bounded feature "
                    "revision. Retain the frozen full baseline only as an unpromoted "
                    "provisional challenger and do not score test.**",
                )
            ),
            "",
            f"- Best simpler variant: `{decision.candidate.variant.display_name}` "
            f"with AP {_format_rate(candidate_ap)}.",
            f"- Frozen full baseline AP: {_format_rate(baseline_ap)}.",
            f"- Absolute AP difference: {_format_signed(decision.ap_improvement)}.",
            "- This validation-selected comparison has only "
            f"{report.validation_frauds:,} fraud labels and is not a significance",
            "  test or model-promotion decision.",
            "",
            "## Interpretation Limits",
            "",
            "- Feature groups are compared under one fixed model configuration; a",
            "  result may reflect interactions with that configuration.",
            "- Synthetic customer and terminal IDs are dataset-specific categories,",
            "  not portable behavioral risk features.",
            "- The small and uneven validation fraud support makes differences",
            "  unstable. Test evidence remains frozen and is not reused here.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: FeatureAblationReport, output_path: Path) -> None:
    """Write the feature-ablation report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _make_feature_decision(
    results: tuple[FeatureAblationResult, ...], config: FeatureAblationConfig
) -> FeatureDirectionDecision:
    baseline = next(
        result
        for result in results
        if result.variant.key == config.baseline_variant_key
    )
    candidates = tuple(result for result in results if result is not baseline)
    candidate = max(
        candidates,
        key=lambda result: (
            result.metrics.average_precision
            if result.metrics.average_precision is not None
            else float("-inf")
        ),
    )
    if (
        candidate.metrics.average_precision is None
        or baseline.metrics.average_precision is None
    ):
        improvement = None
        revision_recommended = False
    else:
        improvement = (
            candidate.metrics.average_precision - baseline.metrics.average_precision
        )
        revision_recommended = improvement >= config.minimum_ap_improvement
    return FeatureDirectionDecision(
        candidate=candidate,
        baseline=baseline,
        ap_improvement=improvement,
        revision_recommended=revision_recommended,
    )


def _load_variant(value: object) -> FeatureVariant:
    if not isinstance(value, dict):
        raise ValueError("each variant must be a table")
    return FeatureVariant(
        key=_require_str(value, "key"),
        display_name=_require_str(value, "display_name"),
        include_temporal_features=_require_bool(value, "include_temporal_features"),
        include_synthetic_ids=_require_bool(value, "include_synthetic_ids"),
        include_terminal_temporal_features=_optional_bool(
            value, "include_terminal_temporal_features", default=False
        ),
    )


def _require_str(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_number(section: Mapping[str, object], key: str) -> float:
    value = section[key]
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be a number")
    return float(value)


def _require_bool(section: Mapping[str, object], key: str) -> bool:
    value = section[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_bool(section: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = section.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"
