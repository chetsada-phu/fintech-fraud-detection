"""Fixed train-only XGBoost baseline with temporal customer features."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from fraud_detection.features.matrix import JoinedFeatureRow
from fraud_detection.models.model_input import ProvisionalModelInput

BASE_NUMERIC_FEATURE_COUNT: Final = 4
CUSTOMER_TEMPORAL_NUMERIC_FEATURE_COUNT: Final = 5
TERMINAL_TEMPORAL_NUMERIC_FEATURE_COUNT: Final = 5
SYNTHETIC_ID_FEATURE_COUNT: Final = 2


@dataclass(frozen=True, slots=True)
class XGBoostConfig:
    """Frozen parameters for the first leakage-safe main-model baseline."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    column_sample_by_tree: float
    regularization_lambda: float
    regularization_alpha: float
    use_training_class_ratio: bool
    random_state: int
    n_jobs: int
    flag_threshold: float

    def __post_init__(self) -> None:
        if type(self.n_estimators) is not int or self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        positive_values = (
            ("learning_rate", self.learning_rate),
            ("min_child_weight", self.min_child_weight),
            ("regularization_lambda", self.regularization_lambda),
        )
        for name, value in positive_values:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.regularization_alpha)
            or self.regularization_alpha < 0
        ):
            raise ValueError("regularization_alpha must be finite and non-negative")
        for name, value in (
            ("subsample", self.subsample),
            ("column_sample_by_tree", self.column_sample_by_tree),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if type(self.use_training_class_ratio) is not bool:
            raise ValueError("use_training_class_ratio must be a boolean")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        if type(self.n_jobs) is not int or self.n_jobs <= 0:
            raise ValueError("n_jobs must be a positive integer")
        if not math.isfinite(self.flag_threshold) or not (0 < self.flag_threshold < 1):
            raise ValueError("flag_threshold must be in (0, 1)")


def load_xgboost_config(path: Path) -> XGBoostConfig:
    """Load and type-check the frozen XGBoost configuration."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("xgboost_baseline")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain an [xgboost_baseline] table")
    try:
        return XGBoostConfig(
            n_estimators=_require_int(section, "n_estimators"),
            max_depth=_require_int(section, "max_depth"),
            learning_rate=_require_number(section, "learning_rate"),
            min_child_weight=_require_number(section, "min_child_weight"),
            subsample=_require_number(section, "subsample"),
            column_sample_by_tree=_require_number(section, "column_sample_by_tree"),
            regularization_lambda=_require_number(section, "regularization_lambda"),
            regularization_alpha=_require_number(section, "regularization_alpha"),
            use_training_class_ratio=_require_bool(section, "use_training_class_ratio"),
            random_state=_require_int(section, "random_state"),
            n_jobs=_require_int(section, "n_jobs"),
            flag_threshold=_require_number(section, "flag_threshold"),
        )
    except KeyError as error:
        raise ValueError(f"missing XGBoost setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid XGBoost configuration: {error}") from error


def fit_xgboost_baseline(
    rows: Sequence[JoinedFeatureRow], config: XGBoostConfig
) -> Pipeline:
    """Fit preprocessing and XGBoost using training rows only."""
    return fit_xgboost_feature_variant(
        rows,
        config,
        include_temporal_features=True,
        include_synthetic_ids=True,
        include_terminal_temporal_features=False,
    )


def fit_xgboost_feature_variant(
    rows: Sequence[JoinedFeatureRow],
    config: XGBoostConfig,
    *,
    include_temporal_features: bool,
    include_synthetic_ids: bool,
    include_terminal_temporal_features: bool = False,
) -> Pipeline:
    """Fit one fixed-parameter feature variant using training rows only."""
    if not rows:
        raise ValueError("training rows must not be empty")
    _validate_feature_flags(
        include_temporal_features,
        include_synthetic_ids,
        include_terminal_temporal_features,
    )
    labels = [row.transaction.tx_fraud for row in rows]
    frauds = sum(labels)
    legitimate = len(labels) - frauds
    if frauds == 0 or legitimate == 0:
        raise ValueError("training rows must contain both fraud classes")
    scale_pos_weight = legitimate / frauds if config.use_training_class_ratio else 1.0
    pipeline = _build_pipeline(
        config,
        scale_pos_weight,
        include_temporal_features=include_temporal_features,
        include_synthetic_ids=include_synthetic_ids,
        include_terminal_temporal_features=include_terminal_temporal_features,
    )
    pipeline.fit(
        _feature_rows(
            rows,
            include_temporal_features=include_temporal_features,
            include_synthetic_ids=include_synthetic_ids,
            include_terminal_temporal_features=include_terminal_temporal_features,
        ),
        labels,
    )
    return pipeline


def predict_xgboost_scores(
    pipeline: Pipeline, rows: Sequence[JoinedFeatureRow]
) -> tuple[float, ...]:
    """Return positive-class probabilities from joined decision-time rows."""
    return predict_xgboost_feature_variant_scores(
        pipeline,
        rows,
        include_temporal_features=True,
        include_synthetic_ids=True,
        include_terminal_temporal_features=False,
    )


def predict_xgboost_feature_variant_scores(
    pipeline: Pipeline,
    rows: Sequence[JoinedFeatureRow],
    *,
    include_temporal_features: bool,
    include_synthetic_ids: bool,
    include_terminal_temporal_features: bool = False,
) -> tuple[float, ...]:
    """Score one feature variant with the same feature contract used for fitting."""
    if not rows:
        raise ValueError("scoring rows must not be empty")
    _validate_feature_flags(
        include_temporal_features,
        include_synthetic_ids,
        include_terminal_temporal_features,
    )
    probabilities = pipeline.predict_proba(
        _feature_rows(
            rows,
            include_temporal_features=include_temporal_features,
            include_synthetic_ids=include_synthetic_ids,
            include_terminal_temporal_features=include_terminal_temporal_features,
        )
    )[:, 1]
    return tuple(float(probability) for probability in probabilities)


def _build_pipeline(
    config: XGBoostConfig,
    scale_pos_weight: float,
    *,
    include_temporal_features: bool,
    include_synthetic_ids: bool,
    include_terminal_temporal_features: bool,
) -> Pipeline:
    numeric_feature_count = (
        BASE_NUMERIC_FEATURE_COUNT
        + (CUSTOMER_TEMPORAL_NUMERIC_FEATURE_COUNT if include_temporal_features else 0)
        + (
            TERMINAL_TEMPORAL_NUMERIC_FEATURE_COUNT
            if include_terminal_temporal_features
            else 0
        )
    )
    transformers: list[tuple[str, object, tuple[int, ...]]] = [
        ("numeric", "passthrough", tuple(range(numeric_feature_count)))
    ]
    if include_synthetic_ids:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                tuple(
                    range(
                        numeric_feature_count,
                        numeric_feature_count + SYNTHETIC_ID_FEATURE_COUNT,
                    )
                ),
            )
        )
    preprocessing = ColumnTransformer(transformers=transformers)
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.column_sample_by_tree,
        reg_lambda=config.regularization_lambda,
        reg_alpha=config.regularization_alpha,
        scale_pos_weight=scale_pos_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    return Pipeline(
        steps=(
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        )
    )


def _feature_rows(
    rows: Sequence[JoinedFeatureRow],
    *,
    include_temporal_features: bool,
    include_synthetic_ids: bool,
    include_terminal_temporal_features: bool,
) -> tuple[tuple[float | str, ...], ...]:
    if (
        include_temporal_features
        and include_synthetic_ids
        and not include_terminal_temporal_features
    ):
        return tuple(
            ProvisionalModelInput.from_joined_feature_row(row).to_feature_values()
            for row in rows
        )
    feature_rows = []
    for row in rows:
        transaction = row.transaction
        temporal = row.temporal
        hour_fraction = (
            transaction.tx_datetime.hour
            + (transaction.tx_datetime.minute / 60)
            + (transaction.tx_datetime.second / 3_600)
        ) / 24
        angle = 2 * math.pi * hour_fraction
        values: list[float | str] = [
            float(transaction.tx_amount),
            float(transaction.tx_time_days),
            math.sin(angle),
            math.cos(angle),
        ]
        if include_temporal_features:
            values.extend(
                (
                    float(temporal.customer_tx_count_short_window),
                    float(temporal.customer_tx_count_long_window),
                    _optional_float(temporal.customer_amount_mean_prior),
                    _optional_float(temporal.customer_amount_deviation_from_mean_prior),
                    _optional_float(temporal.customer_seconds_since_previous),
                )
            )
        if include_terminal_temporal_features:
            values.extend(
                (
                    float(temporal.terminal_tx_count_short_window),
                    float(temporal.terminal_tx_count_long_window),
                    _optional_float(temporal.terminal_amount_mean_prior),
                    _optional_float(temporal.terminal_amount_deviation_from_mean_prior),
                    _optional_float(temporal.terminal_seconds_since_previous),
                )
            )
        if include_synthetic_ids:
            values.extend(
                (
                    f"customer_{transaction.customer_id}",
                    f"terminal_{transaction.terminal_id}",
                )
            )
        feature_rows.append(tuple(values))
    return tuple(feature_rows)


def _validate_feature_flags(
    include_temporal_features: bool,
    include_synthetic_ids: bool,
    include_terminal_temporal_features: bool,
) -> None:
    if type(include_temporal_features) is not bool:
        raise ValueError("include_temporal_features must be a boolean")
    if type(include_synthetic_ids) is not bool:
        raise ValueError("include_synthetic_ids must be a boolean")
    if type(include_terminal_temporal_features) is not bool:
        raise ValueError("include_terminal_temporal_features must be a boolean")


def _optional_float(value: object | None) -> float:
    if value is None:
        return math.nan
    return float(value)


def _require_number(section: Mapping[str, object], key: str) -> float:
    value = section[key]
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be a number")
    return float(value)


def _require_int(section: Mapping[str, object], key: str) -> int:
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _require_bool(section: Mapping[str, object], key: str) -> bool:
    value = section[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value
