"""Train-only Logistic Regression baseline using decision-time fields."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.data.schema import Transaction
from fraud_detection.data.splitter import MODEL_FEATURE_COLUMNS

LOGISTIC_SOURCE_COLUMNS: Final = (
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_DAYS",
)
NUMERIC_FEATURE_INDICES: Final = (0, 1, 2, 3)
CATEGORICAL_FEATURE_INDICES: Final = (4, 5)

if not set(LOGISTIC_SOURCE_COLUMNS).issubset(MODEL_FEATURE_COLUMNS):
    raise RuntimeError("logistic source columns violate the model feature contract")


@dataclass(frozen=True, slots=True)
class LogisticConfig:
    """Fixed, versioned parameters for the interpretable ML baseline."""

    regularization_c: float
    class_weight: str
    solver: str
    max_iterations: int
    random_state: int
    flag_threshold: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.regularization_c) or self.regularization_c <= 0:
            raise ValueError("regularization_c must be finite and positive")
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be 'balanced' for this baseline")
        if self.solver != "liblinear":
            raise ValueError("solver must be 'liblinear' for this baseline")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        if not math.isfinite(self.flag_threshold) or not (0 < self.flag_threshold < 1):
            raise ValueError("flag_threshold must be in (0, 1)")


def load_logistic_config(path: Path) -> LogisticConfig:
    """Load and type-check the fixed Logistic Regression configuration."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("logistic_baseline")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [logistic_baseline] table")

    try:
        return LogisticConfig(
            regularization_c=_require_number(section, "regularization_c"),
            class_weight=_require_string(section, "class_weight"),
            solver=_require_string(section, "solver"),
            max_iterations=_require_int(section, "max_iterations"),
            random_state=_require_int(section, "random_state"),
            flag_threshold=_require_number(section, "flag_threshold"),
        )
    except KeyError as error:
        raise ValueError(f"missing logistic setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid logistic configuration: {error}") from error


def fit_logistic_baseline(
    transactions: Sequence[Transaction], config: LogisticConfig
) -> Pipeline:
    """Fit preprocessing and Logistic Regression on one training split."""
    if not transactions:
        raise ValueError("training transactions must not be empty")
    labels = [transaction.tx_fraud for transaction in transactions]
    if set(labels) != {0, 1}:
        raise ValueError("training transactions must contain both fraud classes")

    pipeline = _build_pipeline(config)
    pipeline.fit(_feature_rows(transactions), labels)
    return pipeline


def predict_fraud_scores(
    pipeline: Pipeline, transactions: Sequence[Transaction]
) -> tuple[float, ...]:
    """Return positive-class probabilities for decision-time feature rows."""
    if not transactions:
        raise ValueError("scoring transactions must not be empty")
    probabilities = pipeline.predict_proba(_feature_rows(transactions))[:, 1]
    return tuple(float(probability) for probability in probabilities)


def _build_pipeline(config: LogisticConfig) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=(
            ("numeric", StandardScaler(), NUMERIC_FEATURE_INDICES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_INDICES,
            ),
        )
    )
    classifier = LogisticRegression(
        C=config.regularization_c,
        class_weight=config.class_weight,
        solver=config.solver,
        max_iter=config.max_iterations,
        random_state=config.random_state,
    )
    return Pipeline(
        steps=(
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        )
    )


def _feature_rows(
    transactions: Sequence[Transaction],
) -> tuple[tuple[float | str, ...], ...]:
    rows = []
    for transaction in transactions:
        hour_fraction = (
            transaction.tx_datetime.hour
            + (transaction.tx_datetime.minute / 60)
            + (transaction.tx_datetime.second / 3_600)
        ) / 24
        angle = 2 * math.pi * hour_fraction
        rows.append(
            (
                float(transaction.tx_amount),
                float(transaction.tx_time_days),
                math.sin(angle),
                math.cos(angle),
                f"customer_{transaction.customer_id}",
                f"terminal_{transaction.terminal_id}",
            )
        )
    return tuple(rows)


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


def _require_string(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value
