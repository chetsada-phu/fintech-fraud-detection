"""Deterministic label-free input drift analysis using PSI."""

from __future__ import annotations

import bisect
import csv
import hashlib
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

from fraud_detection.data.schema import CSV_COLUMNS
from fraud_detection.data.splitter import SPLIT_FILENAMES
from fraud_detection.features.matrix import load_temporal_feature_csv
from fraud_detection.features.temporal import FEATURE_FILENAMES
from fraud_detection.models.model_input import ProvisionalModelInput

NUMERIC_INPUT_FEATURES: Final = (
    "tx_amount",
    "tx_time_days",
    "customer_tx_count_short_window",
    "customer_tx_count_long_window",
    "customer_amount_mean_prior",
    "customer_amount_deviation_from_mean_prior",
    "customer_seconds_since_previous",
)
ALLOWED_SPLITS: Final = ("train", "validation")
MISSING_BIN_LABEL: Final = "Missing"
_CONFIG_KEYS: Final = {
    "comparison_split",
    "features",
    "high_psi_threshold",
    "moderate_psi_threshold",
    "monitor_version",
    "quantile_bin_count",
    "reference_split",
    "smoothing_epsilon",
}


@dataclass(frozen=True, slots=True)
class InputDriftConfig:
    """Versioned contract for one descriptive PSI comparison."""

    monitor_version: str
    reference_split: str
    comparison_split: str
    quantile_bin_count: int
    smoothing_epsilon: float
    moderate_psi_threshold: float
    high_psi_threshold: float
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.monitor_version.strip():
            raise ValueError("monitor_version must not be empty")
        if self.reference_split != "train":
            raise ValueError("reference_split must be 'train'")
        if self.comparison_split != "validation":
            raise ValueError("comparison_split must be 'validation'")
        if not 2 <= self.quantile_bin_count <= 20:
            raise ValueError("quantile_bin_count must be between 2 and 20")
        if not math.isfinite(self.smoothing_epsilon) or not (
            0 < self.smoothing_epsilon < 0.01
        ):
            raise ValueError("smoothing_epsilon must be between 0 and 0.01")
        thresholds = (self.moderate_psi_threshold, self.high_psi_threshold)
        if any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("PSI thresholds must be finite and positive")
        if self.moderate_psi_threshold >= self.high_psi_threshold:
            raise ValueError("moderate PSI threshold must be below high threshold")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be non-empty and unique")
        unknown = set(self.features).difference(NUMERIC_INPUT_FEATURES)
        if unknown:
            raise ValueError(f"unsupported numeric input features: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class DriftBin:
    """Reference and comparison shares for one fixed bin."""

    label: str
    reference_rows: int
    comparison_rows: int
    reference_share: float
    comparison_share: float
    psi_contribution: float


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """PSI result for one numeric input feature."""

    feature: str
    boundaries: tuple[float, ...]
    bins: tuple[DriftBin, ...]
    psi: float
    level: str


@dataclass(frozen=True, slots=True)
class InputDriftReport:
    """One label-free chronological input comparison."""

    config: InputDriftConfig
    reference_rows: int
    comparison_rows: int
    reference_sha256: str
    comparison_sha256: str
    reference_features_sha256: str
    comparison_features_sha256: str
    features: tuple[FeatureDrift, ...]


def load_input_drift_config(path: Path) -> InputDriftConfig:
    """Load and strictly validate the input drift TOML contract."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    if set(document) != {"input_drift"}:
        raise ValueError("configuration must contain only [input_drift]")
    section = document["input_drift"]
    if not isinstance(section, Mapping) or set(section) != _CONFIG_KEYS:
        raise ValueError("input_drift fields do not match the contract")
    raw_features = section["features"]
    if not isinstance(raw_features, list) or not all(
        isinstance(feature, str) for feature in raw_features
    ):
        raise ValueError("features must be an array of strings")
    return InputDriftConfig(
        monitor_version=_require_string(section, "monitor_version"),
        reference_split=_require_string(section, "reference_split"),
        comparison_split=_require_string(section, "comparison_split"),
        quantile_bin_count=_require_integer(section, "quantile_bin_count"),
        smoothing_epsilon=_require_float(section, "smoothing_epsilon"),
        moderate_psi_threshold=_require_float(section, "moderate_psi_threshold"),
        high_psi_threshold=_require_float(section, "high_psi_threshold"),
        features=tuple(raw_features),
    )


def load_label_free_inputs(
    transaction_path: Path, feature_path: Path
) -> tuple[ProvisionalModelInput, ...]:
    """Load one split while ignoring every post-event label value."""
    temporal_rows = load_temporal_feature_csv(feature_path)
    with transaction_path.open(encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"{transaction_path}: columns must match the transaction contract"
            )
        rows = tuple(reader)
    if not rows or len(rows) != len(temporal_rows):
        raise ValueError("transaction and temporal feature row counts must match")

    inputs = []
    previous_datetime: datetime | None = None
    seen_ids: set[int] = set()
    for row, temporal in zip(rows, temporal_rows, strict=True):
        transaction_id = int(row["TRANSACTION_ID"])
        tx_datetime = datetime.fromisoformat(row["TX_DATETIME"])
        if transaction_id in seen_ids:
            raise ValueError("transaction IDs must be unique within each split")
        seen_ids.add(transaction_id)
        if transaction_id != temporal.transaction_id:
            raise ValueError("transaction and temporal feature IDs must align")
        if previous_datetime is not None and tx_datetime < previous_datetime:
            raise ValueError("transactions must remain chronologically ordered")
        previous_datetime = tx_datetime
        inputs.append(
            ProvisionalModelInput(
                transaction_id=transaction_id,
                tx_amount=Decimal(row["TX_AMOUNT"]),
                tx_time_days=int(row["TX_TIME_DAYS"]),
                tx_datetime=tx_datetime,
                customer_tx_count_short_window=(
                    temporal.customer_tx_count_short_window
                ),
                customer_tx_count_long_window=temporal.customer_tx_count_long_window,
                customer_amount_mean_prior=temporal.customer_amount_mean_prior,
                customer_amount_deviation_from_mean_prior=(
                    temporal.customer_amount_deviation_from_mean_prior
                ),
                customer_seconds_since_previous=(
                    temporal.customer_seconds_since_previous
                ),
                customer_id=int(row["CUSTOMER_ID"]),
                terminal_id=int(row["TERMINAL_ID"]),
            )
        )
    return tuple(inputs)


def analyze_input_drift(
    reference: Sequence[ProvisionalModelInput],
    comparison: Sequence[ProvisionalModelInput],
    config: InputDriftConfig,
) -> tuple[FeatureDrift, ...]:
    """Compare label-free inputs with fixed reference-derived PSI bins."""
    if not reference or not comparison:
        raise ValueError("reference and comparison inputs must not be empty")
    if reference[-1].tx_datetime >= comparison[0].tx_datetime:
        raise ValueError("reference must end before comparison begins")
    return tuple(
        _analyze_feature(reference, comparison, feature, config)
        for feature in config.features
    )


def build_input_drift_report(
    config: InputDriftConfig,
    input_directory: Path,
    feature_directory: Path,
) -> InputDriftReport:
    """Load only configured train and validation sources and build the report."""
    reference_transaction_path = (
        input_directory / SPLIT_FILENAMES[config.reference_split]
    )
    comparison_transaction_path = (
        input_directory / SPLIT_FILENAMES[config.comparison_split]
    )
    reference_feature_path = (
        feature_directory / FEATURE_FILENAMES[config.reference_split]
    )
    comparison_feature_path = (
        feature_directory / FEATURE_FILENAMES[config.comparison_split]
    )
    reference = load_label_free_inputs(
        reference_transaction_path, reference_feature_path
    )
    comparison = load_label_free_inputs(
        comparison_transaction_path, comparison_feature_path
    )
    return InputDriftReport(
        config=config,
        reference_rows=len(reference),
        comparison_rows=len(comparison),
        reference_sha256=_sha256(reference_transaction_path),
        comparison_sha256=_sha256(comparison_transaction_path),
        reference_features_sha256=_sha256(reference_feature_path),
        comparison_features_sha256=_sha256(comparison_feature_path),
        features=analyze_input_drift(reference, comparison, config),
    )


def render_markdown(report: InputDriftReport) -> str:
    """Render a stable descriptive input drift report."""
    config = report.config
    lines = [
        "# Label-free input drift report",
        "",
        "Generated by `fraud-monitor-input-drift`. The report compares",
        f"`{config.reference_split}` with the later `{config.comparison_split}`",
        "split. It does not read fraud labels, score a model, or trigger alerts.",
        "",
        "## Contract",
        "",
        f"- Monitor version: `{config.monitor_version}`.",
        f"- Reference rows: {report.reference_rows:,}.",
        f"- Comparison rows: {report.comparison_rows:,}.",
        f"- Requested reference quantile bins: {config.quantile_bin_count}.",
        f"- Zero-count smoothing epsilon: {config.smoothing_epsilon:.6f}.",
        f"- Descriptive PSI levels: moderate at {config.moderate_psi_threshold:.2f}",
        f"  and high at {config.high_psi_threshold:.2f}.",
        "- Repeated reference quantiles are collapsed before counting bins.",
        "- Missing values use a separate bin.",
        "",
        "## Source identity",
        "",
        f"- Reference transactions SHA-256: `{report.reference_sha256}`.",
        f"- Comparison transactions SHA-256: `{report.comparison_sha256}`.",
        f"- Reference temporal features SHA-256: `{report.reference_features_sha256}`.",
        f"- Comparison temporal features SHA-256: `{report.comparison_features_sha256}`.",
        "",
        "## Summary",
        "",
        "| Input | PSI | Descriptive level | Fixed numeric bins |",
        "| --- | ---: | --- | ---: |",
    ]
    lines.extend(
        f"| `{feature.feature}` | {feature.psi:.6f} | {feature.level} | "
        f"{len(feature.boundaries) + 1} |"
        for feature in report.features
    )
    for feature in report.features:
        lines.extend(
            [
                "",
                f"## `{feature.feature}`",
                "",
                "| Bin | Reference rows | Reference share | Comparison rows | "
                "Comparison share | PSI contribution |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {bin_.label} | {bin_.reference_rows:,} | "
            f"{bin_.reference_share:.6f} | {bin_.comparison_rows:,} | "
            f"{bin_.comparison_share:.6f} | {bin_.psi_contribution:.6f} |"
            for bin_ in feature.bins
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "PSI describes distribution change in this small synthetic sample.",
            "The configured levels are project conventions, not calibrated alert",
            "thresholds. A high value does not identify a cause, prove model decay,",
            "or justify retraining. Performance monitoring requires delayed labels",
            "and is outside this label-free example.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: InputDriftReport, output_path: Path) -> None:
    """Write the report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _analyze_feature(
    reference: Sequence[ProvisionalModelInput],
    comparison: Sequence[ProvisionalModelInput],
    feature: str,
    config: InputDriftConfig,
) -> FeatureDrift:
    reference_values = tuple(_feature_value(row, feature) for row in reference)
    comparison_values = tuple(_feature_value(row, feature) for row in comparison)
    boundaries = _reference_quantile_boundaries(
        reference_values, config.quantile_bin_count
    )
    reference_counts = _bin_counts(reference_values, boundaries)
    comparison_counts = _bin_counts(comparison_values, boundaries)
    reference_shares = _smoothed_shares(reference_counts, config.smoothing_epsilon)
    comparison_shares = _smoothed_shares(comparison_counts, config.smoothing_epsilon)
    contributions = tuple(
        (current - expected) * math.log(current / expected)
        for expected, current in zip(reference_shares, comparison_shares, strict=True)
    )
    psi = sum(contributions)
    labels = (*_numeric_bin_labels(boundaries), MISSING_BIN_LABEL)
    level = (
        "high"
        if psi >= config.high_psi_threshold
        else ("moderate" if psi >= config.moderate_psi_threshold else "low")
    )
    return FeatureDrift(
        feature=feature,
        boundaries=boundaries,
        bins=tuple(
            DriftBin(
                label, expected_rows, current_rows, expected, current, contribution
            )
            for label, expected_rows, current_rows, expected, current, contribution in zip(
                labels,
                reference_counts,
                comparison_counts,
                reference_shares,
                comparison_shares,
                contributions,
                strict=True,
            )
        ),
        psi=psi,
        level=level,
    )


def _feature_value(row: ProvisionalModelInput, feature: str) -> float | None:
    value = getattr(row, feature)
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{feature} must be finite when present")
    return parsed


def _reference_quantile_boundaries(
    values: Sequence[float | None], bin_count: int
) -> tuple[float, ...]:
    observed = sorted(value for value in values if value is not None)
    if not observed:
        raise ValueError("reference feature must contain at least one observed value")
    candidates = (
        observed[math.ceil(len(observed) * index / bin_count) - 1]
        for index in range(1, bin_count)
    )
    return tuple(dict.fromkeys(candidates))


def _bin_counts(
    values: Sequence[float | None], boundaries: Sequence[float]
) -> tuple[int, ...]:
    counts = [0] * (len(boundaries) + 2)
    for value in values:
        index = (
            len(counts) - 1 if value is None else bisect.bisect_left(boundaries, value)
        )
        counts[index] += 1
    return tuple(counts)


def _smoothed_shares(counts: Sequence[int], epsilon: float) -> tuple[float, ...]:
    denominator = sum(counts) + (epsilon * len(counts))
    return tuple((count + epsilon) / denominator for count in counts)


def _numeric_bin_labels(boundaries: Sequence[float]) -> tuple[str, ...]:
    if not boundaries:
        return ("All observed values",)
    labels = [f"<= {_format_boundary(boundaries[0])}"]
    labels.extend(
        f"> {_format_boundary(lower)} and <= {_format_boundary(upper)}"
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    labels.append(f"> {_format_boundary(boundaries[-1])}")
    return tuple(labels)


def _format_boundary(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_integer(section: Mapping[str, object], key: str) -> int:
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _require_float(section: Mapping[str, object], key: str) -> float:
    value = section[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)
