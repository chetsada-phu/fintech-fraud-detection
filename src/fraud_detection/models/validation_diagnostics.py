"""Validation-only calibration and segment diagnostics for XGBoost scores."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from fraud_detection.features.matrix import JoinedFeatureRow


@dataclass(frozen=True, slots=True)
class ValidationDiagnosticsConfig:
    """Versioned settings for descriptive validation diagnostics."""

    calibration_bin_count: int
    amount_band_upper_bounds: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if (
            type(self.calibration_bin_count) is not int
            or not 2 <= self.calibration_bin_count <= 20
        ):
            raise ValueError("calibration_bin_count must be an integer from 2 to 20")
        if not self.amount_band_upper_bounds:
            raise ValueError("amount_band_upper_bounds must not be empty")
        previous = Decimal(0)
        for boundary in self.amount_band_upper_bounds:
            if (
                not isinstance(boundary, Decimal)
                or not boundary.is_finite()
                or boundary <= previous
            ):
                raise ValueError(
                    "amount_band_upper_bounds must be finite, positive, and "
                    "strictly increasing"
                )
            previous = boundary


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Observed labels and scores within one fixed probability interval."""

    lower_bound: float
    upper_bound: float
    includes_upper_bound: bool
    rows: int
    frauds: int
    mean_score: float | None
    observed_fraud_rate: float | None
    absolute_gap: float | None


@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    """Threshold errors and score behavior within one validation segment."""

    segment_name: str
    rows: int
    frauds: int
    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_score: float | None

    @property
    def fraud_rate(self) -> float | None:
        """Return the observed fraud-label rate for the segment."""
        if self.rows == 0:
            return None
        return self.frauds / self.rows

    @property
    def flag_rate(self) -> float | None:
        """Return the fraction flagged at the frozen threshold."""
        if self.rows == 0:
            return None
        return self.flagged / self.rows

    @property
    def precision(self) -> float | None:
        """Return precision among flagged rows when the segment has flags."""
        if self.flagged == 0:
            return None
        return self.true_positives / self.flagged

    @property
    def recall(self) -> float | None:
        """Return recall when the segment contains observed fraud."""
        if self.frauds == 0:
            return None
        return self.true_positives / self.frauds


@dataclass(frozen=True, slots=True)
class ValidationDiagnostics:
    """Descriptive calibration and segment evidence from validation only."""

    config: ValidationDiagnosticsConfig
    brier_score: float
    expected_calibration_error: float
    calibration_bins: tuple[CalibrationBin, ...]
    amount_segments: tuple[SegmentMetrics, ...]
    history_segments: tuple[SegmentMetrics, ...]


def load_validation_diagnostics_config(path: Path) -> ValidationDiagnosticsConfig:
    """Load and type-check fixed validation diagnostic settings."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("validation_diagnostics")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [validation_diagnostics] table")
    try:
        return ValidationDiagnosticsConfig(
            calibration_bin_count=_require_int(section, "calibration_bin_count"),
            amount_band_upper_bounds=_require_decimal_tuple(
                section, "amount_band_upper_bounds"
            ),
        )
    except KeyError as error:
        raise ValueError(
            f"missing validation diagnostic setting: {error.args[0]}"
        ) from error
    except ValueError as error:
        raise ValueError(
            f"invalid validation diagnostic configuration: {error}"
        ) from error


def analyze_validation_scores(
    rows: Sequence[JoinedFeatureRow],
    scores: Sequence[float],
    threshold: float,
    config: ValidationDiagnosticsConfig,
) -> ValidationDiagnostics:
    """Measure calibration and segment errors without fitting or tuning."""
    if not rows:
        raise ValueError("validation rows must not be empty")
    if len(rows) != len(scores):
        raise ValueError("validation rows and scores must have equal lengths")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and within (0, 1)")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("validation scores must be finite and within [0, 1]")

    labels = tuple(row.transaction.tx_fraud for row in rows)
    calibration_bins = _build_calibration_bins(
        labels, scores, config.calibration_bin_count
    )
    brier_score = sum(
        (score - label) ** 2 for score, label in zip(scores, labels, strict=True)
    ) / len(rows)
    expected_calibration_error = sum(
        (calibration_bin.rows / len(rows)) * (calibration_bin.absolute_gap or 0.0)
        for calibration_bin in calibration_bins
    )
    amount_names = tuple(
        _amount_segment_name(index, config.amount_band_upper_bounds)
        for index in range(len(config.amount_band_upper_bounds) + 1)
    )
    amount_assignments = tuple(
        _amount_segment_index(
            row.transaction.tx_amount, config.amount_band_upper_bounds
        )
        for row in rows
    )
    history_assignments = tuple(_history_segment_index(row) for row in rows)
    return ValidationDiagnostics(
        config=config,
        brier_score=brier_score,
        expected_calibration_error=expected_calibration_error,
        calibration_bins=calibration_bins,
        amount_segments=_build_segment_metrics(
            amount_names, amount_assignments, labels, scores, threshold
        ),
        history_segments=_build_segment_metrics(
            ("Missing prior history", "Prior history available"),
            history_assignments,
            labels,
            scores,
            threshold,
        ),
    )


def _build_calibration_bins(
    labels: Sequence[int], scores: Sequence[float], bin_count: int
) -> tuple[CalibrationBin, ...]:
    bin_labels: list[list[int]] = [[] for _ in range(bin_count)]
    bin_scores: list[list[float]] = [[] for _ in range(bin_count)]
    for label, score in zip(labels, scores, strict=True):
        bin_index = min(int(score * bin_count), bin_count - 1)
        bin_labels[bin_index].append(label)
        bin_scores[bin_index].append(score)

    calibration_bins = []
    for index, (group_labels, group_scores) in enumerate(
        zip(bin_labels, bin_scores, strict=True)
    ):
        rows = len(group_scores)
        frauds = sum(group_labels)
        mean_score = sum(group_scores) / rows if rows else None
        fraud_rate = frauds / rows if rows else None
        calibration_bins.append(
            CalibrationBin(
                lower_bound=index / bin_count,
                upper_bound=(index + 1) / bin_count,
                includes_upper_bound=index == bin_count - 1,
                rows=rows,
                frauds=frauds,
                mean_score=mean_score,
                observed_fraud_rate=fraud_rate,
                absolute_gap=(
                    abs(mean_score - fraud_rate)
                    if mean_score is not None and fraud_rate is not None
                    else None
                ),
            )
        )
    return tuple(calibration_bins)


def _build_segment_metrics(
    segment_names: Sequence[str],
    assignments: Sequence[int],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> tuple[SegmentMetrics, ...]:
    metrics = []
    for segment_index, segment_name in enumerate(segment_names):
        group = tuple(
            (label, score)
            for assignment, label, score in zip(
                assignments, labels, scores, strict=True
            )
            if assignment == segment_index
        )
        group_labels = tuple(label for label, _ in group)
        group_scores = tuple(score for _, score in group)
        flags = tuple(score >= threshold for score in group_scores)
        true_positives = sum(
            flag and label == 1 for flag, label in zip(flags, group_labels, strict=True)
        )
        false_positives = sum(
            flag and label == 0 for flag, label in zip(flags, group_labels, strict=True)
        )
        metrics.append(
            SegmentMetrics(
                segment_name=segment_name,
                rows=len(group),
                frauds=sum(group_labels),
                flagged=sum(flags),
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=sum(group_labels) - true_positives,
                mean_score=(
                    sum(group_scores) / len(group_scores) if group_scores else None
                ),
            )
        )
    return tuple(metrics)


def _amount_segment_index(amount: Decimal, upper_bounds: Sequence[Decimal]) -> int:
    for index, upper_bound in enumerate(upper_bounds):
        if amount <= upper_bound:
            return index
    return len(upper_bounds)


def _amount_segment_name(index: int, upper_bounds: Sequence[Decimal]) -> str:
    if index == 0:
        return f"Amount <= {_format_decimal(upper_bounds[0])}"
    if index == len(upper_bounds):
        return f"Amount > {_format_decimal(upper_bounds[-1])}"
    return (
        f"{_format_decimal(upper_bounds[index - 1])} < amount <= "
        f"{_format_decimal(upper_bounds[index])}"
    )


def _history_segment_index(row: JoinedFeatureRow) -> int:
    temporal = row.temporal
    missing_fields = (
        temporal.customer_amount_mean_prior is None,
        temporal.customer_amount_deviation_from_mean_prior is None,
        temporal.customer_seconds_since_previous is None,
    )
    if len(set(missing_fields)) != 1:
        raise ValueError("prior customer-history fields must be missing together")
    return 0 if all(missing_fields) else 1


def _format_decimal(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def _require_int(section: Mapping[str, object], key: str) -> int:
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _require_decimal_tuple(
    section: Mapping[str, object], key: str
) -> tuple[Decimal, ...]:
    value = section[key]
    if not isinstance(value, list) or any(
        type(item) not in {int, float} for item in value
    ):
        raise ValueError(f"{key} must be an array of numbers")
    return tuple(Decimal(str(item)) for item in value)
