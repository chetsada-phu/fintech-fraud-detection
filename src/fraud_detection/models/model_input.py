"""Label-free input contract for the frozen provisional XGBoost pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final

from fraud_detection.features.matrix import JoinedFeatureRow

XGBOOST_FEATURE_CONTRACT: Final = (
    "TX_AMOUNT",
    "TX_TIME_DAYS",
    "TX_HOUR_SIN",
    "TX_HOUR_COS",
    "CUSTOMER_TX_COUNT_SHORT_WINDOW",
    "CUSTOMER_TX_COUNT_LONG_WINDOW",
    "CUSTOMER_AMOUNT_MEAN_PRIOR",
    "CUSTOMER_AMOUNT_DEVIATION_FROM_MEAN_PRIOR",
    "CUSTOMER_SECONDS_SINCE_PREVIOUS",
    "CUSTOMER_ID",
    "TERMINAL_ID",
)


@dataclass(frozen=True, slots=True)
class ProvisionalModelInput:
    """One precomputed decision-time record with no outcome-label fields."""

    transaction_id: int
    tx_amount: Decimal
    tx_time_days: int
    tx_datetime: datetime
    customer_tx_count_short_window: int
    customer_tx_count_long_window: int
    customer_amount_mean_prior: Decimal | None
    customer_amount_deviation_from_mean_prior: Decimal | None
    customer_seconds_since_previous: int | None
    customer_id: int
    terminal_id: int

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.transaction_id, "transaction_id")
        _validate_non_negative_integer(self.tx_time_days, "tx_time_days")
        _validate_non_negative_integer(self.customer_id, "customer_id")
        _validate_non_negative_integer(self.terminal_id, "terminal_id")
        _validate_non_negative_integer(
            self.customer_tx_count_short_window,
            "customer_tx_count_short_window",
        )
        _validate_non_negative_integer(
            self.customer_tx_count_long_window,
            "customer_tx_count_long_window",
        )
        if self.customer_tx_count_short_window > self.customer_tx_count_long_window:
            raise ValueError(
                "customer short-window count cannot exceed long-window count"
            )
        if not isinstance(self.tx_amount, Decimal):
            raise ValueError("tx_amount must be a Decimal")
        if not self.tx_amount.is_finite() or self.tx_amount <= 0:
            raise ValueError("tx_amount must be finite and positive")
        if not isinstance(self.tx_datetime, datetime):
            raise ValueError("tx_datetime must be a datetime")
        if self.tx_datetime.tzinfo is None or self.tx_datetime.utcoffset() != timedelta(
            0
        ):
            raise ValueError("tx_datetime must be timezone-aware UTC")

        missing_history = (
            self.customer_amount_mean_prior is None,
            self.customer_amount_deviation_from_mean_prior is None,
            self.customer_seconds_since_previous is None,
        )
        if len(set(missing_history)) != 1:
            raise ValueError("prior customer-history fields must be missing together")
        if self.customer_amount_mean_prior is not None:
            if not isinstance(self.customer_amount_mean_prior, Decimal):
                raise ValueError("customer_amount_mean_prior must be a Decimal")
            if (
                not self.customer_amount_mean_prior.is_finite()
                or self.customer_amount_mean_prior <= 0
            ):
                raise ValueError(
                    "customer_amount_mean_prior must be finite and positive"
                )
        if self.customer_amount_deviation_from_mean_prior is not None:
            if not isinstance(
                self.customer_amount_deviation_from_mean_prior,
                Decimal,
            ):
                raise ValueError(
                    "customer_amount_deviation_from_mean_prior must be a Decimal"
                )
            if not self.customer_amount_deviation_from_mean_prior.is_finite():
                raise ValueError(
                    "customer_amount_deviation_from_mean_prior must be finite"
                )
        if self.customer_seconds_since_previous is not None:
            _validate_non_negative_integer(
                self.customer_seconds_since_previous,
                "customer_seconds_since_previous",
            )

    @classmethod
    def from_joined_feature_row(cls, row: JoinedFeatureRow) -> ProvisionalModelInput:
        """Copy decision-time fields from an offline row without copying labels."""
        transaction = row.transaction
        temporal = row.temporal
        if transaction.transaction_id != temporal.transaction_id:
            raise ValueError("transaction and temporal feature IDs must align")
        return cls(
            transaction_id=transaction.transaction_id,
            tx_amount=transaction.tx_amount,
            tx_time_days=transaction.tx_time_days,
            tx_datetime=transaction.tx_datetime,
            customer_tx_count_short_window=(temporal.customer_tx_count_short_window),
            customer_tx_count_long_window=temporal.customer_tx_count_long_window,
            customer_amount_mean_prior=temporal.customer_amount_mean_prior,
            customer_amount_deviation_from_mean_prior=(
                temporal.customer_amount_deviation_from_mean_prior
            ),
            customer_seconds_since_previous=(temporal.customer_seconds_since_previous),
            customer_id=transaction.customer_id,
            terminal_id=transaction.terminal_id,
        )

    def to_feature_values(self) -> tuple[float | str, ...]:
        """Map fields to the exact ordered frozen XGBoost feature contract."""
        hour_fraction = (
            self.tx_datetime.hour
            + (self.tx_datetime.minute / 60)
            + (self.tx_datetime.second / 3_600)
        ) / 24
        angle = 2 * math.pi * hour_fraction
        return (
            float(self.tx_amount),
            float(self.tx_time_days),
            math.sin(angle),
            math.cos(angle),
            float(self.customer_tx_count_short_window),
            float(self.customer_tx_count_long_window),
            _optional_float(self.customer_amount_mean_prior),
            _optional_float(self.customer_amount_deviation_from_mean_prior),
            _optional_float(self.customer_seconds_since_previous),
            f"customer_{self.customer_id}",
            f"terminal_{self.terminal_id}",
        )


def _optional_float(value: object | None) -> float:
    if value is None:
        return math.nan
    return float(value)


def _validate_non_negative_integer(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
