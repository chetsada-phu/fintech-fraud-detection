"""Label-free request and response schemas for provisional serving."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraud_detection.decisioning.inference import (
    TransactionDecisionContext,
    TransactionDecisionResult,
)
from fraud_detection.decisioning.policy import Decision
from fraud_detection.decisioning.scored_inference import (
    ScoredTransactionDecisionResult,
)
from fraud_detection.models.model_input import ProvisionalModelInput


class DecisionRequest(BaseModel):
    """Supplied score plus current and strictly past decision-time context."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int = Field(ge=0, strict=True)
    risk_score: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    transaction_amount: Decimal = Field(gt=0, allow_inf_nan=False)
    customer_tx_count_short_window: int = Field(ge=0, strict=True)
    customer_amount_mean_prior: Decimal | None = Field(allow_inf_nan=False)
    customer_amount_deviation_from_mean_prior: Decimal | None = Field(
        allow_inf_nan=False
    )
    customer_seconds_since_previous: int | None = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_decision_context(self) -> Self:
        """Reject inconsistent history fields before domain decisioning."""
        self.to_context()
        return self

    def to_context(self) -> TransactionDecisionContext:
        """Convert validated API fields to the framework-free domain contract."""
        return TransactionDecisionContext(
            transaction_id=self.transaction_id,
            transaction_amount=self.transaction_amount,
            customer_tx_count_short_window=self.customer_tx_count_short_window,
            customer_amount_mean_prior=self.customer_amount_mean_prior,
            customer_amount_deviation_from_mean_prior=(
                self.customer_amount_deviation_from_mean_prior
            ),
            customer_seconds_since_previous=self.customer_seconds_since_previous,
        )


class DecisionResponse(BaseModel):
    """Public result returned by the decision-only endpoint."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    risk_score: float
    decision: Decision
    reason_codes: tuple[str, ...]
    policy_version: str

    @classmethod
    def from_result(cls, result: TransactionDecisionResult) -> Self:
        """Map the framework-free result to the public response schema."""
        return cls(
            transaction_id=result.transaction_id,
            risk_score=result.risk_score,
            decision=result.decision,
            reason_codes=result.reason_codes,
            policy_version=result.policy_version,
        )


class ScoredDecisionRequest(BaseModel):
    """One label-free record with features precomputed at decision time."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int = Field(ge=0, strict=True)
    tx_amount: Decimal = Field(gt=0, allow_inf_nan=False)
    tx_time_days: int = Field(ge=0, strict=True)
    tx_datetime: datetime
    customer_tx_count_short_window: int = Field(ge=0, strict=True)
    customer_tx_count_long_window: int = Field(ge=0, strict=True)
    customer_amount_mean_prior: Decimal | None = Field(allow_inf_nan=False)
    customer_amount_deviation_from_mean_prior: Decimal | None = Field(
        allow_inf_nan=False
    )
    customer_seconds_since_previous: int | None = Field(ge=0, strict=True)
    customer_id: int = Field(ge=0, strict=True)
    terminal_id: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_model_input(self) -> Self:
        """Reject inconsistent values before request-time prediction."""
        self.to_model_input()
        return self

    def to_model_input(self) -> ProvisionalModelInput:
        """Convert the public schema to the frozen framework-free contract."""
        return ProvisionalModelInput(
            transaction_id=self.transaction_id,
            tx_amount=self.tx_amount,
            tx_time_days=self.tx_time_days,
            tx_datetime=self.tx_datetime,
            customer_tx_count_short_window=self.customer_tx_count_short_window,
            customer_tx_count_long_window=self.customer_tx_count_long_window,
            customer_amount_mean_prior=self.customer_amount_mean_prior,
            customer_amount_deviation_from_mean_prior=(
                self.customer_amount_deviation_from_mean_prior
            ),
            customer_seconds_since_previous=self.customer_seconds_since_previous,
            customer_id=self.customer_id,
            terminal_id=self.terminal_id,
        )


class ScoredDecisionResponse(BaseModel):
    """Scored decision plus the exact model and policy identity."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    risk_score: float
    decision: Decision
    reason_codes: tuple[str, ...]
    policy_version: str
    score_source_label: str
    artifact_version: str
    artifact_sha256: str
    xgboost_config_sha256: str

    @classmethod
    def from_result(cls, result: ScoredTransactionDecisionResult) -> Self:
        """Map the framework-free scored result to the public schema."""
        return cls(
            transaction_id=result.transaction_id,
            risk_score=result.risk_score,
            decision=result.decision,
            reason_codes=result.reason_codes,
            policy_version=result.policy_version,
            score_source_label=result.score_source_label,
            artifact_version=result.artifact_version,
            artifact_sha256=result.artifact_sha256,
            xgboost_config_sha256=result.xgboost_config_sha256,
        )
