"""Schema and validation rules for simulated transaction records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final

SECONDS_PER_DAY: Final = 86_400
CSV_COLUMNS: Final = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)


class TransactionValidationError(ValueError):
    """Raised when a transaction collection violates the data contract."""


@dataclass(frozen=True, slots=True)
class Transaction:
    """One labeled synthetic payment transaction."""

    transaction_id: int
    tx_datetime: datetime
    customer_id: int
    terminal_id: int
    tx_amount: Decimal
    tx_time_seconds: int
    tx_time_days: int
    tx_fraud: int
    tx_fraud_scenario: int

    def to_csv_row(self) -> dict[str, int | str]:
        """Return the record using the public ULB-style column names."""
        return {
            "TRANSACTION_ID": self.transaction_id,
            "TX_DATETIME": self.tx_datetime.isoformat(timespec="seconds"),
            "CUSTOMER_ID": self.customer_id,
            "TERMINAL_ID": self.terminal_id,
            "TX_AMOUNT": format(self.tx_amount, ".2f"),
            "TX_TIME_SECONDS": self.tx_time_seconds,
            "TX_TIME_DAYS": self.tx_time_days,
            "TX_FRAUD": self.tx_fraud,
            "TX_FRAUD_SCENARIO": self.tx_fraud_scenario,
        }


def validate_transactions(
    transactions: Sequence[Transaction], start_datetime: datetime
) -> None:
    """Validate identifiers, chronology, money, and label consistency."""
    if start_datetime.tzinfo is None or start_datetime.utcoffset() != timedelta(0):
        raise TransactionValidationError("start_datetime must be timezone-aware UTC")
    if not transactions:
        raise TransactionValidationError("the transaction collection must not be empty")

    previous_seconds = -1
    seen_ids: set[int] = set()

    for row_number, transaction in enumerate(transactions):
        prefix = f"row {row_number}"
        if transaction.transaction_id in seen_ids:
            raise TransactionValidationError(f"{prefix}: TRANSACTION_ID must be unique")
        seen_ids.add(transaction.transaction_id)
        if transaction.transaction_id != row_number:
            raise TransactionValidationError(
                f"{prefix}: TRANSACTION_ID must follow chronological row order"
            )
        if transaction.customer_id < 0 or transaction.terminal_id < 0:
            raise TransactionValidationError(
                f"{prefix}: customer and terminal IDs must be non-negative"
            )
        if not transaction.tx_amount.is_finite() or transaction.tx_amount <= 0:
            raise TransactionValidationError(f"{prefix}: TX_AMOUNT must be positive")
        if transaction.tx_time_seconds < previous_seconds:
            raise TransactionValidationError(
                f"{prefix}: transactions must be sorted chronologically"
            )
        if transaction.tx_time_seconds < 0:
            raise TransactionValidationError(
                f"{prefix}: TX_TIME_SECONDS must be non-negative"
            )
        previous_seconds = transaction.tx_time_seconds
        if transaction.tx_time_days != transaction.tx_time_seconds // SECONDS_PER_DAY:
            raise TransactionValidationError(
                f"{prefix}: TX_TIME_DAYS does not match TX_TIME_SECONDS"
            )
        expected_datetime = start_datetime + timedelta(
            seconds=transaction.tx_time_seconds
        )
        if transaction.tx_datetime != expected_datetime:
            raise TransactionValidationError(
                f"{prefix}: TX_DATETIME does not match TX_TIME_SECONDS"
            )
        if transaction.tx_fraud not in {0, 1}:
            raise TransactionValidationError(f"{prefix}: TX_FRAUD must be 0 or 1")
        if transaction.tx_fraud_scenario not in {0, 1, 2, 3}:
            raise TransactionValidationError(
                f"{prefix}: TX_FRAUD_SCENARIO must be between 0 and 3"
            )
        expected_fraud = int(transaction.tx_fraud_scenario != 0)
        if transaction.tx_fraud != expected_fraud:
            raise TransactionValidationError(
                f"{prefix}: TX_FRAUD and TX_FRAUD_SCENARIO disagree"
            )
