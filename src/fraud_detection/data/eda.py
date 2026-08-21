"""Deterministic focused EDA for chronological transaction splits."""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from fraud_detection.data.schema import CSV_COLUMNS, Transaction, validate_transactions
from fraud_detection.data.splitter import LABEL_COLUMNS, SPLIT_FILENAMES

CENT = Decimal("0.01")
SPLIT_ORDER = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Measured facts for one transaction collection."""

    name: str
    rows: int
    start: datetime
    end: datetime
    frauds: int
    amount_min: Decimal
    amount_median: Decimal
    amount_mean: Decimal
    amount_p95: Decimal
    amount_max: Decimal
    customers: int
    terminals: int
    scenario_counts: tuple[int, int, int, int]

    @property
    def fraud_rate_percent(self) -> float:
        """Return the observed fraud-label percentage."""
        return (self.frauds / self.rows) * 100


@dataclass(frozen=True, slots=True)
class EdaReport:
    """Profiles and validated boundary facts for all chronological splits."""

    splits: tuple[DatasetProfile, DatasetProfile, DatasetProfile]
    overall: DatasetProfile


@dataclass(frozen=True, slots=True)
class ProcessedSplits:
    """Validated chronological transactions grouped by split."""

    train: tuple[Transaction, ...]
    validation: tuple[Transaction, ...]
    test: tuple[Transaction, ...]

    @property
    def ordered(self) -> tuple[tuple[Transaction, ...], ...]:
        """Return splits in train, validation, and test order."""
        return (self.train, self.validation, self.test)


def profile_processed_splits(input_directory: Path) -> EdaReport:
    """Load, validate, and profile train, validation, and test CSVs."""
    processed_splits = load_processed_splits(input_directory)
    transactions_by_split = processed_splits.ordered
    all_transactions = tuple(
        transaction
        for transactions in transactions_by_split
        for transaction in transactions
    )
    profiles = tuple(
        _profile_transactions(name, transactions)
        for name, transactions in zip(SPLIT_ORDER, transactions_by_split, strict=True)
    )
    return EdaReport(
        splits=profiles,
        overall=_profile_transactions("overall", all_transactions),
    )


def load_processed_splits(input_directory: Path) -> ProcessedSplits:
    """Load and validate the complete chronological split collection."""
    transactions_by_split = tuple(
        _read_transaction_rows(input_directory / SPLIT_FILENAMES[name])
        for name in SPLIT_ORDER
    )
    all_transactions = tuple(
        transaction
        for transactions in transactions_by_split
        for transaction in transactions
    )
    start_datetime = all_transactions[0].tx_datetime - timedelta(
        seconds=all_transactions[0].tx_time_seconds
    )
    validate_transactions(all_transactions, start_datetime)

    for earlier_name, earlier, later_name, later in zip(
        SPLIT_ORDER[:-1],
        transactions_by_split[:-1],
        SPLIT_ORDER[1:],
        transactions_by_split[1:],
        strict=True,
    ):
        if earlier[-1].tx_datetime >= later[0].tx_datetime:
            raise ValueError(
                f"{earlier_name} and {later_name} timestamps must be strictly ordered"
            )

    return ProcessedSplits(
        train=transactions_by_split[0],
        validation=transactions_by_split[1],
        test=transactions_by_split[2],
    )


def load_training_transactions(path: Path) -> tuple[Transaction, ...]:
    """Load and validate one chronological training CSV in isolation."""
    transactions = _read_transaction_rows(path)
    start_datetime = transactions[0].tx_datetime - timedelta(
        seconds=transactions[0].tx_time_seconds
    )
    validate_transactions(transactions, start_datetime)
    return transactions


def render_markdown(report: EdaReport) -> str:
    """Render a deterministic, portfolio-safe Markdown EDA report."""
    profiles = (*report.splits, report.overall)
    scenario_support_note = _scenario_support_note(report)
    lines = [
        "# Phase 1 Focused EDA",
        "",
        "Generated reproducibly by `fraud-profile-data` from the chronological",
        "files under `data/processed/`. All values below are measured from the",
        "current synthetic sample; they are not production fraud benchmarks.",
        "",
        "## Split Summary",
        "",
        "| Dataset | Rows | Start (UTC) | End (UTC) | Frauds | Fraud rate |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {_display_name(profile.name)} | {profile.rows:,} | {profile.start.isoformat()} | {profile.end.isoformat()} | {profile.frauds:,} | {profile.fraud_rate_percent:.4f}% |"
        for profile in profiles
    )
    lines.extend(
        [
            "",
            "## Transaction Amount Summary",
            "",
            "Nearest-rank is used for P95. Amounts retain the synthetic currency's",
            "two-decimal precision.",
            "",
            "| Dataset | Minimum | Median | Mean | P95 | Maximum |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {_display_name(profile.name)} | {_format_amount(profile.amount_min)} | {_format_amount(profile.amount_median)} | {_format_amount(profile.amount_mean)} | {_format_amount(profile.amount_p95)} | {_format_amount(profile.amount_max)} |"
        for profile in profiles
    )
    lines.extend(
        [
            "",
            "## Entity Coverage",
            "",
            "| Dataset | Unique customers | Unique terminals |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {_display_name(profile.name)} | {profile.customers:,} | "
        f"{profile.terminals:,} |"
        for profile in profiles
    )
    lines.extend(
        [
            "",
            "## Fraud Scenario Counts",
            "",
            "Scenario 0 is legitimate; scenarios 1-3 are synthetic fraud mechanisms",
            "defined in `docs/data_contract.md`.",
            "",
            "| Dataset | Scenario 0 | Scenario 1 | Scenario 2 | Scenario 3 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {_display_name(profile.name)} | {profile.scenario_counts[0]:,} | {profile.scenario_counts[1]:,} | {profile.scenario_counts[2]:,} | {profile.scenario_counts[3]:,} |"
        for profile in profiles
    )
    lines.extend(
        [
            "",
            "## Data Quality and Leakage Checks",
            "",
            f"- All {report.overall.rows:,} transaction IDs are contiguous and appear "
            "exactly once across the three splits.",
            "- Train, validation, and test timestamps are strictly ordered with no "
            "boundary overlap.",
            "- Every row passes the raw schema checks for timestamps, amounts, IDs, "
            "and label consistency.",
            f"- The post-event fields `{LABEL_COLUMNS[0]}` and `{LABEL_COLUMNS[1]}` "
            "remain evaluation labels and are excluded from the model-feature "
            "contract.",
            "",
            "## Interpretation Limits",
            "",
            "- The data is synthetic and scaled down for pipeline development.",
            "- Fraud counts are small, so split-level rates can vary materially and "
            "must not be interpreted as bank or market prevalence.",
            scenario_support_note,
            "- These descriptive measurements do not establish model performance, "
            "financial impact, fairness, or production readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: EdaReport, output_path: Path) -> None:
    """Write a rendered EDA report atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_transaction_rows(path: Path) -> tuple[Transaction, ...]:
    with path.open(encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"{path}: columns must exactly match the transaction data contract"
            )
        try:
            transactions = tuple(_transaction_from_row(row) for row in reader)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid transaction value: {error}") from error

    if not transactions:
        raise ValueError(f"{path}: split must contain at least one transaction")
    return transactions


def _transaction_from_row(row: Mapping[str, str]) -> Transaction:
    if any(row.get(column) in {None, ""} for column in CSV_COLUMNS):
        raise ValueError("every transaction field must be populated")
    return Transaction(
        transaction_id=int(row["TRANSACTION_ID"]),
        tx_datetime=datetime.fromisoformat(row["TX_DATETIME"]),
        customer_id=int(row["CUSTOMER_ID"]),
        terminal_id=int(row["TERMINAL_ID"]),
        tx_amount=Decimal(row["TX_AMOUNT"]),
        tx_time_seconds=int(row["TX_TIME_SECONDS"]),
        tx_time_days=int(row["TX_TIME_DAYS"]),
        tx_fraud=int(row["TX_FRAUD"]),
        tx_fraud_scenario=int(row["TX_FRAUD_SCENARIO"]),
    )


def _profile_transactions(
    name: str, transactions: Sequence[Transaction]
) -> DatasetProfile:
    amounts = sorted(transaction.tx_amount for transaction in transactions)
    scenario_counts = tuple(
        sum(transaction.tx_fraud_scenario == scenario for transaction in transactions)
        for scenario in range(4)
    )
    return DatasetProfile(
        name=name,
        rows=len(transactions),
        start=transactions[0].tx_datetime,
        end=transactions[-1].tx_datetime,
        frauds=sum(transaction.tx_fraud for transaction in transactions),
        amount_min=amounts[0],
        amount_median=_median(amounts),
        amount_mean=(sum(amounts) / len(amounts)).quantize(
            CENT, rounding=ROUND_HALF_UP
        ),
        amount_p95=amounts[math.ceil(len(amounts) * 0.95) - 1],
        amount_max=amounts[-1],
        customers=len({transaction.customer_id for transaction in transactions}),
        terminals=len({transaction.terminal_id for transaction in transactions}),
        scenario_counts=scenario_counts,
    )


def _median(amounts: Sequence[Decimal]) -> Decimal:
    middle = len(amounts) // 2
    if len(amounts) % 2:
        return amounts[middle]
    return ((amounts[middle - 1] + amounts[middle]) / 2).quantize(
        CENT, rounding=ROUND_HALF_UP
    )


def _display_name(name: str) -> str:
    return name.capitalize()


def _scenario_support_note(report: EdaReport) -> str:
    missing_support = []
    for profile in report.splits:
        missing = [
            str(scenario)
            for scenario in range(1, 4)
            if profile.scenario_counts[scenario] == 0
        ]
        if missing:
            missing_support.append(
                f"{_display_name(profile.name)} has no scenario {', '.join(missing)}"
            )
    if not missing_support:
        return "- Every split contains at least one example of each fraud scenario."
    return (
        "- Fraud-scenario support is uneven: "
        + "; ".join(missing_support)
        + ". Later evaluation must report scenario support and avoid treating "
        "split-level rates as stable estimates."
    )


def _format_amount(amount: Decimal) -> str:
    return f"{amount:,.2f}"
