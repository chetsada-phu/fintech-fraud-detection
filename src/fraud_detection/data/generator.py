"""Deterministic generator for a small ULB-style transaction dataset."""

from __future__ import annotations

import csv
import math
import os
import random
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from fraud_detection.data.schema import (
    CSV_COLUMNS,
    SECONDS_PER_DAY,
    Transaction,
    validate_transactions,
)

CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Versioned parameters controlling synthetic data generation."""

    seed: int
    start_datetime: datetime
    days: int
    customers: int
    terminals: int
    terminals_per_customer: int
    min_customer_mean_amount: Decimal
    max_customer_mean_amount: Decimal
    min_customer_mean_transactions_per_day: float
    max_customer_mean_transactions_per_day: float
    high_amount_fraud_threshold: Decimal
    terminal_compromise_events: int
    terminal_compromise_duration_days: int
    customer_compromise_events: int
    customer_compromise_duration_days: int
    customer_fraud_fraction: float
    customer_fraud_amount_multiplier: int

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce meaningful data."""
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if (
            self.start_datetime.tzinfo is None
            or self.start_datetime.utcoffset() != timedelta(0)
        ):
            raise ValueError("start_datetime must be timezone-aware UTC")
        if self.days <= 0 or self.customers <= 0 or self.terminals <= 0:
            raise ValueError("days, customers, and terminals must be positive")
        if not 1 <= self.terminals_per_customer <= self.terminals:
            raise ValueError("terminals_per_customer must be within terminal count")
        if not (
            Decimal(0) < self.min_customer_mean_amount <= self.max_customer_mean_amount
        ):
            raise ValueError("customer amount bounds must be positive and ordered")
        if not (
            0
            <= self.min_customer_mean_transactions_per_day
            <= self.max_customer_mean_transactions_per_day
        ):
            raise ValueError("customer transaction-rate bounds must be ordered")
        if self.high_amount_fraud_threshold <= 0:
            raise ValueError("high_amount_fraud_threshold must be positive")
        if self.terminal_compromise_events < 0 or self.customer_compromise_events < 0:
            raise ValueError("compromise event counts must be non-negative")
        if not 1 <= self.terminal_compromise_duration_days <= self.days:
            raise ValueError("terminal compromise duration must fit within days")
        if not 1 <= self.customer_compromise_duration_days <= self.days:
            raise ValueError("customer compromise duration must fit within days")
        if not 0 < self.customer_fraud_fraction <= 1:
            raise ValueError("customer_fraud_fraction must be in (0, 1]")
        if self.customer_fraud_amount_multiplier <= 1:
            raise ValueError("customer_fraud_amount_multiplier must exceed 1")


@dataclass(frozen=True, slots=True)
class _CustomerProfile:
    customer_id: int
    mean_amount: float
    standard_deviation: float
    mean_transactions_per_day: float
    terminal_ids: tuple[int, ...]


@dataclass(slots=True)
class _DraftTransaction:
    time_seconds: int
    customer_id: int
    terminal_id: int
    amount: Decimal
    fraud: int = 0
    fraud_scenario: int = 0


def load_simulation_config(path: Path) -> SimulationConfig:
    """Load and type-check simulation parameters from a TOML file."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("simulation")
    if not isinstance(section, dict):
        raise ValueError("configuration must contain a [simulation] table")

    try:
        start_datetime = datetime.fromisoformat(
            _require_string(section, "start_datetime")
        )
        return SimulationConfig(
            seed=_require_int(section, "seed"),
            start_datetime=start_datetime,
            days=_require_int(section, "days"),
            customers=_require_int(section, "customers"),
            terminals=_require_int(section, "terminals"),
            terminals_per_customer=_require_int(section, "terminals_per_customer"),
            min_customer_mean_amount=_require_decimal(
                section, "min_customer_mean_amount"
            ),
            max_customer_mean_amount=_require_decimal(
                section, "max_customer_mean_amount"
            ),
            min_customer_mean_transactions_per_day=_require_number(
                section, "min_customer_mean_transactions_per_day"
            ),
            max_customer_mean_transactions_per_day=_require_number(
                section, "max_customer_mean_transactions_per_day"
            ),
            high_amount_fraud_threshold=_require_decimal(
                section, "high_amount_fraud_threshold"
            ),
            terminal_compromise_events=_require_int(
                section, "terminal_compromise_events"
            ),
            terminal_compromise_duration_days=_require_int(
                section, "terminal_compromise_duration_days"
            ),
            customer_compromise_events=_require_int(
                section, "customer_compromise_events"
            ),
            customer_compromise_duration_days=_require_int(
                section, "customer_compromise_duration_days"
            ),
            customer_fraud_fraction=_require_number(section, "customer_fraud_fraction"),
            customer_fraud_amount_multiplier=_require_int(
                section, "customer_fraud_amount_multiplier"
            ),
        )
    except KeyError as error:
        raise ValueError(f"missing simulation setting: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"invalid simulation configuration: {error}") from error


def generate_transactions(config: SimulationConfig) -> tuple[Transaction, ...]:
    """Generate, label, sort, and validate deterministic transactions."""
    transaction_rng = random.Random(config.seed)
    fraud_rng = random.Random(config.seed + 1)
    profiles = _generate_customer_profiles(config, transaction_rng)
    drafts: list[_DraftTransaction] = []

    for profile in profiles:
        for day in range(config.days):
            transaction_count = _sample_poisson(
                transaction_rng, profile.mean_transactions_per_day
            )
            for _ in range(transaction_count):
                second_of_day = _sample_second_of_day(transaction_rng)
                drafts.append(
                    _DraftTransaction(
                        time_seconds=(day * SECONDS_PER_DAY) + second_of_day,
                        customer_id=profile.customer_id,
                        terminal_id=transaction_rng.choice(profile.terminal_ids),
                        amount=_sample_amount(profile, transaction_rng),
                    )
                )

    drafts.sort(
        key=lambda transaction: (
            transaction.time_seconds,
            transaction.customer_id,
            transaction.terminal_id,
        )
    )
    _apply_fraud_scenarios(drafts, config, fraud_rng)

    transactions = tuple(
        Transaction(
            transaction_id=transaction_id,
            tx_datetime=config.start_datetime + timedelta(seconds=draft.time_seconds),
            customer_id=draft.customer_id,
            terminal_id=draft.terminal_id,
            tx_amount=draft.amount,
            tx_time_seconds=draft.time_seconds,
            tx_time_days=draft.time_seconds // SECONDS_PER_DAY,
            tx_fraud=draft.fraud,
            tx_fraud_scenario=draft.fraud_scenario,
        )
        for transaction_id, draft in enumerate(drafts)
    )
    validate_transactions(transactions, config.start_datetime)
    return transactions


def write_transactions_csv(
    transactions: Sequence[Transaction], output_path: Path
) -> None:
    """Write records atomically so interrupted runs cannot leave partial data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(transaction.to_csv_row() for transaction in transactions)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _generate_customer_profiles(
    config: SimulationConfig, rng: random.Random
) -> tuple[_CustomerProfile, ...]:
    profiles = []
    terminal_population = range(config.terminals)
    minimum_amount = float(config.min_customer_mean_amount)
    maximum_amount = float(config.max_customer_mean_amount)

    for customer_id in range(config.customers):
        mean_amount = rng.uniform(minimum_amount, maximum_amount)
        profiles.append(
            _CustomerProfile(
                customer_id=customer_id,
                mean_amount=mean_amount,
                standard_deviation=mean_amount / 2,
                mean_transactions_per_day=rng.uniform(
                    config.min_customer_mean_transactions_per_day,
                    config.max_customer_mean_transactions_per_day,
                ),
                terminal_ids=tuple(
                    sorted(
                        rng.sample(terminal_population, config.terminals_per_customer)
                    )
                ),
            )
        )
    return tuple(profiles)


def _sample_poisson(rng: random.Random, rate: float) -> int:
    if rate == 0:
        return 0
    limit = math.exp(-rate)
    product = 1.0
    count = 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def _sample_second_of_day(rng: random.Random) -> int:
    while True:
        candidate = int(rng.gauss(SECONDS_PER_DAY / 2, 20_000))
        if 0 <= candidate < SECONDS_PER_DAY:
            return candidate


def _sample_amount(profile: _CustomerProfile, rng: random.Random) -> Decimal:
    candidate = rng.gauss(profile.mean_amount, profile.standard_deviation)
    if candidate <= 0:
        candidate = rng.uniform(0.01, profile.mean_amount * 2)
    candidate = max(candidate, 0.01)
    return Decimal(str(candidate)).quantize(CENT, rounding=ROUND_HALF_UP)


def _apply_fraud_scenarios(
    transactions: list[_DraftTransaction],
    config: SimulationConfig,
    rng: random.Random,
) -> None:
    for transaction in transactions:
        if transaction.amount > config.high_amount_fraud_threshold:
            _mark_fraud(transaction, scenario=1)

    _apply_terminal_compromises(transactions, config, rng)
    _apply_customer_compromises(transactions, config, rng)


def _apply_terminal_compromises(
    transactions: list[_DraftTransaction],
    config: SimulationConfig,
    rng: random.Random,
) -> None:
    latest_start = config.days - config.terminal_compromise_duration_days
    for _ in range(config.terminal_compromise_events):
        start_day = rng.randint(0, latest_start)
        end_day = start_day + config.terminal_compromise_duration_days
        eligible_terminals = sorted(
            {
                transaction.terminal_id
                for transaction in transactions
                if start_day <= transaction.time_seconds // SECONDS_PER_DAY < end_day
                and transaction.fraud == 0
            }
        )
        if not eligible_terminals:
            continue
        compromised_terminal = rng.choice(eligible_terminals)
        for transaction in transactions:
            transaction_day = transaction.time_seconds // SECONDS_PER_DAY
            if (
                start_day <= transaction_day < end_day
                and transaction.terminal_id == compromised_terminal
                and transaction.fraud == 0
            ):
                _mark_fraud(transaction, scenario=2)


def _apply_customer_compromises(
    transactions: list[_DraftTransaction],
    config: SimulationConfig,
    rng: random.Random,
) -> None:
    latest_start = config.days - config.customer_compromise_duration_days
    multiplier = Decimal(config.customer_fraud_amount_multiplier)

    for _ in range(config.customer_compromise_events):
        start_day = rng.randint(0, latest_start)
        end_day = start_day + config.customer_compromise_duration_days
        eligible_customers = sorted(
            {
                transaction.customer_id
                for transaction in transactions
                if start_day <= transaction.time_seconds // SECONDS_PER_DAY < end_day
                and transaction.fraud == 0
            }
        )
        if not eligible_customers:
            continue
        compromised_customer = rng.choice(eligible_customers)
        candidates = [
            transaction
            for transaction in transactions
            if start_day <= transaction.time_seconds // SECONDS_PER_DAY < end_day
            and transaction.customer_id == compromised_customer
            and transaction.fraud == 0
        ]
        fraud_count = max(1, int(len(candidates) * config.customer_fraud_fraction))
        for transaction in rng.sample(candidates, fraud_count):
            transaction.amount = (transaction.amount * multiplier).quantize(CENT)
            _mark_fraud(transaction, scenario=3)


def _mark_fraud(transaction: _DraftTransaction, scenario: int) -> None:
    transaction.fraud = 1
    transaction.fraud_scenario = scenario


def _require_string(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_int(section: Mapping[str, object], key: str) -> int:
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _require_number(section: Mapping[str, object], key: str) -> float:
    value = section[key]
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be a number")
    return float(value)


def _require_decimal(section: Mapping[str, object], key: str) -> Decimal:
    return Decimal(str(_require_number(section, key)))
