"""Validation-only business-cost sensitivity for the three-way policy."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fraud_detection.data.schema import Transaction
from fraud_detection.decisioning.costs import BusinessCostConfig
from fraud_detection.decisioning.policy import (
    DecisionPolicyConfig,
    PolicySelection,
    select_policy_thresholds,
)


@dataclass(frozen=True, slots=True)
class CostSensitivityScenario:
    """One isolated set of optional overrides to the base cost assumptions."""

    key: str
    display_name: str
    fraud_loss_multiplier: Decimal | None = None
    manual_review_cost: Decimal | None = None
    max_manual_review_rate: float | None = None
    false_decline_cost: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError(
                "scenario key must contain letters, numbers, or underscores"
            )
        if not self.display_name.strip():
            raise ValueError("scenario display_name must not be empty")
        for name, value in (
            ("fraud_loss_multiplier", self.fraud_loss_multiplier),
            ("manual_review_cost", self.manual_review_cost),
            ("false_decline_cost", self.false_decline_cost),
        ):
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.fraud_loss_multiplier == 0:
            raise ValueError("fraud_loss_multiplier must be positive")
        if self.max_manual_review_rate is not None and (
            not math.isfinite(self.max_manual_review_rate)
            or not 0 < self.max_manual_review_rate <= 1
        ):
            raise ValueError("max_manual_review_rate must be in (0, 1]")

    @property
    def has_override(self) -> bool:
        """Return whether this scenario changes at least one base assumption."""
        return any(
            value is not None
            for value in (
                self.fraud_loss_multiplier,
                self.manual_review_cost,
                self.max_manual_review_rate,
                self.false_decline_cost,
            )
        )

    def resolve(self, base: BusinessCostConfig) -> BusinessCostConfig:
        """Create a new frozen cost config without mutating the base object."""
        return BusinessCostConfig(
            fraud_loss_multiplier=(
                self.fraud_loss_multiplier
                if self.fraud_loss_multiplier is not None
                else base.fraud_loss_multiplier
            ),
            manual_review_cost=(
                self.manual_review_cost
                if self.manual_review_cost is not None
                else base.manual_review_cost
            ),
            max_manual_review_rate=(
                self.max_manual_review_rate
                if self.max_manual_review_rate is not None
                else base.max_manual_review_rate
            ),
            false_decline_cost=(
                self.false_decline_cost
                if self.false_decline_cost is not None
                else base.false_decline_cost
            ),
        )


@dataclass(frozen=True, slots=True)
class PolicySensitivityConfig:
    """Versioned scenario order and validation-only evaluation contract."""

    evaluation_split: str
    base_scenario_key: str
    scenarios: tuple[CostSensitivityScenario, ...]

    def __post_init__(self) -> None:
        if self.evaluation_split != "validation":
            raise ValueError("evaluation_split must be 'validation'")
        if len(self.scenarios) < 2:
            raise ValueError("at least two cost-sensitivity scenarios are required")
        keys = tuple(scenario.key for scenario in self.scenarios)
        if len(set(keys)) != len(keys):
            raise ValueError("cost-sensitivity scenario keys must be unique")
        if not keys or keys[0] != self.base_scenario_key:
            raise ValueError("base_scenario_key must identify the first scenario")
        if self.scenarios[0].has_override:
            raise ValueError("the base scenario must not override business costs")
        if any(not scenario.has_override for scenario in self.scenarios[1:]):
            raise ValueError("every non-base scenario must override an assumption")
        signatures = tuple(
            (
                scenario.fraud_loss_multiplier,
                scenario.manual_review_cost,
                scenario.max_manual_review_rate,
                scenario.false_decline_cost,
            )
            for scenario in self.scenarios
        )
        if len(set(signatures)) != len(signatures):
            raise ValueError("cost-sensitivity scenarios must use unique overrides")


@dataclass(frozen=True, slots=True)
class PolicySensitivityResult:
    """Selected validation policy under one resolved scenario."""

    scenario: CostSensitivityScenario
    cost_config: BusinessCostConfig
    selection: PolicySelection


@dataclass(frozen=True, slots=True)
class PolicySensitivityAnalysis:
    """Deterministically ordered scenario results from shared validation scores."""

    results: tuple[PolicySensitivityResult, ...]


def load_policy_sensitivity_config(path: Path) -> PolicySensitivityConfig:
    """Load and type-check the versioned business-cost scenarios."""
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    section = document.get("decision_policy_sensitivity")
    if not isinstance(section, dict):
        raise ValueError(
            "configuration must contain a [decision_policy_sensitivity] table"
        )
    try:
        raw_scenarios = section["scenarios"]
        if not isinstance(raw_scenarios, list):
            raise ValueError("scenarios must be an array of tables")
        return PolicySensitivityConfig(
            evaluation_split=_require_str(section, "evaluation_split"),
            base_scenario_key=_require_str(section, "base_scenario_key"),
            scenarios=tuple(_load_scenario(value) for value in raw_scenarios),
        )
    except KeyError as error:
        raise ValueError(
            f"missing decision-policy sensitivity setting: {error.args[0]}"
        ) from error
    except ValueError as error:
        raise ValueError(
            f"invalid decision-policy sensitivity configuration: {error}"
        ) from error


def analyze_policy_cost_sensitivity(
    transactions: Sequence[Transaction],
    risk_scores: Sequence[float],
    policy_config: DecisionPolicyConfig,
    sensitivity_config: PolicySensitivityConfig,
    base_cost_config: BusinessCostConfig,
    *,
    split_name: str,
) -> PolicySensitivityAnalysis:
    """Select thresholds for ordered isolated scenarios using shared scores."""
    if split_name != "validation" or split_name != sensitivity_config.evaluation_split:
        raise ValueError("policy cost sensitivity may evaluate validation only")
    results = []
    for scenario in sensitivity_config.scenarios:
        scenario_costs = scenario.resolve(base_cost_config)
        selection = select_policy_thresholds(
            transactions,
            risk_scores,
            policy_config,
            scenario_costs,
            split_name="validation",
        )
        results.append(
            PolicySensitivityResult(
                scenario=scenario,
                cost_config=scenario_costs,
                selection=selection,
            )
        )
    return PolicySensitivityAnalysis(results=tuple(results))


def _load_scenario(value: object) -> CostSensitivityScenario:
    if not isinstance(value, dict):
        raise ValueError("each scenario must be a table")
    return CostSensitivityScenario(
        key=_require_str(value, "key"),
        display_name=_require_str(value, "display_name"),
        fraud_loss_multiplier=_optional_decimal_string(value, "fraud_loss_multiplier"),
        manual_review_cost=_optional_decimal_string(value, "manual_review_cost"),
        max_manual_review_rate=_optional_number(value, "max_manual_review_rate"),
        false_decline_cost=_optional_decimal_string(value, "false_decline_cost"),
    )


def _require_str(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_decimal_string(section: Mapping[str, object], key: str) -> Decimal | None:
    if key not in section:
        return None
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a decimal string")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{key} must be a valid decimal string") from error


def _optional_number(section: Mapping[str, object], key: str) -> float | None:
    if key not in section:
        return None
    value = section[key]
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be a number")
    return float(value)
