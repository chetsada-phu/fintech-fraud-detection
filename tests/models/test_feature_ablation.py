"""Integration tests for validation-only XGBoost feature ablation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.features.temporal import (
    build_processed_temporal_features,
    load_temporal_feature_config,
)
from fraud_detection.models.feature_ablation import (
    FeatureAblationConfig,
    FeatureVariant,
    load_feature_ablation_config,
)
from fraud_detection.models.feature_ablation_cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "xgboost_feature_ablation.toml"
TERMINAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "xgboost_terminal_challenger.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_versioned_feature_ablation_contract_is_validation_only() -> None:
    assert load_feature_ablation_config(CONFIG_PATH) == FeatureAblationConfig(
        evaluation_split="validation",
        baseline_variant_key="full_baseline",
        minimum_ap_improvement=0.01,
        variants=(
            FeatureVariant("base_only", "Base only", False, False),
            FeatureVariant("base_temporal", "Base + temporal", True, False),
            FeatureVariant("base_ids", "Base + synthetic IDs", False, True),
            FeatureVariant("full_baseline", "Frozen full baseline", True, True),
        ),
    )


def test_versioned_terminal_challenger_contract_is_validation_only() -> None:
    assert load_feature_ablation_config(TERMINAL_CONFIG_PATH) == (
        FeatureAblationConfig(
            evaluation_split="validation",
            baseline_variant_key="full_baseline",
            minimum_ap_improvement=0.01,
            variants=(
                FeatureVariant(
                    "customer_temporal",
                    "Portable customer history",
                    True,
                    False,
                    False,
                ),
                FeatureVariant(
                    "customer_terminal_temporal",
                    "Portable customer + terminal history",
                    True,
                    False,
                    True,
                ),
                FeatureVariant(
                    "full_baseline",
                    "Frozen full baseline",
                    True,
                    True,
                    False,
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    ("config_path", "expected_variants"),
    (
        (
            CONFIG_PATH,
            (
                "Base only",
                "Base + temporal",
                "Base + synthetic IDs",
                "Frozen full baseline",
            ),
        ),
        (
            TERMINAL_CONFIG_PATH,
            (
                "Portable customer history",
                "Portable customer + terminal history",
                "Frozen full baseline",
            ),
        ),
    ),
)
def test_cli_compares_feature_groups_without_scoring_test(
    tmp_path: Path, config_path: Path, expected_variants: tuple[str, ...]
) -> None:
    input_directory = tmp_path / "processed"
    feature_directory = tmp_path / "features"
    output_path = tmp_path / "ablation.md"
    input_directory.mkdir()
    transactions = _transactions()
    write_transactions_csv(transactions[:10], input_directory / "train.csv")
    write_transactions_csv(transactions[10:14], input_directory / "validation.csv")
    write_transactions_csv(transactions[14:], input_directory / "test.csv")
    build_processed_temporal_features(
        input_directory,
        feature_directory,
        load_temporal_feature_config(
            PROJECT_ROOT / "configs" / "temporal_features.toml"
        ),
    )

    exit_code = main(
        [
            "--input-directory",
            str(input_directory),
            "--feature-directory",
            str(feature_directory),
            "--xgboost-config",
            str(PROJECT_ROOT / "configs" / "xgboost_baseline.toml"),
            "--ablation-config",
            str(config_path),
            "--cost-config",
            str(PROJECT_ROOT / "configs" / "business_costs.toml"),
            "--output",
            str(output_path),
        ]
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert markdown.startswith("# Phase 3 XGBoost Feature Ablation\n")
    assert "scores only validation" in markdown
    for variant in expected_variants:
        assert f"| {variant} |" in markdown
    assert "## Feature Direction Decision" in markdown
    assert "| Test |" not in markdown


def test_feature_ablation_rejects_non_validation_contract() -> None:
    with pytest.raises(ValueError, match="evaluation_split"):
        FeatureAblationConfig(
            evaluation_split="test",
            baseline_variant_key="full",
            minimum_ap_improvement=0.01,
            variants=(
                FeatureVariant("simple", "Simple", False, False),
                FeatureVariant("full", "Full", True, True),
            ),
        )


def _transactions() -> tuple[Transaction, ...]:
    fraud_scenarios = {2: 1, 7: 2, 11: 3, 16: 1}
    return tuple(
        Transaction(
            transaction_id=index,
            tx_datetime=START + timedelta(hours=index * 4),
            customer_id=index % 5,
            terminal_id=index % 4,
            tx_amount=(
                Decimal("260.00")
                if index in fraud_scenarios
                else Decimal("40.00") + index
            ),
            tx_time_seconds=index * 4 * 3_600,
            tx_time_days=(index * 4 * 3_600) // SECONDS_PER_DAY,
            tx_fraud=int(index in fraud_scenarios),
            tx_fraud_scenario=fraud_scenarios.get(index, 0),
        )
        for index in range(18)
    )
