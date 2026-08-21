"""Integration test for validation-only decision-policy cost sensitivity."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.decisioning.policy_sensitivity_cli import main
from fraud_detection.features.temporal import (
    build_processed_temporal_features,
    load_temporal_feature_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_cli_reuses_validation_for_ordered_deterministic_scenarios(
    tmp_path: Path,
) -> None:
    """The command must preserve base config and expose no test-tuning route."""
    input_directory = tmp_path / "processed"
    feature_directory = tmp_path / "features"
    output_path = tmp_path / "sensitivity.md"
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
    cost_config_path = PROJECT_ROOT / "configs" / "business_costs.toml"
    original_cost_config = cost_config_path.read_bytes()
    arguments = [
        "--input-directory",
        str(input_directory),
        "--feature-directory",
        str(feature_directory),
        "--xgboost-config",
        str(PROJECT_ROOT / "configs" / "xgboost_baseline.toml"),
        "--policy-config",
        str(PROJECT_ROOT / "configs" / "decision_policy.toml"),
        "--sensitivity-config",
        str(PROJECT_ROOT / "configs" / "decision_policy_sensitivity.toml"),
        "--cost-config",
        str(cost_config_path),
        "--output",
        str(output_path),
    ]

    first_exit_code = main(arguments)
    first_markdown = output_path.read_text(encoding="utf-8")
    second_exit_code = main(arguments)
    second_markdown = output_path.read_text(encoding="utf-8")

    assert first_exit_code == second_exit_code == 0
    assert first_markdown == second_markdown
    assert cost_config_path.read_bytes() == original_cost_config
    assert first_markdown.startswith("# Phase 4 Decision-policy Cost Sensitivity\n")
    assert "one shared score set" in first_markdown
    assert "for chronological validation" in first_markdown
    assert "The one-time test split is not scored" in first_markdown
    assert "XGBoost remains unpromoted" in first_markdown
    scenario_positions = tuple(
        first_markdown.index(f"| {name} |")
        for name in (
            "Base assumptions",
            "Higher fraud loss",
            "Lower review cost",
            "Higher false-decline cost",
            "Tighter review capacity",
        )
    )
    assert scenario_positions == tuple(sorted(scenario_positions))
    assert "Every selected policy satisfies" in first_markdown
    assert "| Test |" not in first_markdown


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
