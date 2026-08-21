"""Integration test for validation-only decision-policy selection."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from fraud_detection.data.generator import write_transactions_csv
from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.decisioning.policy_cli import main
from fraud_detection.features.temporal import (
    build_processed_temporal_features,
    load_temporal_feature_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
START = datetime(2018, 4, 1, tzinfo=UTC)


def test_cli_selects_policy_on_validation_without_scoring_test(
    tmp_path: Path,
) -> None:
    """The Phase 4 command must expose no route to test-based threshold tuning."""
    input_directory = tmp_path / "processed"
    feature_directory = tmp_path / "features"
    output_path = tmp_path / "policy.md"
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
            "--policy-config",
            str(PROJECT_ROOT / "configs" / "decision_policy.toml"),
            "--cost-config",
            str(PROJECT_ROOT / "configs" / "business_costs.toml"),
            "--reason-config",
            str(PROJECT_ROOT / "configs" / "decision_reasons.toml"),
            "--output",
            str(output_path),
        ]
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert markdown.startswith("# Phase 4 Provisional Decision Policy\n")
    assert "only chronological\nvalidation" in markdown
    assert "The one-time test split is not scored" in markdown
    assert "XGBoost engineering-only provisional score source" in markdown
    assert "XGBoost remains unpromoted" in markdown
    assert "Manual-review capacity: **satisfied**" in markdown
    assert "## Deterministic Decision Reasons" in markdown
    assert "Fraud labels, fraud scenarios, and future rows" in markdown
    assert "`HIGH_TRANSACTION_AMOUNT`" in markdown
    assert "`RISK_SCORE_REVIEW`" in markdown
    assert "`RISK_SCORE_DECLINE`" in markdown
    assert "| Test |" not in markdown


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
