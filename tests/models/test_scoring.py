"""Tests for label-free, single-record provisional model scoring."""

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from fraud_detection.data.schema import SECONDS_PER_DAY, Transaction
from fraud_detection.features.matrix import JoinedFeatureRow, join_transaction_features
from fraud_detection.features.temporal import (
    TemporalFeatureConfig,
    build_temporal_features,
)
from fraud_detection.models.artifact import (
    ARTIFACT_VERSION,
    LoadedProvisionalModel,
    ProvisionalModelMetadata,
    TrainingProvenance,
)
from fraud_detection.models.model_input import (
    XGBOOST_FEATURE_CONTRACT,
    ProvisionalModelInput,
)
from fraud_detection.models.scoring import score_provisional_model
from fraud_detection.models.xgboost_model import (
    fit_xgboost_baseline,
    load_xgboost_config,
    predict_xgboost_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "xgboost_baseline.toml"
START = datetime(2018, 4, 1, tzinfo=UTC)
TEMPORAL_CONFIG = TemporalFeatureConfig(3_600, 86_400, 4)


def test_single_record_score_matches_existing_batch_scorer() -> None:
    """The adapter and existing batch path must share one feature mapping."""
    rows = _joined_rows()
    pipeline = fit_xgboost_baseline(
        rows[:30],
        load_xgboost_config(CONFIG_PATH),
    )
    held_out = rows[30]

    result = score_provisional_model(
        _loaded_model(pipeline),
        ProvisionalModelInput.from_joined_feature_row(held_out),
    )

    assert result.risk_score == pytest.approx(
        predict_xgboost_scores(pipeline, (held_out,))[0],
        abs=0.0,
    )
    assert result.transaction_id == held_out.transaction.transaction_id
    assert result.score_source_label == "provisional test score source"
    assert result.artifact_version == ARTIFACT_VERSION
    assert result.artifact_sha256 == "1" * 64
    assert result.xgboost_config_sha256 == "2" * 64


def test_model_input_maps_the_exact_frozen_feature_order() -> None:
    """Every named contract field must occupy its declared model position."""
    model_input = ProvisionalModelInput(
        transaction_id=42,
        tx_amount=Decimal("125.50"),
        tx_time_days=2,
        tx_datetime=datetime(2018, 4, 3, 6, tzinfo=UTC),
        customer_tx_count_short_window=3,
        customer_tx_count_long_window=7,
        customer_amount_mean_prior=Decimal("100.00"),
        customer_amount_deviation_from_mean_prior=Decimal("25.50"),
        customer_seconds_since_previous=60,
        customer_id=12,
        terminal_id=8,
    )

    mapped = dict(
        zip(
            XGBOOST_FEATURE_CONTRACT,
            model_input.to_feature_values(),
            strict=True,
        )
    )

    assert tuple(mapped) == XGBOOST_FEATURE_CONTRACT
    assert mapped["TX_AMOUNT"] == 125.5
    assert mapped["TX_TIME_DAYS"] == 2.0
    assert mapped["TX_HOUR_SIN"] == pytest.approx(1.0)
    assert mapped["TX_HOUR_COS"] == pytest.approx(0.0, abs=1e-15)
    assert mapped["CUSTOMER_TX_COUNT_SHORT_WINDOW"] == 3.0
    assert mapped["CUSTOMER_TX_COUNT_LONG_WINDOW"] == 7.0
    assert mapped["CUSTOMER_AMOUNT_MEAN_PRIOR"] == 100.0
    assert mapped["CUSTOMER_AMOUNT_DEVIATION_FROM_MEAN_PRIOR"] == 25.5
    assert mapped["CUSTOMER_SECONDS_SINCE_PREVIOUS"] == 60.0
    assert mapped["CUSTOMER_ID"] == "customer_12"
    assert mapped["TERMINAL_ID"] == "terminal_8"


def test_missing_history_and_unseen_ids_are_scoreable() -> None:
    """Native missing values and one-hot unknown handling must remain usable."""
    rows = _joined_rows()
    pipeline = fit_xgboost_baseline(
        rows[:30],
        load_xgboost_config(CONFIG_PATH),
    )
    model_input = ProvisionalModelInput(
        transaction_id=100,
        tx_amount=Decimal("75.00"),
        tx_time_days=5,
        tx_datetime=datetime(2018, 4, 6, 12, tzinfo=UTC),
        customer_tx_count_short_window=0,
        customer_tx_count_long_window=0,
        customer_amount_mean_prior=None,
        customer_amount_deviation_from_mean_prior=None,
        customer_seconds_since_previous=None,
        customer_id=999,
        terminal_id=888,
    )

    result = score_provisional_model(_loaded_model(pipeline), model_input)

    assert 0 <= result.risk_score <= 1


def test_input_contract_has_no_labels_and_relabeling_cannot_change_score() -> None:
    """Offline outcome labels must not enter the scoring input or result."""
    rows = _joined_rows()
    pipeline = fit_xgboost_baseline(
        rows[:30],
        load_xgboost_config(CONFIG_PATH),
    )
    original = rows[30]
    relabeled = replace(
        original,
        transaction=replace(
            original.transaction,
            tx_fraud=1 - original.transaction.tx_fraud,
            tx_fraud_scenario=(0 if original.transaction.tx_fraud else 3),
        ),
    )

    original_input = ProvisionalModelInput.from_joined_feature_row(original)
    relabeled_input = ProvisionalModelInput.from_joined_feature_row(relabeled)

    assert {field.name for field in fields(ProvisionalModelInput)}.isdisjoint(
        {"tx_fraud", "tx_fraud_scenario"}
    )
    assert original_input == relabeled_input
    assert score_provisional_model(
        _loaded_model(pipeline),
        original_input,
    ).risk_score == pytest.approx(
        score_provisional_model(
            _loaded_model(pipeline),
            relabeled_input,
        ).risk_score,
        abs=0.0,
    )


def test_scoring_never_fits_the_preloaded_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request-time scoring must call prediction only on the existing fit."""
    rows = _joined_rows()
    pipeline = fit_xgboost_baseline(
        rows[:30],
        load_xgboost_config(CONFIG_PATH),
    )

    def unexpected_fit(*args: object, **kwargs: object) -> None:
        pytest.fail(f"pipeline.fit was called during scoring: {args}, {kwargs}")

    monkeypatch.setattr(pipeline, "fit", unexpected_fit)

    score_provisional_model(
        _loaded_model(pipeline),
        ProvisionalModelInput.from_joined_feature_row(rows[30]),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"tx_amount": Decimal("0.00")}, "finite and positive"),
        (
            {
                "customer_tx_count_short_window": 2,
                "customer_tx_count_long_window": 1,
            },
            "cannot exceed",
        ),
        ({"customer_amount_mean_prior": None}, "missing together"),
        ({"tx_datetime": datetime(2018, 4, 3, 6)}, "timezone-aware UTC"),
    ),
)
def test_model_input_rejects_impossible_or_partial_values(
    changes: dict[str, object],
    message: str,
) -> None:
    """Invalid decision-time fields must fail before pipeline prediction."""
    values: dict[str, object] = {
        "transaction_id": 42,
        "tx_amount": Decimal("125.50"),
        "tx_time_days": 2,
        "tx_datetime": datetime(2018, 4, 3, 6, tzinfo=UTC),
        "customer_tx_count_short_window": 1,
        "customer_tx_count_long_window": 3,
        "customer_amount_mean_prior": Decimal("100.00"),
        "customer_amount_deviation_from_mean_prior": Decimal("25.50"),
        "customer_seconds_since_previous": 60,
        "customer_id": 12,
        "terminal_id": 8,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        ProvisionalModelInput(**values)  # type: ignore[arg-type]


def _loaded_model(pipeline: Pipeline) -> LoadedProvisionalModel:
    return LoadedProvisionalModel(
        pipeline=pipeline,
        metadata=ProvisionalModelMetadata(
            artifact_version=ARTIFACT_VERSION,
            artifact_sha256="1" * 64,
            xgboost_config_sha256="2" * 64,
            score_source_label="provisional test score source",
            feature_contract=XGBOOST_FEATURE_CONTRACT,
            training_provenance=TrainingProvenance(
                split="train",
                transaction_rows=30,
                fraud_rows=5,
                transactions_sha256="3" * 64,
                temporal_features_sha256="4" * 64,
            ),
        ),
    )


def _joined_rows() -> tuple[JoinedFeatureRow, ...]:
    transactions = _transactions()
    return join_transaction_features(
        transactions,
        build_temporal_features(transactions, TEMPORAL_CONFIG),
    )


def _transactions() -> tuple[Transaction, ...]:
    fraud_ids = {3, 9, 17, 25, 31}
    return tuple(
        Transaction(
            transaction_id=index,
            tx_datetime=START + timedelta(minutes=index * 45),
            customer_id=index % 6,
            terminal_id=index % 4,
            tx_amount=(
                Decimal("260.00") if index in fraud_ids else Decimal("30.00") + index
            ),
            tx_time_seconds=index * 45 * 60,
            tx_time_days=(index * 45 * 60) // SECONDS_PER_DAY,
            tx_fraud=int(index in fraud_ids),
            tx_fraud_scenario=int(index in fraud_ids),
        )
        for index in range(36)
    )
