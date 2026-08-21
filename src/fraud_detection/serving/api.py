"""FastAPI factories for supplied-score and artifact-backed decisioning."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from fraud_detection.decisioning.inference import (
    SelectedDecisionPolicyConfig,
    decide_context_from_score,
    load_selected_decision_policy_config,
)
from fraud_detection.decisioning.reasons import (
    DecisionReasonConfig,
    load_decision_reason_config,
)
from fraud_detection.decisioning.scored_inference import (
    score_and_decide_provisional_transaction,
)
from fraud_detection.models.artifact import (
    LoadedProvisionalModel,
    load_provisional_model_artifact,
)
from fraud_detection.serving.analyst_demo import load_analyst_demo_html
from fraud_detection.serving.schemas import (
    DecisionRequest,
    DecisionResponse,
    ScoredDecisionRequest,
    ScoredDecisionResponse,
)

DEFAULT_POLICY_CONFIG_PATH = Path("configs/selected_decision_policy.toml")
DEFAULT_REASON_CONFIG_PATH = Path("configs/decision_reasons.toml")
DEFAULT_ARTIFACT_PATH = Path("models/xgboost_engineering_provisional.joblib")
DEFAULT_METADATA_PATH = Path("models/xgboost_engineering_provisional.metadata.json")
DEFAULT_XGBOOST_CONFIG_PATH = Path("configs/xgboost_baseline.toml")
DEFAULT_INPUT_DIRECTORY = Path("data/processed")
DEFAULT_FEATURE_DIRECTORY = Path("data/processed/features")


def create_app(
    policy_config_path: Path = DEFAULT_POLICY_CONFIG_PATH,
    reason_config_path: Path = DEFAULT_REASON_CONFIG_PATH,
) -> FastAPI:
    """Create the decision-only app and load immutable configuration once."""
    policy_config = load_selected_decision_policy_config(policy_config_path)
    reason_config = load_decision_reason_config(reason_config_path)
    return _build_app(policy_config, reason_config)


def create_scored_app(
    policy_config_path: Path = DEFAULT_POLICY_CONFIG_PATH,
    reason_config_path: Path = DEFAULT_REASON_CONFIG_PATH,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    xgboost_config_path: Path = DEFAULT_XGBOOST_CONFIG_PATH,
    input_directory: Path = DEFAULT_INPUT_DIRECTORY,
    feature_directory: Path = DEFAULT_FEATURE_DIRECTORY,
) -> FastAPI:
    """Create the scored app and validate all immutable runtime inputs once."""
    policy_config = load_selected_decision_policy_config(policy_config_path)
    reason_config = load_decision_reason_config(reason_config_path)
    loaded_model = load_provisional_model_artifact(
        artifact_path,
        metadata_path,
        xgboost_config_path,
        policy_config.score_source_label,
        input_directory,
        feature_directory,
    )
    return _build_app(policy_config, reason_config, loaded_model)


def _build_app(
    policy_config: SelectedDecisionPolicyConfig,
    reason_config: DecisionReasonConfig,
    loaded_model: LoadedProvisionalModel | None = None,
) -> FastAPI:
    """Register routes around already loaded, immutable runtime objects."""
    scored = loaded_model is not None
    application = FastAPI(
        title=(
            "Provisional Fraud Scored-decision API"
            if scored
            else "Provisional Fraud Decision API"
        ),
        version="0.1.0",
        description=(
            "Applies a frozen provisional model and policy to precomputed "
            "decision-time features. No request fits a model or builds "
            "temporal state."
            if scored
            else "Applies a frozen provisional policy to supplied risk scores. "
            "This endpoint does not score raw transactions or fit a model."
        ),
    )
    application.state.policy_config = policy_config
    application.state.reason_config = reason_config
    application.state.loaded_model = loaded_model

    @application.get("/healthz", include_in_schema=False)
    def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "mode": "scored" if scored else "decision-only",
        }

    @application.post(
        "/decisions",
        response_model=DecisionResponse,
        summary="Apply the frozen decision policy to a supplied score",
    )
    def create_decision(request: DecisionRequest) -> DecisionResponse:
        result = decide_context_from_score(
            request.to_context(),
            request.risk_score,
            policy_config,
            reason_config,
        )
        return DecisionResponse.from_result(result)

    if loaded_model is not None:

        @application.get(
            "/analyst-demo",
            response_class=HTMLResponse,
            include_in_schema=False,
        )
        def analyst_demo() -> HTMLResponse:
            return HTMLResponse(
                load_analyst_demo_html(),
                headers={
                    "Content-Security-Policy": (
                        "default-src 'self'; script-src 'unsafe-inline'; "
                        "style-src 'unsafe-inline'; connect-src 'self'; "
                        "img-src 'self' data:; object-src 'none'; "
                        "base-uri 'none'; frame-ancestors 'none'"
                    ),
                    "X-Content-Type-Options": "nosniff",
                },
            )

        @application.post(
            "/scored-decisions",
            response_model=ScoredDecisionResponse,
            summary="Score precomputed features and apply the frozen policy",
        )
        def create_scored_decision(
            request: ScoredDecisionRequest,
        ) -> ScoredDecisionResponse:
            result = score_and_decide_provisional_transaction(
                loaded_model,
                request.to_model_input(),
                policy_config,
                reason_config,
            )
            return ScoredDecisionResponse.from_result(result)

    return application


app = create_app()
