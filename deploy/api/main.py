"""
Enhanced FastAPI endpoint with authentication, monitoring, and rate limiting.

This is an improved version of the UAIS-V API with:
- API key authentication
- Prometheus metrics
- Health checks
- Rate limiting
- Better error handling
- Request validation
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from base64 import b64decode
from binascii import Error as Base64DecodeError
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from . import scope_guard
from .auth import API_KEYS, authenticate
from .kga_routes import router as kga_router
from .model_governance import router as model_governance_router
from .rate_limit import REDIS_URL, RateLimitMiddleware

logger = logging.getLogger(__name__)

# Monitoring is optional; authentication is mandatory and must fail at startup if missing.
try:
    from .monitoring import (
        MODEL_LOADED,
        InferenceMetrics,
        MetricsMiddleware,
        export_prometheus_metrics,
        get_system_metrics,
        health_checker,
    )

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


def _csv_env(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


CORS_ORIGINS = _csv_env("UAIS_CORS_ORIGINS")
MAX_FEATURES = _int_env("UAIS_MAX_FEATURES", 4096)
MAX_SCORES = _int_env("UAIS_MAX_SCORES", 64)
MAX_DOMAINS = _int_env("UAIS_MAX_DOMAINS", 32)
MAX_EMBEDDINGS = _int_env("UAIS_MAX_EMBEDDINGS", 8192)
MAX_TEXT_CHARS = _int_env("UAIS_MAX_TEXT_CHARS", 10000)
MAX_IMAGE_BASE64_CHARS = _int_env("UAIS_MAX_IMAGE_BASE64_CHARS", 8_000_000)
MAX_IMAGE_PIXELS = _int_env("UAIS_MAX_IMAGE_PIXELS", 25_000_000)
RATE_LIMIT_REQUESTS = _int_env("UAIS_RATE_LIMIT_REQUESTS", 120)
RATE_LIMIT_WINDOW_SECONDS = _int_env("UAIS_RATE_LIMIT_WINDOW_SECONDS", 60)
REQUEST_TIMEOUT_SECONDS = _int_env("UAIS_REQUEST_TIMEOUT_SECONDS", 30)
PRODUCTION_MODE = _bool_env("UAIS_PRODUCTION_MODE", False)
REQUIRED_MODELS = tuple(_csv_env("UAIS_REQUIRED_MODELS") or ["fraud", "cyber", "fusion"])
TRUSTED_MODEL_ARTIFACTS = _bool_env("UAIS_TRUSTED_MODEL_ARTIFACTS", False)
REQUIRE_MODEL_CHECKSUMS = _bool_env("UAIS_REQUIRE_MODEL_CHECKSUMS", True)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with explicit timezone information."""
    return datetime.now(timezone.utc).isoformat()


def runtime_mode() -> str:
    return "production" if PRODUCTION_MODE else "development"


def production_config_errors() -> list[str]:
    """Return production-blocking configuration errors without exposing secrets."""
    errors: list[str] = []
    if not API_KEYS:
        errors.append("UAIS_API_KEYS must be configured with at least one API key")
    if PRODUCTION_MODE and not CORS_ORIGINS:
        errors.append("UAIS_CORS_ORIGINS must be configured with explicit origins in production mode")
    if any(origin == "*" for origin in CORS_ORIGINS):
        errors.append("UAIS_CORS_ORIGINS cannot use a wildcard origin in production")
    if PRODUCTION_MODE and not REQUIRE_MODEL_CHECKSUMS:
        errors.append("UAIS_REQUIRE_MODEL_CHECKSUMS must remain true in production mode")
    return errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail production startup if required security configuration is invalid."""
    if PRODUCTION_MODE:
        errors = production_config_errors()
        if errors:
            raise RuntimeError("Invalid production configuration: " + "; ".join(errors))
    yield


# FastAPI app
app = FastAPI(
    title="UAIS-V Enhanced API",
    version="2.0",
    description="Universal Anomaly Intelligence System with authentication and monitoring",
    lifespan=lifespan,
)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Bound per-request wall-clock so a slow inference can't pin a worker (P14)."""

    def __init__(self, app, timeout_seconds: int):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "Request timed out"},
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured request logging with correlation IDs and no payload logging."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            logger.exception(
                "request_failed request_id=%s method=%s path=%s status=500 duration_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


app.add_middleware(
    RateLimitMiddleware,
    limit=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)
app.add_middleware(TimeoutMiddleware, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
app.add_middleware(RequestLoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=bool(CORS_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# Add metrics middleware if available
if MONITORING_AVAILABLE:
    app.add_middleware(MetricsMiddleware)

# KGA certificate routes + model governance (model_version / rollback)
app.include_router(kga_router)
app.include_router(model_governance_router)

# Model paths
project_root = Path(__file__).resolve().parents[2]
fraud_model_path = project_root / "models" / "fraud" / "supervised" / "fraud_model.pkl"
cyber_model_path = project_root / "models" / "cyber" / "supervised" / "cyber_model.pkl"
fusion_model_path = project_root / "experiments" / "fusion" / "models" / "fusion_meta_model.pkl"
attention_model_dir = project_root / "models" / "fusion" / "attention"
attention_ckpt_path = attention_model_dir / "attention_fusion.pt"
nlp_model_dir = project_root / "models" / "nlp" / "distilbert"
vision_model_dir = project_root / "models" / "vision" / "resnet"

MODEL_ARTIFACT_PATHS = {
    "fraud": fraud_model_path,
    "cyber": cyber_model_path,
    "fusion": fusion_model_path,
    "attention_fusion": attention_ckpt_path,
    "nlp": nlp_model_dir / "model.pt",
    "vision": vision_model_dir / "model.pt",
}


def _artifact_checksum_env(model_type: str) -> str | None:
    return os.getenv(f"UAIS_MODEL_SHA256_{model_type.upper()}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_is_trusted(path: Path, model_type: str) -> bool:
    if not path.exists() or not TRUSTED_MODEL_ARTIFACTS:
        return False
    expected = _artifact_checksum_env(model_type)
    if expected:
        return _sha256(path).lower() == expected.strip().lower()
    return not REQUIRE_MODEL_CHECKSUMS


def _load_joblib_artifact(path: Path, model_type: str):
    if not _artifact_is_trusted(path, model_type):
        return None
    return joblib.load(path)


def _load_torch_artifact(torch_module, path: Path, model_type: str):
    if not _artifact_is_trusted(path, model_type):
        return None
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return None


# Load models
fraud_model = _load_joblib_artifact(fraud_model_path, "fraud")
cyber_model = _load_joblib_artifact(cyber_model_path, "cyber")
fusion_model = _load_joblib_artifact(fusion_model_path, "fusion")

# Track model loading status
if MONITORING_AVAILABLE:
    MODEL_LOADED.labels(model_type="fraud").set(1 if fraud_model else 0)
    MODEL_LOADED.labels(model_type="cyber").set(1 if cyber_model else 0)
    MODEL_LOADED.labels(model_type="fusion").set(1 if fusion_model else 0)

# Lazy-loaded artifacts
_nlp_artifacts_loaded = False
_nlp_model = None
_nlp_tokenizer = None
_vision_model = None
_attention_loaded = False
_attention_model = None
_attention_meta = {}


def _loaded_model_status(model_type: str) -> bool:
    if model_type == "fraud":
        return fraud_model is not None
    if model_type == "cyber":
        return cyber_model is not None
    if model_type == "fusion":
        return fusion_model is not None
    if model_type == "attention_fusion":
        return _attention_model is not None
    if model_type == "nlp":
        return _nlp_model is not None and _nlp_tokenizer is not None
    if model_type == "vision":
        return bool(_vision_model)
    return False


def model_statuses() -> dict[str, dict[str, object]]:
    statuses: dict[str, dict[str, object]] = {}
    for model_type, path in MODEL_ARTIFACT_PATHS.items():
        statuses[model_type] = {
            "required": model_type in REQUIRED_MODELS,
            "available": _loaded_model_status(model_type),
            "artifact_path": str(path.relative_to(project_root)),
            "artifact_exists": path.exists(),
            "artifact_trusted": _artifact_is_trusted(path, model_type),
        }
    return statuses


def readiness_report() -> dict[str, object]:
    config_errors = production_config_errors()
    models = model_statuses()
    missing_models = [model_type for model_type in REQUIRED_MODELS if not models.get(model_type, {}).get("available")]
    monitoring_ready = MONITORING_AVAILABLE
    ready = not config_errors and not missing_models and monitoring_ready
    return {
        "ready": ready,
        "mode": runtime_mode(),
        "timestamp": utc_timestamp(),
        "checks": {
            "configuration": {
                "status": "pass" if not config_errors else "fail",
                "errors": config_errors,
            },
            "models": {
                "status": "pass" if not missing_models else "fail",
                "required": list(REQUIRED_MODELS),
                "missing": missing_models,
            },
            "monitoring": {
                "status": "pass" if monitoring_ready else "fail",
            },
        },
        "models": models,
    }


def model_unavailable(model_type: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "model_unavailable",
            "model_type": model_type,
            "message": f"{model_type} model is not production-ready or not loaded",
        },
    )


def _load_nlp():
    """Lazy load NLP model and tokenizer when a production NLP artifact exists."""
    global _nlp_artifacts_loaded, _nlp_model, _nlp_tokenizer
    if _nlp_artifacts_loaded:
        return _nlp_model, _nlp_tokenizer
    if not nlp_model_dir.exists():
        _nlp_artifacts_loaded = True
        return None, None
    state_path = nlp_model_dir / "model.pt"
    if not state_path.exists():
        _nlp_artifacts_loaded = True
        return None, None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        _nlp_artifacts_loaded = True
        return None, None
    state = _load_torch_artifact(torch, state_path, "nlp")
    if state is None:
        _nlp_artifacts_loaded = True
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        str(nlp_model_dir), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
        str(nlp_model_dir), num_labels=2, local_files_only=True
    )
    model.load_state_dict(state)
    model.eval()
    _nlp_model, _nlp_tokenizer = model, tokenizer
    _nlp_artifacts_loaded = True
    if MONITORING_AVAILABLE:
        MODEL_LOADED.labels(model_type="nlp").set(1)
    return model, tokenizer


def _load_vision():
    """Lazy load vision model."""
    global _vision_model
    if _vision_model is False:
        return None
    if _vision_model is not None:
        return _vision_model
    state_path = vision_model_dir / "model.pt"
    if not state_path.exists():
        _vision_model = False
        return None
    try:
        import torch
        from uais.vision.vision_resnet import VisionConfig, build_resnet_classifier
    except Exception:
        _vision_model = False
        return None
    cfg = VisionConfig(model_name="resnet18", num_classes=2, pretrained=False)
    model = build_resnet_classifier(cfg)
    state = _load_torch_artifact(torch, state_path, "vision")
    if state is None:
        _vision_model = False
        return None
    model.load_state_dict(state)
    model.eval()
    _vision_model = model
    if MONITORING_AVAILABLE:
        MODEL_LOADED.labels(model_type="vision").set(1)
    return model


def _load_attention_fusion():
    """Lazy load attention fusion model."""
    global _attention_loaded, _attention_model, _attention_meta
    if _attention_loaded:
        return _attention_model, _attention_meta
    _attention_loaded = True
    if not attention_ckpt_path.exists():
        return None, {}
    try:
        import torch

        from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
    except Exception:
        return None, {}

    state = _load_torch_artifact(torch, attention_ckpt_path, "attention_fusion")
    if not isinstance(state, dict):
        return None, {}
    config = state.get("config", {}) if isinstance(state, dict) else {}
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    domain_order = state.get("domain_order") or model_cfg.get("domain_order")
    feature_columns = state.get("feature_columns")
    if not domain_order or not feature_columns:
        return None, {}
    confidence_column = data_cfg.get("confidence_column", "confidence")
    use_input_confidence = bool(model_cfg.get("use_input_confidence", True))
    confidence_index = (
        feature_columns.index(confidence_column)
        if use_input_confidence and confidence_column in feature_columns
        else None
    )

    model = AttentionFusionModel(
        num_domains=len(domain_order),
        input_dim=len(feature_columns),
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_confidence=bool(model_cfg.get("use_confidence", True)),
        use_input_confidence=use_input_confidence,
        confidence_index=confidence_index,
        use_attention=bool(model_cfg.get("use_attention", True)),
        use_domain_embeddings=bool(model_cfg.get("use_domain_embeddings", True)),
        use_positional_embeddings=bool(model_cfg.get("use_positional_embeddings", True)),
        use_missing_embedding=bool(model_cfg.get("use_missing_embedding", True)),
    )
    model.load_state_dict(state["model_state"])
    model.eval()
    _attention_model = model
    _attention_meta = {
        "domain_order": list(domain_order),
        "feature_columns": list(feature_columns),
        "score_column": data_cfg.get("score_column", "score"),
        "confidence_column": confidence_column,
        "confidence_index": confidence_index,
    }
    if MONITORING_AVAILABLE:
        MODEL_LOADED.labels(model_type="attention_fusion").set(1)
    return model, _attention_meta


# Request/Response models
class FraudRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=1,
        max_length=MAX_FEATURES,
        description="Feature vector for fraud detection",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Features must be finite numbers")
        return v


class FraudResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    confidence: float


class CyberRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=1,
        max_length=MAX_FEATURES,
        description="Feature vector for cyber attack detection",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Features must be finite numbers")
        return v


class CyberResponse(BaseModel):
    cyber_attack_probability: float = Field(..., ge=0.0, le=1.0)
    threat_level: str


class FusionRequest(BaseModel):
    scores: dict[str, float] = Field(
        ...,
        min_length=1,
        max_length=MAX_SCORES,
        description="Domain-specific anomaly scores",
    )

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v):
        for score in v.values():
            if not np.isfinite(score) or not (0 <= score <= 1):
                raise ValueError("Scores must be between 0 and 1")
        return v


class FusionResponse(BaseModel):
    fusion_risk: float
    domains: list[str]
    overall_risk_level: str
    scope: Optional[dict] = None  # out-of-envelope / drift annotation (P12)


class AttentionDomainInput(BaseModel):
    domain: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    embeddings: Optional[list[float]] = Field(None, max_length=MAX_EMBEDDINGS)

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v):
        if v is not None and any(not np.isfinite(x) for x in v):
            raise ValueError("Embeddings must be finite numbers")
        return v


class AttentionFusionRequest(BaseModel):
    domains: list[AttentionDomainInput] = Field(..., min_length=1, max_length=MAX_DOMAINS)


class AttentionFusionResponse(BaseModel):
    fusion_risk: float
    domain_order: list[str]
    attention_mean: Optional[list[float]]
    domain_confidence: Optional[list[float]]


class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS, description="Text to analyze")


class VisionRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=1,
        max_length=MAX_IMAGE_BASE64_CHARS,
        description="Base64-encoded image (jpg/png)",
    )


# Helper functions
def get_risk_level(probability: float) -> str:
    """Categorize risk based on probability."""
    if probability < 0.3:
        return "low"
    elif probability < 0.6:
        return "medium"
    elif probability < 0.8:
        return "high"
    else:
        return "critical"


def _build_attention_features(req: AttentionFusionRequest, meta: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    domain_order = meta.get("domain_order", [])
    feature_columns = meta.get("feature_columns", [])
    score_column = meta.get("score_column", "score")
    confidence_column = meta.get("confidence_column", "confidence")

    domain_map = {item.domain: item for item in req.domains}
    num_domains = len(domain_order)
    feature_dim = len(feature_columns)
    features = np.zeros((1, num_domains, feature_dim), dtype=np.float32)
    mask = np.ones((1, num_domains), dtype=bool)

    embedding_positions = [idx for idx, col in enumerate(feature_columns) if col.startswith("embedding_")]
    for domain_idx, domain in enumerate(domain_order):
        if domain not in domain_map:
            continue
        mask[0, domain_idx] = False
        item = domain_map[domain]
        for col_idx, col in enumerate(feature_columns):
            if col == score_column:
                features[0, domain_idx, col_idx] = float(item.score)
            elif col == confidence_column:
                features[0, domain_idx, col_idx] = float(item.confidence) if item.confidence is not None else 1.0
        if embedding_positions:
            values = item.embeddings or []
            for pos_idx, col_idx in enumerate(embedding_positions):
                if pos_idx < len(values):
                    features[0, domain_idx, col_idx] = float(values[pos_idx])
    return features, mask


# Health checks
if MONITORING_AVAILABLE:
    health_checker.register_check("fraud_model", lambda: fraud_model is not None)
    health_checker.register_check("cyber_model", lambda: cyber_model is not None)
    health_checker.register_check("fusion_model", lambda: fusion_model is not None)


# Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "UAIS-V Enhanced API",
        "version": "2.0",
        "status": "active",
        "features": {
            "authentication": True,
            "monitoring": MONITORING_AVAILABLE,
        },
        "available_models": {
            model_type: status_info["available"] for model_type, status_info in model_statuses().items()
        },
    }


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok", "timestamp": utc_timestamp()}


@app.get("/scope")
async def scope(authenticated: bool = Depends(authenticate)):
    """Declared validated operating envelope + live drift-guard status (P12/P13)."""
    return {
        "contract": "deploy/SCOPE_CONTRACT.md",
        "validated_regime": "in-domain + differential-reliability stress (RGB+depth/IR/3D); "
                            "clean cross-domain transfer is NOT a validated superiority claim (T9)",
        "reference_envelope_loaded": scope_guard.reference_loaded(),
        "drift_threshold": scope_guard._DRIFT_THRESHOLD,
        "note": "Responses are annotated with an out-of-envelope drift flag; "
                "out-of-envelope traffic should be alerted/held, not trusted blindly.",
    }


@app.get("/ready")
async def ready(authenticated: bool = Depends(authenticate)):
    """Readiness check for authenticated production traffic."""
    report = readiness_report()
    return JSONResponse(
        status_code=status.HTTP_200_OK if report["ready"] else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=report,
    )


@app.get("/health/detailed")
async def health_detailed(authenticated: bool = Depends(authenticate)):
    """Detailed health check with component status."""
    if not MONITORING_AVAILABLE:
        return {"status": "ok", "note": "Detailed monitoring not available"}

    return await health_checker.run_checks()


@app.get("/metrics")
async def metrics(authenticated: bool = Depends(authenticate)):
    """Prometheus metrics endpoint."""
    if not MONITORING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Metrics not available. Install prometheus_client."
        )

    metrics_data = await export_prometheus_metrics()
    return PlainTextResponse(metrics_data, media_type="text/plain")


@app.get("/system")
async def system_info(authenticated: bool = Depends(authenticate)):
    """System resource information."""
    if not MONITORING_AVAILABLE:
        return {"note": "System metrics not available"}

    return get_system_metrics()


@app.post("/predict_fraud", response_model=FraudResponse)
async def predict_fraud(req: FraudRequest, authenticated: bool = Depends(authenticate)):
    """Predict fraud probability from features."""
    if fraud_model is None:
        raise model_unavailable("fraud")

    try:
        if MONITORING_AVAILABLE:
            with InferenceMetrics("fraud") as metrics:
                X = np.array(req.features).reshape(1, -1)
                proba = float(fraud_model.predict_proba(X)[0, 1])
                metrics.record_prediction(proba)
        else:
            X = np.array(req.features).reshape(1, -1)
            proba = float(fraud_model.predict_proba(X)[0, 1])

        return FraudResponse(
            fraud_probability=proba,
            risk_level=get_risk_level(proba),
            confidence=abs(proba - 0.5) * 2,  # 0 at 0.5, 1 at 0 or 1
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed"
        ) from e


@app.post("/predict_cyber", response_model=CyberResponse)
async def predict_cyber(req: CyberRequest, authenticated: bool = Depends(authenticate)):
    """Predict cyber attack probability from network features."""
    if cyber_model is None:
        raise model_unavailable("cyber")

    try:
        if MONITORING_AVAILABLE:
            with InferenceMetrics("cyber") as metrics:
                X = np.array(req.features).reshape(1, -1)
                proba = float(cyber_model.predict_proba(X)[0, 1])
                metrics.record_prediction(proba)
        else:
            X = np.array(req.features).reshape(1, -1)
            proba = float(cyber_model.predict_proba(X)[0, 1])

        return CyberResponse(cyber_attack_probability=proba, threat_level=get_risk_level(proba))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed"
        ) from e


@app.post("/predict_fusion", response_model=FusionResponse)
async def predict_fusion(req: FusionRequest, authenticated: bool = Depends(authenticate)):
    """Predict overall risk from multiple domain scores."""
    if fusion_model is None:
        raise model_unavailable("fusion")

    try:
        keys = sorted(req.scores)
        if MONITORING_AVAILABLE:
            with InferenceMetrics("fusion") as metrics:
                X = np.array([[req.scores[k] for k in keys]])
                proba = float(fusion_model.predict_proba(X)[0, 1])
                metrics.record_prediction(proba)
        else:
            X = np.array([[req.scores[k] for k in keys]])
            proba = float(fusion_model.predict_proba(X)[0, 1])

        scope = scope_guard.evaluate(req.scores, endpoint="fusion")
        return FusionResponse(
            fusion_risk=proba, domains=keys, overall_risk_level=get_risk_level(proba), scope=scope
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed"
        ) from e


@app.post("/predict_attention_fusion", response_model=AttentionFusionResponse)
async def predict_attention_fusion(
    req: AttentionFusionRequest,
    authenticated: bool = Depends(authenticate),
):
    """Predict overall risk using the attention fusion model."""
    model, meta = _load_attention_fusion()
    if model is None:
        raise model_unavailable("attention_fusion")

    try:
        import torch

        features, mask = _build_attention_features(req, meta)
        features_t = torch.as_tensor(features, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)

        if MONITORING_AVAILABLE:
            with InferenceMetrics("attention_fusion") as metrics:
                with torch.no_grad():
                    logits, attn_weights, confidences = model(features_t, key_padding_mask=mask_t)
                    proba = float(torch.sigmoid(logits).squeeze().item())
                metrics.record_prediction(proba)
        else:
            with torch.no_grad():
                logits, attn_weights, confidences = model(features_t, key_padding_mask=mask_t)
                proba = float(torch.sigmoid(logits).squeeze().item())

        attention_mean = None
        if attn_weights is not None:
            attn_vec = attn_weights.mean(dim=1).mean(dim=1).squeeze(0).cpu().numpy()
            attention_mean = attn_vec.tolist()
        domain_confidence = None
        if confidences is not None:
            domain_confidence = confidences.squeeze(0).cpu().numpy().tolist()

        return AttentionFusionResponse(
            fusion_risk=proba,
            domain_order=meta.get("domain_order", []),
            attention_mean=attention_mean,
            domain_confidence=domain_confidence,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Attention fusion prediction failed",
        ) from e


@app.post("/predict_nlp")
async def predict_nlp(req: NLPRequest, authenticated: bool = Depends(authenticate)):
    """Analyze text for suspicious content."""
    model, tokenizer = _load_nlp()
    if model is None or tokenizer is None:
        raise model_unavailable("nlp")

    try:
        import torch

        if MONITORING_AVAILABLE:
            with InferenceMetrics("nlp") as metrics:
                enc = tokenizer(req.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
                with torch.no_grad():
                    logits = model(enc["input_ids"], enc["attention_mask"])
                    proba = float(torch.softmax(logits, dim=1)[:, 1].item())
                metrics.record_prediction(proba)
        else:
            enc = tokenizer(req.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                logits = model(enc["input_ids"], enc["attention_mask"])
                proba = float(torch.softmax(logits, dim=1)[:, 1].item())

        return {"nlp_suspicion_probability": proba, "risk_level": get_risk_level(proba)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="NLP inference failed"
        ) from e


@app.post("/predict_vision")
async def predict_vision(req: VisionRequest, authenticated: bool = Depends(authenticate)):
    """Detect anomalies in images."""
    model = _load_vision()
    if model is None:
        raise model_unavailable("vision")

    try:
        import torch
        from PIL import Image
        from torchvision import transforms

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        try:
            img_bytes = b64decode(req.image_base64, validate=True)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
        except (Base64DecodeError, OSError) as err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image payload") from err

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if MONITORING_AVAILABLE:
            with InferenceMetrics("vision") as metrics:
                x = transform(image).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x)
                    proba = float(torch.softmax(logits, dim=1)[:, 1].item())
                metrics.record_prediction(proba)
        else:
            x = transform(image).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                proba = float(torch.softmax(logits, dim=1)[:, 1].item())

        return {"vision_anomaly_probability": proba, "risk_level": get_risk_level(proba)}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Vision inference failed"
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("UAIS_BIND_HOST", "127.0.0.1"), port=_int_env("UAIS_BIND_PORT", 8000))
