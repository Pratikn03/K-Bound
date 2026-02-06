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
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, validator

# Import monitoring and auth (optional - graceful fallback)
try:
    from .auth import authenticate
    from .monitoring import (
        InferenceMetrics,
        MetricsMiddleware,
        export_prometheus_metrics,
        get_system_metrics,
        health_checker,
        MODEL_LOADED,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    authenticate = None

# FastAPI app
app = FastAPI(
    title="UAIS-V Enhanced API",
    version="2.0",
    description="Universal Anomaly Intelligence System with authentication and monitoring"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware if available
if MONITORING_AVAILABLE:
    app.add_middleware(MetricsMiddleware)

# Model paths
project_root = Path(__file__).resolve().parents[2]
fraud_model_path = project_root / "models" / "fraud" / "supervised" / "fraud_model.pkl"
cyber_model_path = project_root / "models" / "cyber" / "supervised" / "cyber_model.pkl"
fusion_model_path = project_root / "experiments" / "fusion" / "models" / "fusion_meta_model.pkl"
attention_model_dir = project_root / "models" / "fusion" / "attention"
attention_ckpt_path = attention_model_dir / "attention_fusion.pt"
nlp_model_dir = project_root / "models" / "nlp" / "distilbert"
vision_model_dir = project_root / "models" / "vision" / "resnet"

# Load models
fraud_model = joblib.load(fraud_model_path) if fraud_model_path.exists() else None
cyber_model = joblib.load(cyber_model_path) if cyber_model_path.exists() else None
fusion_model = joblib.load(fusion_model_path) if fusion_model_path.exists() else None

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


def _load_nlp():
    """Lazy load NLP model and tokenizer."""
    global _nlp_artifacts_loaded, _nlp_model, _nlp_tokenizer
    if _nlp_artifacts_loaded:
        return _nlp_model, _nlp_tokenizer
    try:
        import torch
        from uais_v.models.nlp_text_model import DistilBERTClassifier, get_tokenizer
    except Exception:
        _nlp_artifacts_loaded = True
        return None, None
    if not nlp_model_dir.exists():
        _nlp_artifacts_loaded = True
        return None, None
    tokenizer = get_tokenizer(str(nlp_model_dir))
    model = DistilBERTClassifier(str(nlp_model_dir), num_labels=2)
    state_path = nlp_model_dir / "model.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    _nlp_model, _nlp_tokenizer = model, tokenizer
    _nlp_artifacts_loaded = True
    if MONITORING_AVAILABLE:
        MODEL_LOADED.labels(model_type="nlp").set(1)
    return model, tokenizer


def _load_vision():
    """Lazy load vision model."""
    global _vision_model
    if _vision_model is not None:
        return _vision_model
    try:
        import torch
        from uais_v.models.vision_resnet import VisionConfig, build_resnet_classifier
    except Exception:
        _vision_model = False
        return None
    state_path = vision_model_dir / "model.pt"
    if not state_path.exists():
        _vision_model = False
        return None
    cfg = VisionConfig(model_name="resnet18", num_classes=2, pretrained=False)
    model = build_resnet_classifier(cfg)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
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

    state = torch.load(attention_ckpt_path, map_location="cpu")
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
    features: List[float] = Field(..., min_items=1, description="Feature vector for fraud detection")

    @validator('features')
    def validate_features(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Features must be finite numbers")
        return v


class FraudResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    confidence: float


class CyberRequest(BaseModel):
    features: List[float] = Field(..., min_items=1, description="Feature vector for cyber attack detection")


class CyberResponse(BaseModel):
    cyber_attack_probability: float = Field(..., ge=0.0, le=1.0)
    threat_level: str


class FusionRequest(BaseModel):
    scores: Dict[str, float] = Field(..., description="Domain-specific anomaly scores")

    @validator('scores')
    def validate_scores(cls, v):
        for score in v.values():
            if not (0 <= score <= 1):
                raise ValueError("Scores must be between 0 and 1")
        return v


class FusionResponse(BaseModel):
    fusion_risk: float
    domains: List[str]
    overall_risk_level: str


class AttentionDomainInput(BaseModel):
    domain: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    embeddings: Optional[List[float]] = None


class AttentionFusionRequest(BaseModel):
    domains: List[AttentionDomainInput]


class AttentionFusionResponse(BaseModel):
    fusion_risk: float
    domain_order: List[str]
    attention_mean: Optional[List[float]]
    domain_confidence: Optional[List[float]]


class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class VisionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (jpg/png)")


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


def _build_attention_features(req: AttentionFusionRequest, meta: Dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
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
            "authentication": authenticate is not None,
            "monitoring": MONITORING_AVAILABLE,
        },
        "available_models": {
            "fraud": fraud_model is not None,
            "cyber": cyber_model is not None,
            "fusion": fusion_model is not None,
            "nlp": nlp_model_dir.exists(),
            "vision": vision_model_dir.exists(),
        },
    }


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok", "timestamp": str(np.datetime64('now'))}


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with component status."""
    if not MONITORING_AVAILABLE:
        return {"status": "ok", "note": "Detailed monitoring not available"}

    return await health_checker.run_checks()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not MONITORING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Metrics not available. Install prometheus_client."
        )

    metrics_data = await export_prometheus_metrics()
    return PlainTextResponse(metrics_data, media_type="text/plain")


@app.get("/system")
async def system_info():
    """System resource information."""
    if not MONITORING_AVAILABLE:
        return {"note": "System metrics not available"}

    return get_system_metrics()


@app.post("/predict_fraud", response_model=FraudResponse)
async def predict_fraud(
    req: FraudRequest,
    authenticated: bool = Depends(authenticate) if authenticate else True
):
    """Predict fraud probability from features."""
    if fraud_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud model not loaded"
        )

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
            confidence=abs(proba - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_cyber", response_model=CyberResponse)
async def predict_cyber(
    req: CyberRequest,
    authenticated: bool = Depends(authenticate) if authenticate else True
):
    """Predict cyber attack probability from network features."""
    if cyber_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cyber model not loaded"
        )

    try:
        if MONITORING_AVAILABLE:
            with InferenceMetrics("cyber") as metrics:
                X = np.array(req.features).reshape(1, -1)
                proba = float(cyber_model.predict_proba(X)[0, 1])
                metrics.record_prediction(proba)
        else:
            X = np.array(req.features).reshape(1, -1)
            proba = float(cyber_model.predict_proba(X)[0, 1])

        return CyberResponse(
            cyber_attack_probability=proba,
            threat_level=get_risk_level(proba)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_fusion", response_model=FusionResponse)
async def predict_fusion(
    req: FusionRequest,
    authenticated: bool = Depends(authenticate) if authenticate else True
):
    """Predict overall risk from multiple domain scores."""
    if fusion_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fusion model not loaded"
        )

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

        return FusionResponse(
            fusion_risk=proba,
            domains=keys,
            overall_risk_level=get_risk_level(proba)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_attention_fusion", response_model=AttentionFusionResponse)
async def predict_attention_fusion(
    req: AttentionFusionRequest,
    authenticated: bool = Depends(authenticate) if authenticate else True,
):
    """Predict overall risk using the attention fusion model."""
    model, meta = _load_attention_fusion()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Attention fusion model not loaded",
        )

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
            detail=f"Attention fusion prediction failed: {str(e)}",
        )


@app.post("/predict_nlp")
async def predict_nlp(
    req: NLPRequest,
    authenticated: bool = Depends(authenticate) if authenticate else True
):
    """Analyze text for suspicious content."""
    model, tokenizer = _load_nlp()
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLP model not available"
        )

    try:
        import torch

        if MONITORING_AVAILABLE:
            with InferenceMetrics("nlp") as metrics:
                enc = tokenizer(req.text, return_tensors="pt", truncation=True,
                              padding="max_length", max_length=128)
                with torch.no_grad():
                    logits = model(enc["input_ids"], enc["attention_mask"])
                    proba = float(torch.softmax(logits, dim=1)[:, 1].item())
                metrics.record_prediction(proba)
        else:
            enc = tokenizer(req.text, return_tensors="pt", truncation=True,
                          padding="max_length", max_length=128)
            with torch.no_grad():
                logits = model(enc["input_ids"], enc["attention_mask"])
                proba = float(torch.softmax(logits, dim=1)[:, 1].item())

        return {
            "nlp_suspicion_probability": proba,
            "risk_level": get_risk_level(proba)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP inference failed: {str(e)}"
        )


@app.post("/predict_vision")
async def predict_vision(
    req: VisionRequest,
    authenticated: bool = Depends(authenticate) if authenticate else True
):
    """Detect anomalies in images."""
    model = _load_vision()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model not available"
        )

    try:
        import torch
        from PIL import Image
        from torchvision import transforms

        img_bytes = b64decode(req.image_base64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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

        return {
            "vision_anomaly_probability": proba,
            "risk_level": get_risk_level(proba)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision inference failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
