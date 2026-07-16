"""FastAPI service exposing the KGA (Knowability-Guided Adaptation) decide API.

Hardened endpoint with API-key authentication, Prometheus monitoring, rate
limiting, request timeouts, and scope-guard drift telemetry. It mounts the KGA
certificate routes (``/kga/*``) and the model-governance routes
(``/model_version`` / rollback). This service is K-Bound only.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import scope_guard  # noqa: F401  (drift telemetry collectors register on import)
from .auth import API_KEYS
from .envutil import bool_env, csv_env, int_env
from .kga_routes import router as kga_router
from .model_governance import router as model_governance_router
from .rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)

# Monitoring is optional; the service runs without it (metrics endpoint degrades).
try:
    from .monitoring import (
        MetricsMiddleware,
        export_prometheus_metrics,
    )

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


CORS_ORIGINS = csv_env("KGA_CORS_ORIGINS", "UAIS_CORS_ORIGINS")
RATE_LIMIT_REQUESTS = int_env("KGA_RATE_LIMIT_REQUESTS", "UAIS_RATE_LIMIT_REQUESTS", default=120)
RATE_LIMIT_WINDOW_SECONDS = int_env(
    "KGA_RATE_LIMIT_WINDOW_SECONDS", "UAIS_RATE_LIMIT_WINDOW_SECONDS", default=60
)
REQUEST_TIMEOUT_SECONDS = int_env(
    "KGA_REQUEST_TIMEOUT_SECONDS", "UAIS_REQUEST_TIMEOUT_SECONDS", default=30
)
PRODUCTION_MODE = bool_env("KGA_PRODUCTION_MODE", "UAIS_PRODUCTION_MODE", default=False)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_mode() -> str:
    return "production" if PRODUCTION_MODE else "development"


def production_config_errors() -> list[str]:
    """Production-blocking configuration errors (no secrets exposed)."""
    errors: list[str] = []
    if not API_KEYS:
        errors.append("KGA_API_KEYS (or legacy UAIS_API_KEYS) must be configured with at least one API key")
    if PRODUCTION_MODE and not CORS_ORIGINS:
        errors.append(
            "KGA_CORS_ORIGINS (or legacy UAIS_CORS_ORIGINS) must be configured with explicit origins in production mode"
        )
    if any(origin == "*" for origin in CORS_ORIGINS):
        errors.append("KGA_CORS_ORIGINS cannot use a wildcard origin in production")
    return errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRODUCTION_MODE:
        errors = production_config_errors()
        if errors:
            raise RuntimeError("Invalid production configuration: " + "; ".join(errors))
    yield


app = FastAPI(
    title="K-Bound / KGA API",
    version="2.0",
    description="Knowability-Guided Adaptation certificate service (adapt / freeze / abstain).",
    lifespan=lifespan,
)


class TimeoutMiddleware(BaseHTTPMiddleware):
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
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed request_id=%s method=%s path=%s",
                request_id, request.method, request.url.path,
            )
            raise
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RateLimitMiddleware, limit=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS)
app.add_middleware(TimeoutMiddleware, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=bool(CORS_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
if MONITORING_AVAILABLE:
    app.add_middleware(MetricsMiddleware)

# KGA certificate routes + model governance (model_version / rollback)
app.include_router(kga_router)
app.include_router(model_governance_router)


def readiness_report() -> dict[str, object]:
    config_errors = production_config_errors()
    ready = not config_errors and MONITORING_AVAILABLE
    return {
        "ready": ready,
        "mode": runtime_mode(),
        "timestamp": utc_timestamp(),
        "checks": {
            "configuration": {"status": "pass" if not config_errors else "fail", "errors": config_errors},
            "monitoring": {"status": "pass" if MONITORING_AVAILABLE else "fail"},
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "mode": runtime_mode(), "timestamp": utc_timestamp()}


@app.get("/ready")
async def ready():
    return readiness_report()


@app.get("/")
async def root():
    return {
        "service": "kbound-kga-api",
        "version": "2.0",
        "endpoints": ["/health", "/ready", "/metrics", "/kga/decide"],
    }


if MONITORING_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(export_prometheus_metrics())
