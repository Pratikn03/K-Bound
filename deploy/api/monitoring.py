"""Monitoring, metrics, and observability for UAIS-V API."""

import time
from collections import defaultdict
from datetime import datetime
from inspect import isawaitable
from typing import Any, Callable, Optional

import psutil
from fastapi import Request, Response
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware


def _existing_collector(name: str):
    names = getattr(REGISTRY, "_names_to_collectors", {})
    base = name.removesuffix("_total")
    for candidate in (name, base, f"{base}_total", f"{name}_total"):
        collector = names.get(candidate)
        if collector is not None:
            return collector
    return None


def _counter(name: str, documentation: str, labelnames: list[str]):
    return _existing_collector(name) or Counter(name, documentation, labelnames)


def _histogram(name: str, documentation: str, labelnames: list[str], **kwargs):
    return _existing_collector(name) or Histogram(name, documentation, labelnames, **kwargs)


def _gauge(name: str, documentation: str, labelnames: list[str] | None = None):
    return _existing_collector(name) or Gauge(name, documentation, labelnames or [])


# Prometheus metrics
REQUEST_COUNT = _counter("uais_requests_total", "Total number of requests", ["method", "endpoint", "status"])

REQUEST_DURATION = _histogram("uais_request_duration_seconds", "Request duration in seconds", ["method", "endpoint"])

MODEL_INFERENCE_COUNT = _counter("uais_model_inferences_total", "Total number of model inferences", ["model_type"])

MODEL_INFERENCE_DURATION = _histogram(
    "uais_model_inference_duration_seconds", "Model inference duration in seconds", ["model_type"]
)

PREDICTION_SCORE = _histogram(
    "uais_prediction_score",
    "Distribution of prediction scores",
    ["model_type"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

SYSTEM_MEMORY_USAGE = _gauge("uais_system_memory_usage_bytes", "System memory usage in bytes")

SYSTEM_CPU_USAGE = _gauge("uais_system_cpu_usage_percent", "System CPU usage percentage")

MODEL_LOADED = _gauge("uais_model_loaded", "Whether model is loaded (1) or not (0)", ["model_type"])


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track metrics."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Extract endpoint (remove query params)
        endpoint = request.url.path

        # Record metrics
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=response.status_code).inc()

        REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(duration)

        # Update system metrics
        SYSTEM_MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=None))

        return response


class InferenceMetrics:
    """Track model inference metrics."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.start_time = None

    def __enter__(self):
        """Start timing inference."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record inference metrics."""
        if self.start_time:
            duration = time.time() - self.start_time
            MODEL_INFERENCE_DURATION.labels(model_type=self.model_type).observe(duration)
            MODEL_INFERENCE_COUNT.labels(model_type=self.model_type).inc()

    def record_prediction(self, score: float):
        """Record a prediction score."""
        PREDICTION_SCORE.labels(model_type=self.model_type).observe(score)


class HealthChecker:
    """Health check utility for API dependencies."""

    def __init__(self):
        self.checks: dict[str, Callable] = {}
        self.history: dict[str, list[dict]] = defaultdict(list)
        self.max_history = 100

    def register_check(self, name: str, check_fn: Callable):
        """Register a health check."""
        self.checks[name] = check_fn

    async def run_checks(self) -> dict[str, Any]:
        """Run all health checks."""
        results = {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "checks": {}}

        for name, check_fn in self.checks.items():
            try:
                start = time.time()
                result = check_fn() if callable(check_fn) else check_fn
                if isawaitable(result):
                    result = await result
                duration = time.time() - start

                check_result = {
                    "status": "pass" if result else "fail",
                    "duration_ms": round(duration * 1000, 2),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                results["checks"][name] = check_result

                # Update history
                self.history[name].append(check_result)
                if len(self.history[name]) > self.max_history:
                    self.history[name].pop(0)

                if not result:
                    results["status"] = "degraded"

            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results["status"] = "unhealthy"

        return results

    def get_history(self, check_name: Optional[str] = None) -> dict:
        """Get health check history."""
        if check_name:
            return {check_name: self.history.get(check_name, [])}
        return dict(self.history)


# Global health checker instance
health_checker = HealthChecker()


def get_system_metrics() -> dict:
    """Get current system metrics."""
    process = psutil.Process()
    memory = process.memory_info()

    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "rss_bytes": memory.rss,
                "rss_mb": round(memory.rss / 1024 / 1024, 2),
                "percent": process.memory_percent(),
            },
            "disk": {"percent": psutil.disk_usage("/").percent},
        },
        "process": {
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.net_connections()),
        },
    }


async def export_prometheus_metrics() -> bytes:
    """Export Prometheus metrics in text format."""
    return generate_latest()
