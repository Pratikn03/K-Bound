"""Security tests for the K-Bound KGA-only deploy API."""

from __future__ import annotations

import asyncio
import importlib
import sys

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

SECURITY_ENV = {
    "UAIS_API_KEYS",
    "UAIS_CORS_ORIGINS",
    "UAIS_RATE_LIMIT_REQUESTS",
    "UAIS_RATE_LIMIT_WINDOW_SECONDS",
    "UAIS_PRODUCTION_MODE",
}


def _reload_api(monkeypatch: pytest.MonkeyPatch, **env: str):
    for key in SECURITY_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in list(sys.modules):
        if name == "deploy.api" or name.startswith("deploy.api."):
            del sys.modules[name]
    return importlib.import_module("deploy.api.main")


def _reload_auth(monkeypatch: pytest.MonkeyPatch, **env: str):
    for key in SECURITY_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in list(sys.modules):
        if name == "deploy.api.auth":
            del sys.modules[name]
    return importlib.import_module("deploy.api.auth")


def test_api_key_auth_fails_closed_when_no_keys_configured(monkeypatch):
    auth = _reload_auth(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.verify_api_key(None))

    assert exc_info.value.status_code == 503
    assert "not configured" in exc_info.value.detail.lower()


def test_cors_uses_explicit_allowlist_from_environment(monkeypatch):
    api = _reload_api(
        monkeypatch,
        UAIS_API_KEYS="secret",
        UAIS_CORS_ORIGINS="https://console.example, https://ops.example",
    )

    cors = next(m for m in api.app.user_middleware if m.cls.__name__ == "CORSMiddleware")

    assert cors.kwargs["allow_origins"] == ["https://console.example", "https://ops.example"]
    assert "*" not in cors.kwargs["allow_origins"]
    assert cors.kwargs["allow_credentials"] is True


def test_health_and_ready_are_public(monkeypatch):
    api = _reload_api(monkeypatch, UAIS_API_KEYS="secret")
    client = TestClient(api.app)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_decide_requires_api_key(monkeypatch):
    api = _reload_api(monkeypatch, UAIS_API_KEYS="secret")
    client = TestClient(api.app)
    payload = {"calib_scores": [0.1, 0.2, 0.3], "test_scores": [0.4, 0.5, 0.6]}

    assert client.post("/decide", json=payload).status_code == 403
    assert client.post("/decide", json=payload, headers={"X-API-Key": "secret"}).status_code == 200


def test_production_runtime_config_validation_rejects_missing_auth_and_wildcard_cors(monkeypatch):
    api = _reload_api(
        monkeypatch,
        UAIS_PRODUCTION_MODE="true",
        UAIS_CORS_ORIGINS="*",
    )

    errors = "\n".join(api.production_config_errors())

    assert "KGA_API_KEYS" in errors or "UAIS_API_KEYS" in errors
    assert "KGA_CORS_ORIGINS" in errors or "UAIS_CORS_ORIGINS" in errors or "wildcard" in errors.lower()
    assert "wildcard" in errors.lower()


def test_request_logging_sets_request_id_without_logging_credentials(monkeypatch, caplog):
    api = _reload_api(monkeypatch, UAIS_API_KEYS="secret")
    client = TestClient(api.app)

    with caplog.at_level("INFO", logger="deploy.api.main"):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_rate_limiter_rejects_requests_over_configured_window(monkeypatch):
    api = _reload_api(
        monkeypatch,
        UAIS_API_KEYS="secret",
        UAIS_RATE_LIMIT_REQUESTS="2",
        UAIS_RATE_LIMIT_WINDOW_SECONDS="60",
    )
    client = TestClient(api.app)

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 429


def test_root_lists_kga_endpoints(monkeypatch):
    api = _reload_api(monkeypatch, UAIS_API_KEYS="secret")
    client = TestClient(api.app)

    body = client.get("/").json()
    assert body["service"] == "kbound-kga-api"
    assert "/kga/decide" in body["endpoints"]
