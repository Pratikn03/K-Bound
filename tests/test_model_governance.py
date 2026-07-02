"""Model governance API tests (Gate P P11)."""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("UAIS_API_KEYS", "test-secret")
    monkeypatch.setenv("UAIS_CORS_ORIGINS", "https://ops.example")
    monkeypatch.setenv("UAIS_PRODUCTION_MODE", "false")
    for name in list(sys.modules):
        if name == "deploy.api" or name.startswith("deploy.api."):
            del sys.modules[name]
    main = importlib.import_module("deploy.api.main")
    return TestClient(main.app)


def test_model_versions_endpoint(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    resp = client.get("/models/versions", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "model_version" in body
    assert "fraud" in body["models"]


def test_model_rollback_rejects_unknown_version(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    resp = client.post(
        "/models/rollback",
        headers={"X-API-Key": "test-secret"},
        json={"model_type": "fraud", "target_version": "nonexistent-v99"},
    )
    assert resp.status_code == 400


def test_model_rollback_applies_default(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    resp = client.post(
        "/models/rollback",
        headers={"X-API-Key": "test-secret"},
        json={"model_type": "fraud", "target_version": "default"},
    )
    assert resp.status_code == 200
    assert resp.json()["model_version"] == "default"
