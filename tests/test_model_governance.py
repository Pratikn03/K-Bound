"""Model governance API tests (Gate P P11)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "MANIFEST.json"


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("KGA_API_KEYS", "test-secret")
    monkeypatch.setenv("KGA_CORS_ORIGINS", "https://ops.example")
    monkeypatch.setenv("KGA_PRODUCTION_MODE", "false")
    for name in list(sys.modules):
        if name == "deploy.api" or name.startswith("deploy.api."):
            del sys.modules[name]
    main = importlib.import_module("deploy.api.main")
    return TestClient(main.app)


@pytest.mark.skipif(not MANIFEST.is_file(), reason="models/MANIFEST.json not present locally")
def test_model_versions_endpoint(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    resp = client.get("/models/versions", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "model_version" in body
    assert body["models"]  # non-empty when a local manifest is present


@pytest.mark.skipif(not MANIFEST.is_file(), reason="models/MANIFEST.json not present locally")
def test_model_rollback_rejects_unknown_version(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    resp = client.post(
        "/models/rollback",
        headers={"X-API-Key": "test-secret"},
        json={"model_type": "fraud", "target_version": "nonexistent-v99"},
    )
    assert resp.status_code == 400


@pytest.mark.skipif(not MANIFEST.is_file(), reason="models/MANIFEST.json not present locally")
def test_model_rollback_applies_default(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    # When a manifest exists, "default" must be listed in that model type's history.
    import json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model_type = next(iter(manifest.get("versions", {})), None)
    if model_type is None:
        pytest.skip("manifest has no version history")
    history = manifest["versions"][model_type]
    target = "default" if "default" in history else history[0]
    resp = client.post(
        "/models/rollback",
        headers={"X-API-Key": "test-secret"},
        json={"model_type": model_type, "target_version": target},
    )
    assert resp.status_code == 200
    assert resp.json()["model_version"] == target


def test_model_versions_endpoint_empty_without_manifest(monkeypatch: pytest.MonkeyPatch):
    if MANIFEST.is_file():
        pytest.skip("local manifest present")
    client = _client(monkeypatch)
    resp = client.get("/models/versions", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("models") == {}
