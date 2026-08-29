"""Tests for KGA HTTP routes (POST /decide, GET /kga/health)."""

from __future__ import annotations

import importlib
import sys

import numpy as np
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


def test_kga_health_no_auth(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    resp = client.get("/kga/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["component"] == "kga"
    assert "version" in body


def test_decide_requires_auth(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    payload = {
        "calib_scores": [0.1, 0.2, 0.3, 0.4],
        "test_scores": [0.5, 0.6, 0.7, 0.8],
    }
    assert client.post("/decide", json=payload).status_code == 403


def test_decide_proxy_mode(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    headers = {"X-API-Key": "test-secret"}
    resp = client.post(
        "/decide",
        headers=headers,
        json={
            "calib_scores": [0.1, 0.2, 0.15, 0.18],
            "test_scores": [0.5, 0.55, 0.52, 0.48],
            "cert_mode": "proxy",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in {"ADAPT", "FREEZE", "ABSTAIN"}
    assert body["cert_mode"] == "proxy"
    assert body["decision"] == "ABSTAIN"
    assert body["epsilon"] is None
    assert not body["radius_feasible"]


def test_decide_full_mode_benefit_scores(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    headers = {"X-API-Key": "test-secret"}
    rng = np.random.default_rng(0)
    benefits = rng.normal(0.4, 0.05, size=50).tolist()
    resp = client.post(
        "/decide",
        headers=headers,
        json={
            "calib_scores": [0.1, 0.2, 0.15, 0.18],
            "test_scores": [0.5, 0.55, 0.52, 0.48],
            "cert_mode": "full",
            "benefit_scores": benefits,
            "method": "ebern",
            "benefit_range": 2.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cert_mode"] == "full"
    assert body["method"] == "ebern"
    assert body["decision"] == "ADAPT"
    assert body["radius_feasible"]


def test_decide_full_mode_rejects_data_dependent_support(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    resp = client.post(
        "/decide",
        headers={"X-API-Key": "test-secret"},
        json={
            "calib_scores": [0.1, 0.2, 0.15, 0.18],
            "test_scores": [0.5, 0.55, 0.52, 0.48],
            "cert_mode": "full",
            "benefit_scores": [0.2] * 20,
            "method": "ebern",
        },
    )
    assert resp.status_code == 422


def test_decide_full_mode_requires_inputs(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    headers = {"X-API-Key": "test-secret"}
    resp = client.post(
        "/decide",
        headers=headers,
        json={
            "calib_scores": [0.1, 0.2, 0.15, 0.18],
            "test_scores": [0.5, 0.55, 0.52, 0.48],
            "cert_mode": "full",
        },
    )
    assert resp.status_code == 422


def test_kga_service_unit():
    from deploy.api.kga_service import perform_kga_decide

    calib = np.array([0.1, 0.2, 0.3, 0.4])
    test = np.array([0.5, 0.6, 0.7, 0.8])
    decision, cert, ev = perform_kga_decide(calib, test, alpha=0.1)
    assert decision.value in {"ADAPT", "FREEZE", "ABSTAIN"}
    assert ev.n_calib == 4
