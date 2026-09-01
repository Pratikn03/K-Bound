"""Operational unavailable-state, predictor-selection, and HTTP JSON contracts.

Synthetic score fixtures only. No target artifacts or model training are used.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from deploy.api.kga_service import assess_kga_decision, perform_kga_decide
from kga import Decision
from kga.certificate import empirical_bernstein


SCORES = np.linspace(0.1, 0.9, 20)


@dataclass
class _ArrayPredictor:
    weights: np.ndarray

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.weights


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cert_mode": "full", "calib_residuals": [0.0] * 19},
        {"cert_mode": "full", "delta_hat": math.nan, "calib_residuals": [0.0] * 19},
        {"cert_mode": "full", "delta_hat": math.inf, "calib_residuals": [0.0] * 19},
        {"cert_mode": "full", "delta_hat": -0.4, "calib_residuals": [math.nan] * 19},
        {"cert_mode": "full", "delta_hat": -0.4, "calib_residuals": []},
        {"cert_mode": "full", "benefit_scores": [], "benefit_range": 2.0},
    ],
)
def test_unavailable_audit_returns_no_fabricated_certificate_and_retains_frozen(kwargs: dict) -> None:
    frozen = _ArrayPredictor(np.array([1.0, 0.0]))
    candidate = _ArrayPredictor(np.array([0.0, 1.0]))
    frozen_before, candidate_before = frozen.weights.copy(), candidate.weights.copy()
    good = assess_kga_decision(SCORES, SCORES, cert_mode="full", delta_hat=0.4, calib_residuals=[0.0] * 19)
    assert good.select_predictor(frozen, candidate) is candidate

    unavailable = assess_kga_decision(SCORES, SCORES, **kwargs)
    assert unavailable.decision is Decision.ABSTAIN
    assert unavailable.availability == "unavailable"
    assert unavailable.reason
    assert unavailable.certificate is None
    assert unavailable.model_action == "retain_frozen"
    selected = unavailable.select_predictor(frozen, candidate)
    assert selected is frozen
    np.testing.assert_array_equal(selected(np.array([[1.0, 2.0]])), frozen(np.array([[1.0, 2.0]])))
    np.testing.assert_array_equal(frozen.weights, frozen_before)
    np.testing.assert_array_equal(candidate.weights, candidate_before)


@pytest.mark.parametrize("bad_scores", [np.array([]), np.array([0.2]), np.array([math.nan, 0.2])])
def test_missing_nonfinite_or_insufficient_evidence_abstains(bad_scores: np.ndarray) -> None:
    result = assess_kga_decision(
        SCORES, bad_scores, cert_mode="full", delta_hat=-0.4, calib_residuals=[0.0] * 19
    )
    assert result.decision is Decision.ABSTAIN
    assert result.availability == "unavailable"
    assert result.certificate is None
    assert result.model_action == "retain_frozen"
    assert result.reason


@pytest.mark.parametrize("delta,expected", [(-0.4, Decision.FREEZE), (0.4, Decision.ADAPT), (0.0, Decision.ABSTAIN)])
def test_valid_explicit_point_and_residual_audit_preserves_trichotomy(delta: float, expected: Decision) -> None:
    result = assess_kga_decision(SCORES, SCORES, cert_mode="full", delta_hat=delta, calib_residuals=[0.0] * 19)
    assert result.decision is expected
    assert result.availability == "available"
    assert result.certificate is not None
    assert result.certificate.delta_hat == delta
    assert result.certificate.epsilon == 0.0
    assert result.decision_scope == "external_estimate_audit"


def test_paired_benefit_audit_matches_unchanged_core_numerically() -> None:
    benefits = np.linspace(0.3, 0.4, 200)
    expected = empirical_bernstein(benefits, alpha=0.1, benefit_range=2.0)
    result = assess_kga_decision(SCORES, SCORES, cert_mode="full", benefit_scores=benefits.tolist(), benefit_range=2.0)
    assert result.certificate == expected
    assert result.decision is Decision.ADAPT
    assert result.decision_scope == "paired_benefit_audit"
    assert result.model_action == "use_candidate"


def test_compatibility_service_retains_three_item_unpacking() -> None:
    decision, certificate, evidence = perform_kga_decide(SCORES, SCORES)
    assert decision is Decision.ABSTAIN
    assert certificate is None
    assert evidence is not None
    decision, certificate, evidence = perform_kga_decide(
        SCORES, SCORES, cert_mode="full", delta_hat=-0.4, calib_residuals=[0.0] * 19
    )
    assert decision is Decision.FREEZE
    assert certificate is not None and evidence is not None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KGA_PRODUCTION_MODE", "false")
    monkeypatch.setenv("UAIS_PRODUCTION_MODE", "false")
    monkeypatch.setenv("KGA_API_KEYS", "runtime-test-key")
    monkeypatch.delenv("KGA_SCOPE_REFERENCE", raising=False)
    monkeypatch.delenv("UAIS_SCOPE_REFERENCE", raising=False)
    from deploy.api import kga_routes, main

    app = main.app
    # Authentication is independently covered by the existing API route suite.
    monkeypatch.setitem(app.dependency_overrides, kga_routes.authenticate, lambda: True)
    with TestClient(app) as test_client:
        yield test_client


def _payload(**kwargs) -> dict:
    return {"calib_scores": SCORES.tolist(), "test_scores": SCORES.tolist(), **kwargs}


def test_http_proxy_reports_unavailable_without_an_invented_zero(client) -> None:
    response = client.post("/decide", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ABSTAIN"
    assert body["delta_hat"] is None and body["epsilon"] is None
    assert body["availability"] == "unavailable"
    assert body["model_action"] == "retain_frozen"
    assert body["decision_scope"] == "evidence_only"
    assert body["reason"]
    json.dumps(body, allow_nan=False)


def test_http_residuals_without_point_estimate_are_unavailable(client) -> None:
    response = client.post("/decide", json=_payload(cert_mode="full", calib_residuals=[0.0] * 19))
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ABSTAIN"
    assert body["delta_hat"] is None and body["epsilon"] is None
    assert body["model_action"] == "retain_frozen"
    assert "explicit delta_hat" in body["reason"]


@pytest.mark.parametrize("delta", [-0.4, 0.4])
def test_http_insufficient_calibration_uses_null_not_infinity(client, delta: float) -> None:
    with pytest.warns(UserWarning, match="ABSTAIN"):
        response = client.post("/decide", json=_payload(cert_mode="full", delta_hat=delta, calib_residuals=[0.0]))
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ABSTAIN"
    assert body["epsilon"] is None
    assert body["delta_hat"] == delta
    assert not body["radius_feasible"]
    assert body["availability"] == "unavailable"
    assert body["model_action"] == "retain_frozen"
    assert "Infinity" not in response.text and "NaN" not in response.text
    json.dumps(body, allow_nan=False)


def test_http_valid_negative_interval_is_a_distinct_freeze_decision(client) -> None:
    response = client.post("/decide", json=_payload(cert_mode="full", delta_hat=-0.4, calib_residuals=[0.0] * 19))
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "FREEZE"
    assert body["availability"] == "available"
    assert body["delta_hat"] == -0.4 and body["epsilon"] == 0.0
    assert body["model_action"] == "retain_frozen"
    assert body["decision_scope"] == "external_estimate_audit"


@pytest.mark.parametrize(
    "field,value",
    [
        ("delta_hat", math.nan),
        ("delta_hat", math.inf),
        ("delta_hat", -math.inf),
        ("test_scores", [0.2, math.nan]),
        ("calib_residuals", [math.inf] * 19),
        ("alpha", math.nan),
    ],
)
def test_http_rejects_nonfinite_request_without_any_commitment(client, field: str, value) -> None:
    # JSON NaN is not a valid externally supplied point estimate. Preserve the
    # existing explicit validation error rather than silently accepting it.
    payload = _payload(cert_mode="full", delta_hat=-0.4, calib_residuals=[0.0] * 19)
    payload[field] = value
    response = client.post(
        "/decide",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json().get("decision") not in {"ADAPT", "FREEZE"}
    json.dumps(response.json(), allow_nan=False)


@pytest.mark.parametrize("bad_scores", [[], [0.2]])
def test_http_insufficient_request_shape_is_rejected_before_assessment(client, bad_scores) -> None:
    response = client.post("/decide", json=_payload(test_scores=bad_scores))
    assert response.status_code == 422
    assert "decision" not in response.json()


def test_http_unavailable_request_does_not_change_active_model_version(client, monkeypatch) -> None:
    from deploy.api import model_governance

    monkeypatch.setitem(model_governance._active_model_version, "runtime_fixture", "frozen-v1")
    before = dict(model_governance._active_model_version)
    response = client.post("/decide", json=_payload(cert_mode="full", calib_residuals=[0.0] * 19))
    assert response.status_code == 200
    assert response.json()["decision"] == "ABSTAIN"
    assert response.json()["model_action"] == "retain_frozen"
    assert model_governance.get_active_model_version("runtime_fixture") == "frozen-v1"
    assert model_governance._active_model_version == before


def test_root_distribution_excludes_reproduction_only_package() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    finder = config["tool"]["setuptools"]["packages"]["find"]
    assert finder["include"] == ["kga", "kga.*"]
    assert "docs*" in finder["exclude"]
    assert finder["namespaces"] is False
    assert config["project"]["scripts"]["kga"] == "kga.cli:main"


def test_public_package_cli_and_http_do_not_import_legacy_heuristic() -> None:
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; from pathlib import Path; import kga, kga.cli; "
        "import deploy.api.kga_service, deploy.api.kga_routes; "
        "assert Path(kga.__file__).resolve().parent == Path.cwd() / 'kga'; "
        "assert not any(n == 'kbound' or n.startswith('kbound.') for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
