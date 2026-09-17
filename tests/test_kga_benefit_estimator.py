from __future__ import annotations

import json

import numpy as np
import pytest

from kga import (
    EVIDENCE_FEATURE_NAMES,
    EVIDENCE_SCHEMA_VERSION,
    KGA,
    FrozenLinearBenefitEstimator,
    fit_frozen_linear_benefit_estimator,
)
from kga.policy import Decision

HASH = "a" * 64


def _estimator(*, intercept: float = 0.2) -> FrozenLinearBenefitEstimator:
    n_features = len(EVIDENCE_FEATURE_NAMES)
    return FrozenLinearBenefitEstimator(
        feature_names=EVIDENCE_FEATURE_NAMES,
        weights=np.zeros(n_features),
        intercept=intercept,
        feature_center=np.zeros(n_features),
        feature_scale=np.ones(n_features),
        residuals=np.zeros(20),
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        protocol_sha256=HASH,
        fit_unit="development-domains",
        calibration_unit="calibration-domains",
    )


def test_label_free_facade_consumes_evidence_and_frozen_estimator():
    rng = np.random.default_rng(3)
    gate = KGA(alpha=0.1)
    gate.evidence(rng.normal(size=(100, 2)), rng.normal(size=(100, 2)))
    estimator = _estimator()
    cert = gate.certify_evidence(estimator, protocol_sha256=HASH)
    assert cert.delta_hat == pytest.approx(0.2)
    assert cert.epsilon == pytest.approx(0.0)
    assert gate.decide(cert) is Decision.ADAPT
    assert gate.explain()["estimator_artifact_sha256"] == estimator.artifact_sha256


def test_label_free_facade_fails_closed_on_schema_or_protocol_drift():
    gate = KGA(alpha=0.1)
    features = dict.fromkeys(EVIDENCE_FEATURE_NAMES, 0.0)
    estimator = _estimator()
    with pytest.raises(ValueError, match="protocol"):
        gate.certify_evidence(
            estimator,
            protocol_sha256="b" * 64,
            features=features,
            evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        )
    with pytest.raises(ValueError, match="schema"):
        gate.certify_evidence(
            estimator,
            protocol_sha256=HASH,
            features=features,
            evidence_schema_version="wrong-schema",
        )


def test_frozen_artifact_roundtrip_and_tamper_detection(tmp_path):
    estimator = _estimator()
    path = tmp_path / "benefit.json"
    estimator.write_new_json(path)
    loaded = FrozenLinearBenefitEstimator.load_json(path)
    assert loaded.artifact_sha256 == estimator.artifact_sha256

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["intercept"] = 999.0
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FrozenLinearBenefitEstimator.from_dict(payload)


def test_reference_fit_uses_fit_scaling_and_disjoint_calibration():
    rng = np.random.default_rng(9)
    x_fit = rng.normal(size=(80, 3))
    y_fit = 0.4 + x_fit @ np.array([0.5, -0.2, 0.1])
    x_cal = rng.normal(size=(20, 3))
    y_cal = 0.4 + x_cal @ np.array([0.5, -0.2, 0.1])
    estimator = fit_frozen_linear_benefit_estimator(
        x_fit,
        y_fit,
        x_cal,
        y_cal,
        feature_names=("a", "b", "c"),
        evidence_schema_version="custom/1",
        protocol_sha256=HASH,
    )
    prediction = estimator.predict(
        {"a": 0.2, "b": -0.1, "c": 0.3},
        evidence_schema_version="custom/1",
        protocol_sha256=HASH,
    )
    assert prediction == pytest.approx(0.55, abs=1e-5)
    assert np.max(estimator.residuals) < 1e-5

    with pytest.raises(ValueError, match="disjoint"):
        fit_frozen_linear_benefit_estimator(
            x_fit,
            y_fit,
            x_cal,
            y_cal,
            feature_names=("a", "b", "c"),
            evidence_schema_version="custom/1",
            protocol_sha256=HASH,
            fit_unit="same",
            calibration_unit="same",
        )
