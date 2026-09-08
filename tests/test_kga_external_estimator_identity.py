"""Deployment identity is supplied by the caller's trusted protocol, not the file."""

import numpy as np
import pytest

from kga import KGA, FrozenLinearBenefitEstimator
from kga.policy import Decision


def estimator(intercept=0.2):
    return FrozenLinearBenefitEstimator(
        feature_names=("signal",),
        weights=np.zeros(1),
        intercept=intercept,
        feature_center=np.zeros(1),
        feature_scale=np.ones(1),
        residuals=np.zeros(30),
        evidence_schema_version="test/1",
        protocol_sha256="a" * 64,
        fit_unit="dev",
        calibration_unit="cal",
    )


def arguments():
    return {"protocol_sha256": "a" * 64, "features": {"signal": 0.0}, "evidence_schema_version": "test/1"}


def test_missing_external_identity_invalidates_previous_authority():
    gate = KGA()
    gate.decide(gate.certify(delta_hat=0.3, calib_residuals=np.zeros(30)))
    assert gate.last_decision is Decision.ADAPT
    with pytest.raises(ValueError, match="externally authorized.*payload"):
        gate.certify_evidence(estimator(), **arguments())
    assert gate.last_certificate is None
    assert gate.last_decision is None
    with pytest.raises(ValueError, match="No certificate"):
        gate.decide()


def test_trusted_identity_accepts_exact_payload():
    model = estimator()
    gate = KGA()
    certificate = gate.certify_evidence(model, expected_estimator_payload_sha256=model.artifact_sha256, **arguments())
    assert gate.decide(certificate) is Decision.ADAPT


def test_same_protocol_does_not_authorize_a_replacement_model():
    original, replacement = estimator(), estimator(-0.2)
    gate = KGA()
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        gate.certify_evidence(replacement, expected_estimator_payload_sha256=original.artifact_sha256, **arguments())
    assert gate.last_certificate is None


@pytest.mark.parametrize("identity", [None, True, 0, [], "", "a" * 63, "g" * 64, " " + "a" * 63])
def test_invalid_external_identity_cannot_certify(identity):
    gate = KGA()
    with pytest.raises(ValueError, match="externally authorized.*payload"):
        gate.certify_evidence(estimator(), expected_estimator_payload_sha256=identity, **arguments())
    assert gate.last_certificate is None


def test_predictor_identity_change_during_call_cannot_restore_authority():
    class ChangingEstimator:
        feature_names = ("signal",)
        evidence_schema_version = "test/1"
        protocol_sha256 = "a" * 64
        residuals = np.zeros(30)
        artifact_sha256 = "b" * 64

        def predict(self, features, **kwargs):
            self.artifact_sha256 = "c" * 64
            return 0.2

    gate = KGA()
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        gate.certify_evidence(ChangingEstimator(), expected_estimator_payload_sha256="b" * 64, **arguments())
    assert gate.last_certificate is None
