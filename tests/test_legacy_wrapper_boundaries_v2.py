"""Regression tests for unsupported strict decisions in reproduction/edge wrappers."""

import math

import numpy as np
import pytest
from kbound.certificate import conformal_radius, decide
from kbound.kga import KGA
from kbound_edge.conformal import calibrate_conformal, conservative_conformal_radius
from kbound_edge.policy import PolicyContext, apply_policy, kga_decide


@pytest.mark.parametrize("candidate", ["entropy_drop", "collapse"])
def test_legacy_gate_without_estimator_never_promotes_evidence_to_direction(candidate):
    frozen = np.full((20, 2), 0.5)
    adapted = np.array([[0.8, 0.2], [0.2, 0.8]] * 10)
    if candidate == "collapse":
        adapted[:] = [0.999, 0.001]
    before_frozen, before_adapted = frozen.copy(), adapted.copy()
    gate = KGA(f0=object(), fa=object())
    frozen_model = gate.f0
    assert gate.decide(frozen, adapted) == "abstain"
    assert gate.f0 is frozen_model
    np.testing.assert_array_equal(frozen, before_frozen)
    np.testing.assert_array_equal(adapted, before_adapted)
    assert gate.evidence(frozen, adapted).shape == (11,)


def test_legacy_live_gate_without_authority_does_not_touch_frozen_model():
    class UntouchedModel:
        def eval(self):
            pytest.fail("an unavailable legacy gate must not change model mode")

        def __call__(self, batch):
            pytest.fail("an unavailable legacy gate must not run inference")

    frozen = UntouchedModel()
    gate = KGA(f0=frozen, fa=UntouchedModel())
    assert gate.decide_from_batch(object()) == "abstain"
    assert gate.f0 is frozen


INVALID_ESTIMATES = [
    math.nan,
    math.inf,
    -math.inf,
    True,
    np.bool_(True),
    "0.2",
    None,
    np.ma.array(0.2, mask=True),
    np.array([0.2]),
    10**1000,
]
INVALID_RADII = [
    -0.1,
    -math.inf,
    math.nan,
    True,
    np.bool_(False),
    "0.1",
    None,
    np.ma.array(0.1, mask=True),
    np.array([0.1]),
    10**1000,
]


@pytest.mark.parametrize("value", INVALID_ESTIMATES)
@pytest.mark.parametrize("gate", [decide, kga_decide])
def test_invalid_estimate_is_rejected_before_any_interval_is_reported(gate, value):
    with pytest.raises(ValueError):
        gate(value, 0.1)


@pytest.mark.parametrize("value", INVALID_RADII)
@pytest.mark.parametrize("gate", [decide, kga_decide])
def test_invalid_radius_is_rejected_before_any_interval_is_reported(gate, value):
    with pytest.raises(ValueError):
        gate(0.2, value)


@pytest.mark.parametrize("value", INVALID_ESTIMATES)
@pytest.mark.parametrize("policy", ["kga_full", "kga_no_radius"])
def test_edge_kga_policies_cannot_bypass_estimate_validation(policy, value):
    with pytest.raises(ValueError):
        apply_policy(policy, PolicyContext(value, 0.1))


@pytest.mark.parametrize("value", INVALID_RADII)
def test_edge_full_policy_cannot_bypass_radius_validation(value):
    with pytest.raises(ValueError):
        apply_policy("kga_full", PolicyContext(0.2, value))


@pytest.mark.parametrize(
    ("estimate", "radius", "action"),
    [
        (0.2, 0.1, "adapt"),
        (-0.2, 0.1, "freeze"),
        (0.1, 0.1, "abstain"),
        (-0.1, 0.1, "abstain"),
        (0, 0, "abstain"),
        (0.2, math.inf, "abstain"),
        (np.float32(0.3), np.float64(0.1), "adapt"),
        (np.int64(-1), np.int32(0), "freeze"),
    ],
)
def test_valid_conditional_intervals_keep_strict_boundary_semantics(estimate, radius, action):
    assert decide(estimate, radius) == action
    record = kga_decide(estimate, radius)
    assert record.decision == action
    if math.isinf(radius):
        assert record.lower == -math.inf
        assert record.upper == math.inf


BAD_RESIDUALS = [
    [0.01] * 99 + [math.nan],
    [0.01] * 99 + [math.inf],
    [-0.01] + [0.01] * 99,
    np.ma.array(np.full(100, 0.01), mask=[True] + [False] * 99),
    [np.ma.array(0.01, mask=True)] * 100,
]


@pytest.mark.parametrize("residuals", BAD_RESIDUALS)
@pytest.mark.parametrize("radius", [conformal_radius, conservative_conformal_radius])
@pytest.mark.filterwarnings("ignore:.*converting a masked element to nan.*:UserWarning")
def test_bad_residuals_cannot_hide_outside_selected_rank(radius, residuals):
    with pytest.raises(ValueError):
        radius(residuals, alpha=0.1)


@pytest.mark.filterwarnings("ignore:split-conformal.*")
def test_edge_rank_is_exact_and_infeasibility_is_not_clamped():
    assert conservative_conformal_radius(np.arange(20.0), alpha=0.1) == 18.0
    radius = conservative_conformal_radius(np.zeros(5), alpha=0.1)
    assert radius == math.inf
    assert kga_decide(0.5, radius).decision == "abstain"


class FixedPrediction:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        return self.prediction


@pytest.mark.parametrize("conservative", [True, False])
@pytest.mark.parametrize("field", ["features", "benefits", "prediction"])
def test_calibration_preserves_masks_until_rejection(field, conservative):
    features = np.zeros((100, 2))
    benefits = np.full(100, 0.1)
    prediction = np.zeros(100)
    if field == "features":
        features = np.ma.array(features, mask=True)
    elif field == "benefits":
        benefits = np.ma.array(benefits, mask=True)
    else:
        prediction = np.ma.array(prediction, mask=True)
    with pytest.raises(ValueError):
        calibrate_conformal(FixedPrediction(prediction), features, benefits, conservative=conservative)


@pytest.mark.parametrize("prediction", [0.0, np.zeros((100, 1)), np.zeros(99), [0.0] * 99 + [math.inf]])
def test_calibration_rejects_wrong_shape_or_nonfinite_predictions(prediction):
    with pytest.raises(ValueError):
        calibrate_conformal(FixedPrediction(prediction), np.zeros((100, 2)), np.zeros(100))


@pytest.mark.parametrize("alpha", [True, np.bool_(False), "0.1", math.nan, math.inf, 0.0, 1.0])
@pytest.mark.parametrize("radius", [conformal_radius, conservative_conformal_radius])
def test_radius_rejects_invalid_alpha_explicitly(radius, alpha):
    with pytest.raises(ValueError):
        radius(np.zeros(20), alpha=alpha)
