import numpy as np
import pytest

from kga.benefit import FrozenLinearBenefitEstimator
from kga.integrations import elara
from kga.policy import Decision

FEATURES = (
    "ks_mean",
    "ks_max",
    "disagree",
    "entropy_shift",
    "conf_shift",
    "ess_frac",
    "best_val_auc",
    "val_gap",
    "val_disagreement",
    "n_experts",
)


def model():
    return FrozenLinearBenefitEstimator(
        feature_names=FEATURES,
        weights=np.zeros(10),
        intercept=0.2,
        feature_center=np.zeros(10),
        feature_scale=np.ones(10),
        residuals=np.zeros(30),
        evidence_schema_version="elara/test",
        protocol_sha256="a" * 64,
        fit_unit="dev",
        calibration_unit="cal",
    )


@pytest.fixture
def inputs(monkeypatch):
    def reliability(scores, labels):
        return {"val_auc": [0.9, 0.7], "best_auc": 0.9, "gap": 0.2, "disagreement": 0.1}

    def route(scores, labels, target, policy, *, action):
        return target[:, 1], "fuse"

    monkeypatch.setattr(elara, "_load_router_api", lambda: (object, reliability, route))
    scores = np.array([[0.2, 0.3], [0.7, 0.8], [0.3, 0.4], [0.8, 0.9]])
    return {
        "s_val": scores,
        "y_val": np.array([0, 1, 0, 1]),
        "s_test": scores,
        "mode": "label_free",
        "protocol_sha256": "a" * 64,
    }


def test_missing_external_identity_returns_unavailable_retention(inputs):
    result = elara.ELARAKGAGuard().decide(estimator=model(), **inputs)
    assert result.decision is Decision.ABSTAIN
    assert result.deployed_action == "retain_frozen"
    assert result.certificate["availability"] == "unavailable"
    assert not result.claim_eligible
    np.testing.assert_array_equal(result.deployed_scores, result.frozen_scores)


def test_wrong_external_identity_never_routes_candidate(inputs):
    result = elara.ELARAKGAGuard().decide(estimator=model(), expected_estimator_payload_sha256="b" * 64, **inputs)
    assert result.decision is Decision.ABSTAIN
    assert result.labels_used_for_decision == 0
    np.testing.assert_array_equal(result.deployed_scores, result.frozen_scores)


def test_exact_identity_preserves_real_adapt_path(inputs):
    estimator = model()
    result = elara.ELARAKGAGuard().decide(
        estimator=estimator, expected_estimator_payload_sha256=estimator.artifact_sha256, **inputs
    )
    assert result.decision is Decision.ADAPT
    np.testing.assert_array_equal(result.deployed_scores, result.candidate_scores)


def test_missing_estimator_returns_unavailable_retention(inputs):
    result = elara.ELARAKGAGuard().decide(estimator=None, **inputs)
    assert result.decision is Decision.ABSTAIN
    assert result.deployed_action == "retain_frozen"


def test_evaluation_separates_commitment_from_measured_inclusion(inputs):
    result = elara.ELARAKGAGuard().decide(estimator=model(), **inputs)
    result.decision = Decision.FREEZE
    result.certificate = {"lower": -1.0, "upper": 1.0}
    evaluated = elara.evaluate_result(result, inputs["y_val"])
    assert "covered" not in evaluated
    assert evaluated["committed"] is True
    assert evaluated["cell_interval_included"] is True
    assert evaluated["false_freeze"] == (evaluated["brier_benefit"] >= 0.0)


def test_unavailable_interval_has_unknown_inclusion(inputs):
    result = elara.ELARAKGAGuard().decide(estimator=model(), **inputs)
    evaluated = elara.evaluate_result(result, inputs["y_val"])
    assert evaluated["committed"] is False
    assert evaluated["cell_interval_included"] is None


def test_label_free_mode_still_rejects_outcome_input(inputs):
    with pytest.raises(ValueError, match="must not receive y_test"):
        elara.ELARAKGAGuard().decide(estimator=model(), y_test=inputs["y_val"], **inputs)


def test_router_mutation_cannot_replace_frozen_fallback(inputs, monkeypatch):
    original = inputs["s_test"].copy()
    policy, reliability, _ = elara._load_router_api()

    def mutating_route(scores, labels, target, config, *, action):
        target[:] = 9.0
        return target[:, 0], "fuse"

    monkeypatch.setattr(elara, "_load_router_api", lambda: (policy, reliability, mutating_route))
    result = elara.ELARAKGAGuard().decide(estimator=None, **inputs)
    assert result.decision is Decision.ABSTAIN
    np.testing.assert_array_equal(result.deployed_scores, original[:, 0])
    np.testing.assert_array_equal(inputs["s_test"], original)


@pytest.mark.parametrize("mode", ["retrospective_audit", "target_label_light"])
def test_brier_certificate_rejects_unbounded_detector_scores(inputs, mode):
    inputs["s_test"] = np.tile([100.0, 0.0], (100, 1))
    inputs["mode"] = mode
    inputs["y_test"] = np.zeros(100)
    if mode == "target_label_light":
        inputs["probe_indices"] = np.arange(100)
    with pytest.raises(ValueError, match="probabilit.*\\[0, 1\\]"):
        elara.ELARAKGAGuard().decide(**inputs)


@pytest.mark.parametrize("field", ["s_val", "s_test", "y_val"])
def test_masked_inputs_cannot_become_authorized_adapt(inputs, field):
    estimator = model()
    inputs[field] = np.ma.array(inputs[field], mask=True)
    # No valid frozen-score/evidence pair exists at this input boundary.
    # Reject it rather than exposing masked values as observations.
    with pytest.raises(ValueError, match="finite"):
        elara.ELARAKGAGuard().decide(
            estimator=estimator,
            expected_estimator_payload_sha256=estimator.payload_sha256,
            **inputs,
        )


def test_masked_candidate_cannot_become_authorized_adapt(inputs, monkeypatch):
    estimator = model()
    policy, reliability, _ = elara._load_router_api()

    def masked_route(scores, labels, target, config, *, action):
        return np.ma.array(target[:, 1], mask=True), "fuse"

    monkeypatch.setattr(elara, "_load_router_api", lambda: (policy, reliability, masked_route))
    result = elara.ELARAKGAGuard().decide(
        estimator=estimator,
        expected_estimator_payload_sha256=estimator.payload_sha256,
        **inputs,
    )
    assert result.decision is Decision.ABSTAIN
    assert result.deployed_action == "retain_frozen"
    np.testing.assert_array_equal(result.deployed_scores, inputs["s_test"][:, 0])
    assert result.certificate["availability"] == "unavailable"
    assert not result.claim_eligible


def test_reliability_extraction_cannot_mutate_frozen_fallback(inputs, monkeypatch):
    original = inputs["s_test"].copy()
    policy, _, route = elara._load_router_api()

    def mutating_reliability(scores, labels):
        scores[:] = 0.0
        labels[:] = 0
        return {"val_auc": [0.9, 0.7], "best_auc": 0.9, "gap": 0.2, "disagreement": 0.1}

    monkeypatch.setattr(elara, "_load_router_api", lambda: (policy, mutating_reliability, route))
    result = elara.ELARAKGAGuard().decide(estimator=None, **inputs)
    assert result.decision is Decision.ABSTAIN
    np.testing.assert_array_equal(result.frozen_scores, original[:, 0])
    np.testing.assert_array_equal(inputs["s_val"], original)
    np.testing.assert_array_equal(inputs["y_val"], [0, 1, 0, 1])


def test_masked_evaluation_labels_are_not_silently_unsealed(inputs):
    result = elara.ELARAKGAGuard().decide(estimator=None, **inputs)
    labels = np.ma.array(inputs["y_val"], mask=True)
    with pytest.raises(ValueError, match="finite"):
        elara.evaluate_result(result, labels)


@pytest.mark.parametrize(
    "indices",
    [
        np.array([0.2, 1.2]),
        np.array([True, False]),
        np.ma.array([0, 1], mask=[False, True]),
        [0, True],
        [np.int64(0), np.bool_(True)],
        [0, 10**100],
    ],
)
def test_probe_identity_is_not_coerced_or_unmasked(inputs, indices):
    inputs.update(mode="target_label_light", y_test=inputs["y_val"], probe_indices=indices)
    with pytest.raises(ValueError, match="probe_indices"):
        elara.ELARAKGAGuard().decide(**inputs)


@pytest.mark.parametrize("val_auc", [np.ma.array([0.9, 0.7], mask=[True, False]), [float("nan"), 0.7], [0.9]])
def test_unavailable_reliability_cannot_select_a_frozen_expert(inputs, monkeypatch, val_auc):
    policy, _, route = elara._load_router_api()

    def reliability(scores, labels):
        return {"val_auc": val_auc, "best_auc": 0.9, "gap": 0.2, "disagreement": 0.1}

    monkeypatch.setattr(elara, "_load_router_api", lambda: (policy, reliability, route))
    with pytest.raises(ValueError, match="val_auc"):
        elara.ELARAKGAGuard().decide(**inputs)


def test_invalid_candidate_does_not_bypass_label_free_input_rejection(inputs, monkeypatch):
    policy, reliability, _ = elara._load_router_api()

    def invalid_route(*args, **kwargs):
        return np.full(4, np.nan), "fuse"

    monkeypatch.setattr(elara, "_load_router_api", lambda: (policy, reliability, invalid_route))
    with pytest.raises(ValueError, match="must not receive y_test"):
        elara.ELARAKGAGuard().decide(**inputs, y_test=inputs["y_val"])


@pytest.mark.parametrize("invalid", [None, {}, [], {"val_auc": "bad"}])
def test_missing_reliability_reports_explicit_validation_error(inputs, monkeypatch, invalid):
    policy, _, route = elara._load_router_api()
    monkeypatch.setattr(elara, "_load_router_api", lambda: (policy, lambda *args: invalid, route))
    with pytest.raises(ValueError, match="val_auc"):
        elara.ELARAKGAGuard().decide(**inputs)
