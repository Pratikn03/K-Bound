"""Missing NumPy observations must never become observed benefit evidence.

Synthetic fixtures only: no training run, dataset, or target artifact is read.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from deploy.api.kga_service import assess_kga_decision
from kga import Decision, FrozenLinearBenefitEstimator, KGA, decide_batch, decide_kga
from kga._validation import as_float_array
from kga.benefit import fit_frozen_linear_benefit_estimator
from kga.certificate import (
    conformal_radii_loo,
    conformal_split,
    empirical_bernstein,
    evalue_anytime,
    hoeffding,
    split_conformal_rank_radius,
    worst_group_conformal_radius,
)
from kga.routing import AnytimeMulticandidatePanel, bonferroni_multicandidate_route, route_panel


SCORES = np.linspace(0.1, 0.9, 20)


def _masked(values, *, all_missing: bool = False) -> np.ma.MaskedArray:
    values = np.asarray(values, dtype=float)
    mask = np.ones(values.shape, dtype=bool) if all_missing else np.zeros(values.shape, dtype=bool)
    mask.flat[0] = True
    return np.ma.array(values, mask=mask)


def _estimator() -> FrozenLinearBenefitEstimator:
    return FrozenLinearBenefitEstimator(
        feature_names=("x",),
        weights=np.array([1.0]),
        intercept=0.0,
        feature_center=np.array([0.0]),
        feature_scale=np.array([1.0]),
        residuals=np.zeros(19),
        evidence_schema_version="masked-input-fixture/1",
        protocol_sha256="a" * 64,
        fit_unit="development",
        calibration_unit="heldout_calibration",
    )


def _assert_invalidated(gate: KGA) -> None:
    report = gate.explain()
    for name in (
        "evidence", "certificate", "decision", "estimator_artifact_sha256",
        "protocol_sha256", "evidence_schema_version",
    ):
        assert report[name] is None, name
    with pytest.raises(ValueError, match="No certificate available"):
        gate.decide()


def test_mask_conversion_preserves_shape_data_and_input_mask_without_imputation() -> None:
    source = np.ma.array([[0.4, -0.4], [0.2, -0.2]], mask=[[True, False], [False, True]])
    data_before, mask_before = source.data.copy(), source.mask.copy()
    converted = as_float_array(source)
    np.testing.assert_array_equal(converted, [[np.nan, -0.4], [0.2, np.nan]])
    np.testing.assert_array_equal(source.data, data_before)
    np.testing.assert_array_equal(source.mask, mask_before)
    assert converted.shape == source.shape
    assert not np.ma.isMaskedArray(converted)


def test_nested_feature_and_row_masks_are_not_discarded() -> None:
    assert np.isnan(as_float_array([np.ma.array(-0.4, mask=True)])[0])
    rows = [np.ma.array([0.1, 0.2], mask=[False, True]), [0.3, 0.4]]
    np.testing.assert_array_equal(as_float_array(rows), [[0.1, np.nan], [0.3, 0.4]])


@pytest.mark.parametrize("delta", [-0.4, 0.4])
@pytest.mark.parametrize("method", ["ebern", "hoeffding", "evalue", "conformal"])
def test_direct_certificates_reject_masked_observations(delta: float, method: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        if method == "conformal":
            conformal_split(delta, _masked(np.zeros(19)))
        elif method == "evalue":
            evalue_anytime(_masked(np.full(200, delta)))
        else:
            estimator = empirical_bernstein if method == "ebern" else hoeffding
            estimator(_masked(np.full(200, delta)), benefit_range=2.0)


@pytest.mark.parametrize("radius", ["rank", "loo", "worst_group"])
def test_no_radius_wrapper_discards_a_calibration_mask(radius: str) -> None:
    residuals = _masked(np.zeros(19), all_missing=True)
    with pytest.raises(ValueError, match="finite"):
        if radius == "rank":
            split_conformal_rank_radius(residuals)
        elif radius == "loo":
            conformal_radii_loo(residuals)
        else:
            worst_group_conformal_radius([np.zeros(19), residuals], 0.1)


@pytest.mark.parametrize("delta", [-0.4, 0.4])
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.parametrize("operation", ["calib_scores", "test_scores", "residuals", "benefits", "probe"])
def test_masked_facade_attempt_clears_previous_authority(delta: float, all_missing: bool, operation: str) -> None:
    gate = KGA()
    gate.certify(delta_hat=delta, calib_residuals=np.zeros(19))
    assert gate.decide() is (Decision.ADAPT if delta > 0 else Decision.FREEZE)
    with pytest.raises(ValueError, match="finite"):
        if operation == "calib_scores":
            gate.evidence(_masked(SCORES, all_missing=all_missing), SCORES)
        elif operation == "test_scores":
            gate.evidence(SCORES, _masked(SCORES, all_missing=all_missing))
        elif operation == "residuals":
            gate.certify(delta_hat=delta, calib_residuals=_masked(np.zeros(19), all_missing=all_missing))
        elif operation == "benefits":
            gate.certify(scores=_masked(np.full(200, delta), all_missing=all_missing), benefit_range=2.0)
        else:
            # Validate the entire declared pool before subsampling it.
            gate.certify_probe(_masked(np.full(200, delta), all_missing=all_missing), k=1, benefit_range=2.0)
    _assert_invalidated(gate)


@pytest.mark.parametrize("delta", [-0.4, 0.4])
def test_masked_feature_cannot_reach_even_a_permissive_custom_predictor(delta: float) -> None:
    class PermissivePredictor(FrozenLinearBenefitEstimator):
        def predict(self, features, **kwargs) -> float:
            pytest.fail("a masked feature must be rejected before prediction")

    estimator = PermissivePredictor(**vars(_estimator()))
    gate = KGA()
    gate.certify(delta_hat=delta, calib_residuals=np.zeros(19))
    gate.decide()
    with pytest.raises(ValueError, match="finite"):
        gate.certify_evidence(
            estimator,
            protocol_sha256=estimator.protocol_sha256,
            evidence_schema_version=estimator.evidence_schema_version,
            features={"x": np.ma.array(delta, mask=True)},
        )
    _assert_invalidated(gate)


@pytest.mark.parametrize("delta", [-0.4, 0.4])
def test_custom_estimator_residual_mask_survives_the_facade_boundary(delta: float) -> None:
    class CustomEstimator:
        feature_names = ("x",)
        evidence_schema_version = "masked-input-fixture/1"
        protocol_sha256 = "a" * 64
        artifact_sha256 = "b" * 64
        residuals = _masked(np.zeros(19), all_missing=True)

        def predict(self, features, **kwargs) -> float:
            return delta

    estimator = CustomEstimator()
    gate = KGA()
    with pytest.raises(ValueError, match="finite"):
        gate.certify_evidence(
            estimator,
            protocol_sha256=estimator.protocol_sha256,
            evidence_schema_version=estimator.evidence_schema_version,
            features={"x": delta},
        )
    _assert_invalidated(gate)


@pytest.mark.parametrize("field", ["weights", "feature_center", "feature_scale", "residuals"])
def test_frozen_estimator_construction_and_dict_loading_do_not_erase_masks(field: str) -> None:
    estimator = _estimator()
    masked = _masked(getattr(estimator, field))
    with pytest.raises(ValueError, match="finite"):
        replace(estimator, **{field: masked})
    payload = estimator.to_dict()
    payload[field] = masked
    with pytest.raises(ValueError, match="finite"):
        FrozenLinearBenefitEstimator.from_dict(payload)


def test_reference_estimator_direct_prediction_rejects_masked_scalar_features() -> None:
    estimator = _estimator()
    with pytest.raises(ValueError, match="finite"):
        estimator.predict(
            {"x": np.ma.array(-0.4, mask=True)},
            evidence_schema_version=estimator.evidence_schema_version,
            protocol_sha256=estimator.protocol_sha256,
        )


@pytest.mark.parametrize("field", ["x_fit", "y_fit", "x_calibration", "y_calibration"])
def test_reference_estimator_fitting_rejects_masked_training_or_calibration_data(field: str) -> None:
    inputs = {
        "x_fit": np.arange(20.0).reshape(-1, 1),
        "y_fit": np.arange(20.0),
        "x_calibration": np.arange(20.0, 40.0).reshape(-1, 1),
        "y_calibration": np.arange(20.0, 40.0),
    }
    inputs[field] = _masked(inputs[field])
    with pytest.raises(ValueError, match="finite"):
        fit_frozen_linear_benefit_estimator(
            **inputs,
            feature_names=("x",),
            evidence_schema_version="masked-input-fixture/1",
            protocol_sha256="a" * 64,
        )


@pytest.mark.parametrize("masked_field", ["estimate", "radius"])
def test_batch_mask_abstains_per_cell_preserving_unmasked_decisions_and_shape(masked_field: str) -> None:
    delta = np.array([[0.4, -0.4], [0.4, -0.4]])
    epsilon = np.full((2, 2), 0.1)
    mask = [[False, False], [True, True]]
    if masked_field == "estimate":
        delta = np.ma.array(delta, mask=mask)
    else:
        epsilon = np.ma.array(epsilon, mask=mask)
    assert decide_batch(delta, epsilon).tolist() == [["ADAPT", "FREEZE"], ["ABSTAIN", "ABSTAIN"]]
    assert decide_batch(delta, np.ma.array(0.1, mask=True)).tolist() == [["ABSTAIN"] * 2] * 2


@pytest.mark.parametrize("field", ["estimate", "benefit"])
@pytest.mark.parametrize("calibration", ["loo", "in_pool"])
def test_replay_rejects_masked_locked_rows_without_dropping_them(field: str, calibration: str) -> None:
    delta, truth = np.full(20, -0.4), np.full(20, -0.4)
    if field == "estimate":
        delta = _masked(delta)
    else:
        truth = _masked(truth)
    with pytest.raises(ValueError, match="finite"):
        decide_kga(delta, truth, calibration=calibration)


@pytest.mark.parametrize("delta", [-0.8, 0.8])
@pytest.mark.parametrize("field", ["deploy", "cal_scores", "cal_truth"])
def test_masked_candidate_is_unavailable_without_reducing_the_family(delta: float, field: str) -> None:
    deploy = np.array([delta, 0.2])
    estimates, truth = np.zeros((2, 19)), np.zeros((2, 19))
    if field == "deploy":
        deploy = _masked(deploy)
    elif field == "cal_scores":
        estimates = _masked(estimates)
    else:
        truth = _masked(truth)
    result = route_panel(deploy, estimates, truth, alpha=0.1)
    assert result.selected == 1 and result.committed
    assert result.bonferroni_alpha == 0.05
    assert len(result.certificates) == 2
    assert not result.certificates[0].available
    assert result.certificates[1].available


@pytest.mark.parametrize("delta", [-0.4, 0.4])
def test_fully_masked_routing_calibration_abstains_for_the_original_family(delta: float) -> None:
    calibration = _masked(np.zeros((2, 19)), all_missing=True)
    result = route_panel(np.array([delta, delta]), calibration, calibration, alpha=0.1)
    assert result.decision == "abstain"
    assert result.selected is None and not result.committed and not result.feasible
    assert result.bonferroni_alpha == 0.05


@pytest.mark.parametrize("selector", ["argmax_lcb", "first_positive"])
def test_masked_positive_lcb_is_not_selectable(selector: str) -> None:
    assert bonferroni_multicandidate_route(_masked([0.8, 0.2]), selector=selector) == 1
    assert bonferroni_multicandidate_route(_masked([0.8, 0.2], all_missing=True), selector=selector) is None


def test_masked_anytime_step_is_rejected_before_any_candidate_advances() -> None:
    panel = AnytimeMulticandidatePanel(2)
    panel.update([0.4, 0.3])
    states = [vars(process).copy() for process in panel._procs]
    step = np.ma.array([0.4, 0.3], mask=[False, True])
    with pytest.raises(ValueError, match="finite"):
        panel.update(step)
    assert panel.steps == 1
    assert [vars(process) for process in panel._procs] == states


@pytest.mark.parametrize("delta", [-0.4, 0.4])
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.parametrize("field", ["calib_scores", "test_scores", "benefit_scores", "calib_residuals"])
def test_service_missing_masked_evidence_abstains_and_retains_frozen(delta: float, all_missing: bool, field: str) -> None:
    inputs = {"calib_scores": SCORES, "test_scores": SCORES}
    kwargs = {"cert_mode": "full", "delta_hat": delta, "calib_residuals": np.zeros(19)}
    if field in inputs:
        inputs[field] = _masked(inputs[field], all_missing=all_missing)
    elif field == "benefit_scores":
        kwargs = {
            "cert_mode": "full", "benefit_scores": _masked(np.full(200, delta), all_missing=all_missing),
            "benefit_range": 2.0,
        }
    else:
        kwargs[field] = _masked(kwargs[field], all_missing=all_missing)
    result = assess_kga_decision(**inputs, **kwargs)
    frozen, candidate = object(), object()
    assert result.decision is Decision.ABSTAIN
    assert result.availability == "unavailable"
    assert result.certificate is None and result.reason
    assert result.model_action == "retain_frozen"
    assert result.select_predictor(frozen, candidate) is frozen


@pytest.mark.parametrize("delta", [-0.4, 0.4])
def test_inactive_masks_preserve_plain_array_results(delta: float) -> None:
    masked_scores = np.ma.array(SCORES, mask=False)
    masked_residuals = np.ma.array(np.zeros(19), mask=False)
    plain = assess_kga_decision(SCORES, SCORES, cert_mode="full", delta_hat=delta, calib_residuals=np.zeros(19))
    masked = assess_kga_decision(
        masked_scores, masked_scores, cert_mode="full", delta_hat=delta, calib_residuals=masked_residuals,
    )
    assert masked == plain
    benefits = np.full(200, delta)
    for estimator in (empirical_bernstein, hoeffding):
        assert estimator(np.ma.array(benefits, mask=False), benefit_range=2.0) == estimator(benefits, benefit_range=2.0)
    assert evalue_anytime(np.ma.array(benefits, mask=False)) == evalue_anytime(benefits)
