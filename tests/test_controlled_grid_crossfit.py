"""Regression tests for the canonical controlled-grid label flow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import kga
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_IMPLEMENTATION = ROOT / "docs/research/kbound/scripts/kbound_decide.py"
RECONCILIATION = ROOT / "scripts/reconcile_result_panels.py"


def _load_historical_implementation():
    spec = importlib.util.spec_from_file_location("historical_kbound_decide", HISTORICAL_IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reconciliation():
    spec = importlib.util.spec_from_file_location("controlled_grid_reconciliation", RECONCILIATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scored_outcome_cannot_change_its_crossfit_prediction_radius_or_action() -> None:
    """Changing B_i must not flow back into any quantity used to route cell i."""

    rng = np.random.default_rng(20260827)
    z = rng.normal(size=(20, 4))
    benefit = rng.normal(scale=0.2, size=20)
    sample_ids = [f"cell-{index:02d}" for index in range(len(benefit))]
    changed = benefit.copy()
    changed[7] += 1000.0

    # This control names the break caught by the regression.  The historical
    # two-stage LOO refit excludes B_7 from prediction_7, but B_7 enters the
    # models behind other residuals and therefore changes radius_7.
    historical = _load_historical_implementation()
    before_historical = historical.decide_kga(z, benefit, n_estimators=30)
    after_historical = historical.decide_kga(z, changed, n_estimators=30)
    assert before_historical[0][7] == after_historical[0][7]
    assert before_historical[1][7] != after_historical[1][7]

    assert hasattr(kga, "controlled_grid_crossfit"), "canonical three-way cross-fit API is missing"
    before = kga.controlled_grid_crossfit(
        z,
        benefit,
        sample_ids=sample_ids,
        alpha=0.1,
        n_folds=5,
        n_estimators=30,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    )
    after = kga.controlled_grid_crossfit(
        z,
        changed,
        sample_ids=sample_ids,
        alpha=0.1,
        n_folds=5,
        n_estimators=30,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    )

    assert before.prediction[7] == after.prediction[7]
    assert before.radius[7] == after.radius[7]
    assert before.action[7] == after.action[7]


def test_crossfit_is_permutation_stable_and_exposes_complete_provenance() -> None:
    rng = np.random.default_rng(41)
    z = rng.normal(size=(20, 3))
    benefit = rng.normal(size=20)
    sample_ids = [f"condition-{index:02d}" for index in range(20)]
    order = rng.permutation(20)

    expected = kga.controlled_grid_crossfit(z, benefit, sample_ids=sample_ids)
    permuted = kga.controlled_grid_crossfit(
        z[order],
        benefit[order],
        sample_ids=[sample_ids[index] for index in order],
    )
    inverse = np.argsort(order)

    assert np.array_equal(permuted.prediction[inverse], expected.prediction)
    assert np.array_equal(permuted.radius[inverse], expected.radius)
    assert np.array_equal(permuted.action[inverse], expected.action)
    protocol = expected.protocol
    assert protocol["status"] == "ok"
    assert protocol["effective_n_folds"] == 5
    assert protocol["gbrt"] == {
        "n_estimators": 250,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 0,
    }
    assert set(protocol["input_sha256"]) == {"Z", "B", "sample_ids"}
    assert set(protocol["implementation_sha256"]) == {
        "crossfit",
        "certificate",
        "policy",
        "numeric_validation",
    }
    assert all(len(value) == 64 for value in protocol["implementation_sha256"].values())
    assert set(protocol["software_versions"]) == {"python", "numpy", "scikit_learn"}
    assert all(
        set(fold) >= {"score_count", "estimator_fit_count", "residual_calibration_count"}
        for fold in protocol["folds"]
    )
    json.dumps(protocol, allow_nan=False)
    json.dumps(expected.to_dict(), allow_nan=False)


def test_crossfit_fails_closed_for_infeasible_or_malformed_inputs_and_rejects_duplicate_ids() -> None:
    small = kga.controlled_grid_crossfit(
        np.arange(22, dtype=float).reshape(11, 2),
        np.linspace(-0.2, 0.2, 11),
        sample_ids=[f"small-{index}" for index in range(11)],
    )
    malformed = kga.controlled_grid_crossfit(
        np.asarray([[0.0], [np.nan], [2.0]]),
        np.asarray([0.1, 0.2, 0.3]),
        sample_ids=["a", "b", "c"],
    )
    for result in (small, malformed):
        assert result.protocol["status"] == "fail_closed"
        assert np.isinf(result.radius).all()
        assert set(result.action) == {"ABSTAIN"}
        serialized = result.to_dict()
        assert serialized["prediction"] == [None] * len(result.prediction)
        assert serialized["radius"] == [None] * len(result.radius)
        assert set(serialized["prediction_status"]) == {"unavailable"}
        assert set(serialized["radius_status"]) == {"positive_infinity"}
        json.dumps(serialized, allow_nan=False)

    with pytest.raises(ValueError, match="sample_ids must be unique"):
        kga.controlled_grid_crossfit(
            np.arange(24, dtype=float).reshape(12, 2),
            np.linspace(-0.2, 0.2, 12),
            sample_ids=["duplicate"] * 12,
        )


@pytest.mark.parametrize(
    ("benefit", "sample_ids"),
    [
        (["not-a-number"] * 20, [f"id-{index}" for index in range(20)]),
        (np.linspace(-0.2, 0.2, 20), None),
        (np.linspace(-0.2, 0.2, 20), 7),
    ],
)
def test_crossfit_malformed_outer_inputs_fail_closed(benefit, sample_ids) -> None:
    result = kga.controlled_grid_crossfit(
        np.arange(40, dtype=float).reshape(20, 2),
        benefit,
        sample_ids=sample_ids,
    )
    assert result.protocol["status"] == "fail_closed"
    assert set(result.action) == {"ABSTAIN"}
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("parameter", ["alpha", "learning_rate", "subsample"])
def test_crossfit_nonfinite_hyperparameters_fail_closed(parameter: str) -> None:
    kwargs = {parameter: float("nan")}
    result = kga.controlled_grid_crossfit(
        np.arange(40, dtype=float).reshape(20, 2),
        np.linspace(-0.2, 0.2, 20),
        sample_ids=[f"id-{index}" for index in range(20)],
        **kwargs,
    )
    assert result.protocol["status"] == "fail_closed"
    json.dumps(result.to_dict(), allow_nan=False)


def test_crossfit_numpy_scalar_hyperparameters_are_normalized_for_strict_json() -> None:
    result = kga.controlled_grid_crossfit(
        np.arange(40, dtype=float).reshape(20, 2),
        np.linspace(-0.2, 0.2, 20),
        sample_ids=[f"id-{index}" for index in range(20)],
        alpha=np.float64(0.1),
        n_folds=np.int64(5),
        n_estimators=np.int64(10),
        max_depth=np.int64(2),
        learning_rate=np.float64(0.05),
        subsample=np.float64(0.8),
        random_state=np.int64(0),
    )
    assert result.protocol["status"] == "ok"
    json.dumps(result.to_dict(), allow_nan=False)


def test_grid_reconciliation_ignores_historical_bhat_and_labels_its_status(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    z = rng.normal(size=(20, 3))
    benefit = rng.normal(scale=0.2, size=20)

    def write_panel(directory: Path, historical_prediction: np.ndarray) -> None:
        directory.mkdir()
        records = []
        for index in range(20):
            records.append(
                {
                    "benchmark": "synthetic-grid",
                    "method": "tent",
                    "seed": 0,
                    "condition": f"condition-{index:02d}",
                    "Z": z[index].tolist(),
                    "Z_names": ["z0", "z1", "z2"],
                    "B": float(benefit[index]),
                    "a0": 0.5,
                    "a_adapted": float(0.5 + benefit[index]),
                    "b_hat": float(historical_prediction[index]),
                    "eps_conformal": 123.0,
                    "kga_decision": "FREEZE",
                }
            )
        (directory / "per_condition_synthetic_tent_seed0.json").write_text(
            json.dumps({"records": records}), encoding="utf-8"
        )

    original = tmp_path / "original"
    poisoned = tmp_path / "poisoned"
    write_panel(original, benefit)
    write_panel(poisoned, np.full(20, 1000.0))
    reconciliation = _load_reconciliation()

    expected = reconciliation._grid_panel(original, expected_seeds={0})
    actual = reconciliation._grid_panel(poisoned, expected_seeds={0})
    expected_score = expected["candidates"]["tent"]
    actual_score = actual["candidates"]["tent"]
    for key in (
        "regret",
        "adapt_count",
        "freeze_count",
        "abstain_count",
        "false_adapt_count",
        "point_beats_both",
    ):
        assert actual_score[key] == expected_score[key]

    expected_file = expected_score["per_file"][0]
    actual_file = actual_score["per_file"][0]
    assert actual_file["current_prediction_sha256"] == expected_file["current_prediction_sha256"]
    assert actual_file["current_radius_sha256"] == expected_file["current_radius_sha256"]
    assert actual_file["current_action_sha256"] == expected_file["current_action_sha256"]
    assert actual_file["historical_fields"]["b_hat"]["sha256"] != expected_file["historical_fields"]["b_hat"]["sha256"]
    assert actual_file["historical_fields"]["b_hat"]["current_authority"] is False
    assert actual["result_scope"] == "retrospective, opened, dependent, and constructed"

    changed = tmp_path / "changed"
    write_panel(changed, benefit)
    changed_path = changed / "per_condition_synthetic_tent_seed0.json"
    changed_payload = json.loads(changed_path.read_text(encoding="utf-8"))
    changed_payload["records"][7]["B"] += 1000.0
    changed_payload["records"][7]["a_adapted"] += 1000.0
    changed_path.write_text(json.dumps(changed_payload), encoding="utf-8")
    changed_panel = reconciliation._grid_panel(changed, expected_seeds={0})
    before_cell = expected_file["current_cell_authority"]["cells"][7]
    after_cell = changed_panel["candidates"]["tent"]["per_file"][0]["current_cell_authority"]["cells"][7]
    assert after_cell["sample_id"] == before_cell["sample_id"]
    assert after_cell["prediction"] == before_cell["prediction"]
    assert after_cell["radius"] == before_cell["radius"]
    assert after_cell["action"] == before_cell["action"]
