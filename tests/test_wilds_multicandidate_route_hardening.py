"""Release regressions for the WILDS agreement-only Route-B implementation."""

from __future__ import annotations

import json
import math
import numbers
import subprocess
import sys

import numpy as np
import pytest

from experiments.kbound.wilds import analysis as route_analysis


def _full_rank_binary_panel(seed: int = 17, n: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(seed)
    panel = rng.integers(0, 2, size=(4, n), dtype=np.int8)
    # Deterministic random rows are overwhelmingly full rank; make that premise
    # explicit so a changed fixture cannot weaken the tests below.
    centered = panel - panel.mean(axis=1, keepdims=True)
    assert np.linalg.matrix_rank(centered) == 4
    return panel


def _route(panel: np.ndarray, **overrides):
    kwargs = {
        "tau_star": 0.08,
        "kappa": 0.0,
        "min_D": 1,
        "task_type": "binary_classification",
        "n_classes": 2,
        "objective": "accuracy",
        "anchor_above_chance": True,
    }
    kwargs.update(overrides)
    return route_analysis.multicandidate_route(panel, **kwargs)


def _numeric_leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _numeric_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _numeric_leaves(child)
    elif isinstance(value, (bool, np.bool_)) or value is None:
        return
    elif isinstance(value, numbers.Real):
        yield float(value)


def test_detectability_undersized_exact_rank_is_null_and_never_calls_estimator(monkeypatch):
    records = [
        {
            "Z": [float(index), float(index % 2)],
            "B": -0.1 if index % 2 else 0.1,
        }
        for index in range(8)
    ]

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("undersized exact-rank estimator must not run")

    monkeypatch.setattr(route_analysis, "decide_kga", should_not_run)
    result = route_analysis.detectability_analysis(records, ["z0", "z1"], alpha=0.1)

    assert result["certificate_calibration_status"] == "INFEASIBLE_UNDERSIZED_EXACT_RANK"
    assert result["certificate_calibration_feasible"] is False
    assert result["certificate_eps"] is None
    assert result["certificate_eps_min"] is None
    assert result["certificate_eps_max"] is None
    json.dumps(result, allow_nan=False)


def test_retired_route_c_prototype_is_not_exposed():
    assert not hasattr(route_analysis, "smooth_drift_route")


def test_spectral_sign_and_scale_cannot_change_route_decision(monkeypatch):
    """The arbitrary-sign, unbounded spectral fit is tau-only, never decisional."""
    panel = _full_rank_binary_panel()
    bounded = np.array([0.20, 0.82, 0.10, -0.15])
    monkeypatch.setattr(route_analysis, "_minor_estimator", lambda _C: bounded.copy())

    monkeypatch.setattr(
        route_analysis,
        "_rankone_fit_offdiag",
        lambda C: (np.array([25.0, -19.0, 12.0, -31.0]), 0.0),
    )
    positive_orientation = _route(panel)
    monkeypatch.setattr(
        route_analysis,
        "_rankone_fit_offdiag",
        lambda C: (np.array([-25.0, 19.0, -12.0, 31.0]), 0.0),
    )
    negative_orientation = _route(panel)

    assert positive_orientation["decision"] == "ADAPT"
    assert positive_orientation["choice"] == 1
    for key in ("decision", "choice", "anchor_b0", "h_hat", "margin", "b_decision", "b_hat"):
        assert positive_orientation[key] == negative_orientation[key]
    assert positive_orientation["b_hat"] == pytest.approx(bounded)
    assert max(abs(x) for x in positive_orientation["b_hat"]) <= 1.0


def test_binary_label_coding_flip_is_invariant(monkeypatch):
    panel = _full_rank_binary_panel(seed=23)
    bounded = np.array([0.15, 0.75, 0.05, -0.20])
    monkeypatch.setattr(route_analysis, "_minor_estimator", lambda _C: bounded.copy())
    monkeypatch.setattr(
        route_analysis,
        "_rankone_fit_offdiag",
        lambda C: (np.array([999.0, -999.0, 4.0, -4.0]), 0.0),
    )

    original = _route(panel)
    relabelled = _route(1 - panel)
    for key in ("decision", "choice", "tau", "h_hat", "margin", "b_decision", "anchor_b0"):
        assert original[key] == relabelled[key]


def test_bounded_minor_estimator_recovers_an_exact_oriented_system():
    truth = np.array([0.25, 0.80, -0.55, 0.40])
    C = np.outer(truth, truth)
    np.fill_diagonal(C, 0.0)
    estimate = route_analysis._minor_estimator(C)

    assert estimate == pytest.approx(truth)
    assert np.isfinite(estimate).all()
    assert np.max(np.abs(estimate)) <= 1.0


def test_successful_route_has_only_finite_bounded_statistics(monkeypatch):
    panel = _full_rank_binary_panel(seed=31)
    bounded = np.array([0.20, 0.85, 0.05, -0.10])
    monkeypatch.setattr(route_analysis, "_minor_estimator", lambda _C: bounded.copy())
    monkeypatch.setattr(
        route_analysis,
        "_rankone_fit_offdiag",
        lambda C: (np.full(C.shape[0], 500.0), 0.01),
    )
    result = _route(panel)

    assert result["status"] == "OK"
    assert result["scorable"] is True
    assert result["b_hat"] == result["b_tilde"] == result["b_decision"]
    assert all(math.isfinite(value) for value in _numeric_leaves(result))
    assert all(-1.0 <= value <= 1.0 for value in result["b_decision"])
    # Release artifacts use strict RFC-compatible JSON; NaN/Infinity must fail.
    json.dumps(result, allow_nan=False)


def test_unbounded_decision_estimator_fails_closed(monkeypatch):
    panel = _full_rank_binary_panel(seed=37)
    monkeypatch.setattr(
        route_analysis,
        "_minor_estimator",
        lambda _C: np.array([0.2, 1.01, 0.1, -0.1]),
    )
    monkeypatch.setattr(
        route_analysis,
        "_rankone_fit_offdiag",
        lambda C: (np.zeros(C.shape[0]), 0.0),
    )
    result = _route(panel)

    assert result["decision"] == "ERROR"
    assert result["status"] == "ERROR"
    assert result["scorable"] is False


def test_nonfinite_spectral_fit_fails_closed_even_though_it_is_not_decisional(monkeypatch):
    panel = _full_rank_binary_panel(seed=39)
    monkeypatch.setattr(
        route_analysis,
        "_rankone_fit_offdiag",
        lambda C: (np.full(C.shape[0], np.nan), 0.0),
    )
    result = _route(panel)

    assert result["decision"] == "ERROR"
    assert result["status"] == "ERROR"
    assert result["scorable"] is False


def test_multiclass_counterexample_is_explicitly_unsupported():
    panel = np.array(
        [
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
            [0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1],
        ]
    )
    result = _route(panel)

    assert result["decision"] == "ABSTAIN"
    assert result["status"] == "UNSUPPORTED"
    assert result["scorable"] is False
    assert "binary identity" in result["reason"]


def test_nonaccuracy_objective_is_explicitly_unsupported():
    result = _route(_full_rank_binary_panel(seed=41), objective="macro_f1")

    assert result["decision"] == "ABSTAIN"
    assert result["status"] == "UNSUPPORTED"
    assert result["scorable"] is False
    assert "objective" in result["reason"]


def test_exact_duplicate_is_rejected_before_nominal_candidate_count():
    panel = _full_rank_binary_panel(seed=43)[:3].copy()
    panel[2] = panel[0]
    result = _route(panel)

    assert result["decision"] == "ABSTAIN"
    assert result["status"] == "DEGENERATE_CANDIDATES"
    assert result["scorable"] is False
    assert result["duplicate_candidate_pairs"] == [[0, 2]]
    assert "M>=4" not in result["reason"]


def test_unique_but_rank_deficient_candidates_are_rejected():
    row_a = np.tile([0, 1], 128)
    row_b = 1 - row_a
    row_c = np.tile([0, 0, 1, 1], 64)
    row_d = 1 - row_c
    panel = np.stack([row_a, row_b, row_c, row_d])
    result = _route(panel)

    assert result["duplicate_candidate_pairs"] == []
    assert result["effective_candidate_rank"] == 2
    assert result["decision"] == "ABSTAIN"
    assert result["status"] == "DEGENERATE_CANDIDATES"
    assert result["scorable"] is False


def test_missing_anchor_premise_fails_closed():
    result = route_analysis.multicandidate_route(
        _full_rank_binary_panel(seed=47),
        objective="accuracy",
        n_classes=2,
    )
    assert result["decision"] == "ABSTAIN"
    assert result["status"] == "UNSUPPORTED"
    assert result["scorable"] is False


def test_error_decision_cannot_be_scored_as_freeze():
    with pytest.raises(ValueError, match="unscorable routing decision"):
        route_analysis.policy_metrics(
            np.array(["ADAPT", "ERROR"]),
            np.array([0.5, 0.5]),
            np.array([0.6, 0.4]),
        )


def test_importing_analysis_does_not_import_matplotlib():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from experiments.kbound.wilds import analysis; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
