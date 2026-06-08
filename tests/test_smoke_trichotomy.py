"""Tests for the hermetic K-Bound trichotomy smoke (scripts.kbound.smoke_trichotomy).

These exercise the REAL ``kga`` package end to end: the synthetic archive is
written into ``tmp_path``, the driver runs the actual evidence -> certificate ->
decision pipeline, and we assert the helpful / harmful / unknowable regimes map
to ADAPT / FREEZE / ABSTAIN.  Pure numpy + kga; well under five seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kga import Decision
from scripts.kbound.make_synth_archive import (
    DEFAULT_DET_NAMES,
    make_synth_archive,
)
from scripts.kbound.smoke_trichotomy import (
    EXPECTED_DECISION,
    paired_benefits_from_task,
    run_smoke,
)

# Expected archive keys (must match the real ELARA-U score archive schema).
_EXPECTED_KEYS = {"Sval", "yval", "Stest", "ytest", "det_names", "val_auc", "domain"}


def test_synth_archive_schema_matches_real(tmp_path: Path) -> None:
    """The synthetic ``.npz`` files carry exactly the real archive's keys/shapes."""
    paths = make_synth_archive(tmp_path / "score_archive")
    assert len(paths) == 3

    for p in paths:
        d = np.load(p, allow_pickle=True)
        assert set(d.files) == _EXPECTED_KEYS
        n_val, n_det = d["Sval"].shape
        n_test, n_det_t = d["Stest"].shape
        assert n_det == n_det_t == len(DEFAULT_DET_NAMES)
        assert d["yval"].shape == (n_val,)
        assert d["ytest"].shape == (n_test,)
        assert d["val_auc"].shape == (n_det,)
        assert d["Sval"].dtype == np.float64
        assert d["Stest"].dtype == np.float64
        assert d["yval"].dtype == np.int64
        assert d["ytest"].dtype == np.int64
        # Labels are binary; both classes present (needed for a real decision).
        assert set(np.unique(d["ytest"]).tolist()) == {0, 1}
        assert str(d["domain"]) == "synthetic"


def test_synth_archive_is_deterministic(tmp_path: Path) -> None:
    """Re-generating with the same seed yields byte-identical arrays."""
    p1 = make_synth_archive(tmp_path / "a", seed=0)
    p2 = make_synth_archive(tmp_path / "b", seed=0)
    for a, b in zip(p1, p2):
        da, db = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        for k in _EXPECTED_KEYS - {"det_names", "domain"}:
            np.testing.assert_array_equal(da[k], db[k])


def test_paired_benefit_signs_separate_regimes(tmp_path: Path) -> None:
    """Helpful/harmful/unknowable produce clearly +/-/~0 mean paired benefits."""
    paths = make_synth_archive(tmp_path / "score_archive")
    means = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        name = Path(p).stem
        b = paired_benefits_from_task(d["Stest"], d["ytest"], d["val_auc"])
        means[name] = float(np.mean(b))

    # Well-separated, not lucky: helpful strongly +, harmful strongly -,
    # unknowable near zero (an order of magnitude smaller in absolute value).
    assert means["synth_helpful"] > 0.15
    assert means["synth_harmful"] < -0.15
    assert abs(means["synth_unknowable"]) < 0.05


def test_run_smoke_maps_regimes_to_expected_decisions(tmp_path: Path) -> None:
    """The full real-KGA pipeline returns ADAPT / FREEZE / ABSTAIN as required."""
    summary = run_smoke(out_dir=str(tmp_path))
    assert summary["status"] == "PASS"
    assert summary["n_tasks"] == 3
    assert summary["failures"] == []

    by_regime = {r["regime"]: r for r in summary["tasks"]}
    assert by_regime["helpful"]["decision"] == Decision.ADAPT.value
    assert by_regime["harmful"]["decision"] == Decision.FREEZE.value
    assert by_regime["unknowable"]["decision"] == Decision.ABSTAIN.value

    # Certificate geometry is consistent with each decision (robustness check).
    assert by_regime["helpful"]["lower"] > 0.0  # ADAPT: lower bound > 0
    assert by_regime["harmful"]["upper"] < 0.0  # FREEZE: upper bound < 0
    assert by_regime["unknowable"]["lower"] <= 0.0 <= by_regime["unknowable"]["upper"]
    for r in summary["tasks"]:
        assert r["method"] == "ebern"
        assert r["epsilon"] >= 0.0


def test_expected_decision_table_is_complete() -> None:
    """The regime->decision contract covers all three trichotomy outcomes."""
    assert set(EXPECTED_DECISION.values()) == {
        Decision.ADAPT,
        Decision.FREEZE,
        Decision.ABSTAIN,
    }
