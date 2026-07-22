"""Golden tests for the canonical kbound_repro.metrics library.

Every expected value below is hand-computed from the definitions in the module
docstring so the test doubles as executable documentation of the boundary.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Import the toolkit without requiring it to be pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbound_repro import metrics as M  # noqa: E402

# A fixed golden fixture that exercises the Delta == 0 boundary case.
DEC = ["adapt", "adapt", "freeze", "abstain", "adapt"]
DELTA = [0.10, -0.20, -0.05, 0.30, 0.00]


def test_boundary_constant():
    assert M.FALSE_ADAPT_BOUNDARY == "delta_le_0"


def test_action_counts_are_integers():
    c = M.action_counts(DEC)
    assert c == {"adapt": 3, "freeze": 1, "abstain": 1}
    assert all(isinstance(v, int) for v in c.values())


def test_regret_vector_golden():
    reg = M.regret_vector(DEC, DELTA)
    assert reg.tolist() == pytest.approx([0.0, 0.20, 0.0, 0.30, 0.0])
    assert (reg >= 0).all()  # regret is provably non-negative


def test_policy_and_fixed_regrets_golden():
    assert M.policy_regret(DEC, DELTA) == pytest.approx(0.10)
    assert M.regret_to_oracle(DEC, DELTA) == pytest.approx(0.10)
    assert M.always_adapt_regret(DELTA) == pytest.approx(0.05)
    assert M.always_freeze_regret(DELTA) == pytest.approx(0.08)


def test_false_adapt_boundary_includes_zero():
    # c1 (adapt, -0.20) and c4 (adapt, 0.00) are both false adapts under <= 0.
    assert M.false_adapt_unconditional(DEC, DELTA) == pytest.approx(2 / 5)
    assert M.false_adapt_conditional(DEC, DELTA) == pytest.approx(2 / 3)


def test_conditional_false_adapt_has_distinct_denominator():
    # FA_c denominator is the number of ADAPT decisions, not n.
    fa_u = M.false_adapt_unconditional(DEC, DELTA)
    fa_c = M.false_adapt_conditional(DEC, DELTA)
    assert fa_c != fa_u
    assert fa_c == pytest.approx(fa_u * 5 / 3)  # n / n_adapt rescaling


def test_fa_c_none_when_no_adapt():
    assert M.false_adapt_conditional(["freeze", "abstain"], [0.1, -0.1]) is None


def test_legacy_strict_boundary_differs_on_ties():
    # The retired "< 0" boundary would miss the Delta == 0 tie -> 1/5 not 2/5.
    import numpy as np

    acts = np.asarray([a.lower() for a in DEC])
    d = np.asarray(DELTA, dtype=float)
    legacy_fa_u = float(np.mean((acts == "adapt") & (d < 0.0)))
    assert legacy_fa_u == pytest.approx(1 / 5)
    assert M.false_adapt_unconditional(DEC, DELTA) == pytest.approx(2 / 5)


def test_decision_summary_retains_counts():
    s = M.decision_summary(DEC, DELTA).to_dict()
    assert s["counts"] == {"adapt": 3, "freeze": 1, "abstain": 1}
    assert s["n"] == 5
    assert s["false_adapt_boundary"] == "delta_le_0"
    assert s["fa_u"] == pytest.approx(2 / 5)
    assert s["mean_realized_benefit"] == pytest.approx(-0.02)


def test_wilson_interval_literature_value():
    lo, hi = M.wilson_interval(8, 10, confidence=0.95)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_interval_edge_zero_successes():
    lo, hi = M.wilson_interval(0, 10, confidence=0.95)
    assert lo == 0.0
    assert 0.25 < hi < 0.30


def test_holm_correction_golden():
    out = M.holm_correction({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06)
    assert out["b"]["p_holm"] == pytest.approx(0.06)
    assert out["a"]["reject"] is True
    assert out["b"]["reject"] is False
    assert out["c"]["reject"] is False


def test_paired_bootstrap_deterministic():
    r = M.paired_bootstrap_diff_ci([1, 1, 1, 1], [0, 0, 0, 0], seed=0)
    assert r["diff"] == pytest.approx(1.0)
    assert r["ci_low"] == pytest.approx(1.0)
    assert r["ci_high"] == pytest.approx(1.0)
    assert r["excludes_zero"] is True


def test_paired_bootstrap_no_difference():
    r = M.paired_bootstrap_diff_ci([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], seed=0)
    assert r["diff"] == pytest.approx(0.0)
    assert r["excludes_zero"] is False


def test_beats_both_structure_and_fa_consistency():
    out = M.beats_both(DEC, DELTA, n_boot=2000, seed=0)
    assert out["fa_u"] == pytest.approx(2 / 5)
    assert set(out) >= {"vs_always_adapt", "vs_always_freeze", "beats_both"}
    assert isinstance(out["beats_both"], bool)


def test_input_validation():
    with pytest.raises(ValueError):
        M.false_adapt_unconditional(["adapt"], [0.1, 0.2])  # length mismatch
    with pytest.raises(ValueError):
        M.action_counts(["teleport"])  # unknown action
    with pytest.raises(ValueError):
        M.regret_vector([], [])  # empty
    with pytest.raises(ValueError):
        M.wilson_interval(11, 10)  # successes > n


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
