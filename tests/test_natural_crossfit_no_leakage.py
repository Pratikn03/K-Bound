from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "docs/research/kbound/scripts/kbound_decide.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("natural_crossfit_kbound_decide", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KB = _load_module()


def test_scored_label_cannot_change_its_prediction_radius_or_decision():
    rng = np.random.default_rng(20260827)
    z = rng.normal(size=(20, 4))
    benefit = rng.normal(scale=0.2, size=20)
    sample_ids = [f"cell-{index:02d}" for index in range(len(benefit))]

    bhat_before, eps_before, decision_before = KB.decide_kga_crossfit(
        z,
        benefit,
        sample_ids=sample_ids,
        n_estimators=30,
    )
    changed = benefit.copy()
    changed[7] += 1000.0
    bhat_after, eps_after, decision_after = KB.decide_kga_crossfit(
        z,
        changed,
        sample_ids=sample_ids,
        n_estimators=30,
    )

    assert bhat_after[7] == bhat_before[7]
    assert eps_after[7] == eps_before[7]
    assert decision_after[7] == decision_before[7]


def test_crossfit_is_permutation_invariant_when_stable_ids_are_supplied():
    rng = np.random.default_rng(41)
    z = rng.normal(size=(20, 3))
    benefit = rng.normal(size=20)
    sample_ids = [f"condition-{index}" for index in range(20)]
    order = rng.permutation(20)

    expected = KB.decide_kga_crossfit(
        z, benefit, sample_ids=sample_ids, n_estimators=20
    )
    permuted = KB.decide_kga_crossfit(
        z[order],
        benefit[order],
        sample_ids=[sample_ids[index] for index in order],
        n_estimators=20,
    )
    inverse = np.argsort(order)

    assert np.allclose(permuted[0][inverse], expected[0])
    assert np.allclose(permuted[1][inverse], expected[1])
    assert np.array_equal(permuted[2][inverse], expected[2])


def test_undersized_three_way_pool_fails_closed():
    z = np.arange(22, dtype=float).reshape(11, 2)
    benefit = np.linspace(-0.2, 0.2, 11)

    _, epsilon, decision = KB.decide_kga_crossfit(z, benefit, n_estimators=10)

    assert np.isinf(epsilon).all()
    assert set(decision) == {"ABSTAIN"}
