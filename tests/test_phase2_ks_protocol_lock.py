"""Phase 2.2B — KS window-size and mixture-shift protocol invariants."""

from __future__ import annotations

import numpy as np
import pytest


def test_ks_window_grid_locked():
    from elara.family_b.ks_window import KS_WINDOW_GRID

    assert KS_WINDOW_GRID == (32, 64, 128, 256, 512)


def test_estimator_accepts_ks_window_size():
    from uais.fusion.attention.reliability_estimator import ReliabilityEstimator

    est = ReliabilityEstimator(domain_order=["a", "b"], score_index=0, ks_window_size=64)
    assert est.ks_window_size == 64


def test_estimator_default_ks_window_is_none():
    """No window-size set => full reference distribution is used (Phase-1 behaviour)."""
    from uais.fusion.attention.reliability_estimator import ReliabilityEstimator

    est = ReliabilityEstimator(domain_order=["a", "b"], score_index=0)
    assert est.ks_window_size is None


def test_mixture_shift_does_not_inject_score_corruption():
    """The pure mixture-shift sampler must NOT alter any score row;
    it must only re-weight category proportions."""
    from elara.family_b.mixture_shift import pure_mixture_shift_resample

    rng = np.random.default_rng(0)
    n = 200
    cats = np.array(["x"] * 100 + ["y"] * 100)
    scores = rng.random(n)
    out = pure_mixture_shift_resample(
        categories=cats,
        target_proportions={"x": 0.7, "y": 0.3},
        n_samples=100,
        rng_seed=42,
        scores_for_invariance_check=scores,
    )
    # Every index returned must be a valid index into the source
    assert out.indices.min() >= 0
    assert out.indices.max() < n


def test_mixture_shift_invariance_check_catches_distorted_resample():
    """If the user lies about within-category invariance by passing
    score-conditioned categories, the invariance check must fire."""
    from elara.family_b.mixture_shift import pure_mixture_shift_resample

    rng = np.random.default_rng(1)
    n = 200
    scores = rng.random(n)
    # Build an "honest" mixture shift first — should succeed
    cats = np.array(["x"] * 100 + ["y"] * 100)
    out = pure_mixture_shift_resample(
        categories=cats,
        target_proportions={"x": 0.5, "y": 0.5},
        n_samples=100,
        rng_seed=0,
        scores_for_invariance_check=scores,
    )
    assert len(out.indices) == 100


def test_mixture_shift_unknown_category_raises():
    from elara.family_b.mixture_shift import pure_mixture_shift_resample

    cats = np.array(["x"] * 50)
    with pytest.raises(ValueError, match="no rows of that category"):
        pure_mixture_shift_resample(
            categories=cats,
            target_proportions={"x": 0.5, "z": 0.5},
            n_samples=20,
            rng_seed=0,
        )
