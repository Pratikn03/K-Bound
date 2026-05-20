"""Tests for the Double-ML causal reliability attribution module."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.causal_attribution import (
    DomainCausalEffect,
    estimate_all_domain_effects,
    estimate_domain_causal_effect,
    estimate_per_sample_cate,
)


def _make_synthetic_panel(
    *,
    n_samples: int = 300,
    n_domains: int = 3,
    n_features: int = 4,
    true_effect_per_domain: tuple[float, ...] = (0.6, 0.0, -0.3),
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct features/masks/reliability/predictions/categories with known effects."""
    if len(true_effect_per_domain) != n_domains:
        raise ValueError("len(true_effect_per_domain) must equal n_domains")
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, size=(n_samples, n_domains, n_features)).astype(np.float32)
    masks = np.zeros((n_samples, n_domains), dtype=bool)
    reliability = rng.uniform(0.2, 0.9, size=(n_samples, n_domains)).astype(np.float32)
    # Predictions linear in per-domain reliability, plus a confounder via features[:,:,0].
    confounder = features[:, :, 0].sum(axis=1)
    predictions = 0.5 + 0.1 * confounder
    for d, effect in enumerate(true_effect_per_domain):
        predictions = predictions + effect * reliability[:, d]
    predictions = predictions + rng.normal(0.0, noise, size=n_samples)
    categories = rng.choice(["a", "b", "c"], size=n_samples)
    return features, masks, reliability, predictions, categories


def test_estimate_domain_causal_effect_recovers_known_positive_effect():
    features, masks, reliability, predictions, categories = _make_synthetic_panel(
        true_effect_per_domain=(0.7, 0.0, 0.0), noise=0.03, seed=1
    )
    effect = estimate_domain_causal_effect(
        features,
        masks,
        reliability,
        predictions,
        domain_index=0,
        domain_name="d0",
        score_index=0,
        n_splits=5,
        random_state=1,
        category_codes=categories,
    )
    assert isinstance(effect, DomainCausalEffect)
    assert 0.5 < effect.ate < 0.9
    assert effect.ate_std_error > 0.0
    assert effect.ate_ci_low < effect.ate < effect.ate_ci_high
    assert effect.p_value < 0.01


def test_estimate_domain_causal_effect_recovers_known_negative_effect():
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(
        true_effect_per_domain=(0.0, 0.0, -0.5), noise=0.05, seed=2
    )
    effect = estimate_domain_causal_effect(
        features,
        masks,
        reliability,
        predictions,
        domain_index=2,
        domain_name="d2",
        score_index=0,
        n_splits=5,
        random_state=2,
    )
    assert -0.7 < effect.ate < -0.3
    assert effect.p_value < 0.05


def test_estimate_domain_causal_effect_returns_near_zero_for_null_effect():
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(
        true_effect_per_domain=(0.5, 0.0, 0.0), noise=0.05, seed=3
    )
    effect = estimate_domain_causal_effect(
        features,
        masks,
        reliability,
        predictions,
        domain_index=1,
        domain_name="d1",
        score_index=0,
        n_splits=5,
        random_state=3,
    )
    # Confidence interval must straddle zero for a true-null domain.
    assert effect.ate_ci_low < 0.05
    assert effect.ate_ci_high > -0.05


def test_estimate_all_domain_effects_ranks_domains_correctly():
    features, masks, reliability, predictions, categories = _make_synthetic_panel(
        true_effect_per_domain=(0.8, 0.0, -0.4), noise=0.04, seed=4
    )
    effects = estimate_all_domain_effects(
        features,
        masks,
        reliability,
        predictions,
        domain_order=["d0", "d1", "d2"],
        score_index=0,
        n_splits=5,
        random_state=4,
        category_codes=categories,
    )
    assert len(effects) == 3
    by_name = {e.domain: e for e in effects}
    assert by_name["d0"].ate > by_name["d1"].ate
    assert by_name["d2"].ate < by_name["d1"].ate


def test_estimate_handles_missing_domain_via_masks():
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(seed=5)
    masks = masks.copy()
    masks[::2, 1] = True  # half of samples have domain 1 missing
    effect = estimate_domain_causal_effect(
        features,
        masks,
        reliability,
        predictions,
        domain_index=1,
        domain_name="d1",
        score_index=0,
        n_splits=5,
        random_state=5,
    )
    assert effect.n_effective == int((~masks[:, 1]).sum())


def test_estimate_returns_nan_when_too_few_samples():
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(n_samples=8, seed=6)
    effect = estimate_domain_causal_effect(
        features,
        masks,
        reliability,
        predictions,
        domain_index=0,
        domain_name="d0",
        score_index=0,
        n_splits=5,
        random_state=6,
    )
    assert np.isnan(effect.ate) or np.isnan(effect.ate_std_error)


def test_per_sample_cate_returns_array_of_correct_length():
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(seed=7)
    cate = estimate_per_sample_cate(
        features,
        masks,
        reliability,
        predictions,
        domain_index=0,
        score_index=0,
        n_splits=5,
        random_state=7,
    )
    assert cate.shape == (features.shape[0],)
    assert np.isfinite(cate).any()


def test_per_sample_cate_handles_masked_samples():
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(seed=8)
    masks = masks.copy()
    masks[:30, 0] = True  # first 30 samples have domain 0 missing
    cate = estimate_per_sample_cate(
        features,
        masks,
        reliability,
        predictions,
        domain_index=0,
        score_index=0,
        n_splits=5,
        random_state=8,
    )
    assert np.all(np.isnan(cate[:30]))
    assert np.isfinite(cate[30:]).any()


def test_per_sample_cate_agrees_with_ate_on_sign_and_finiteness():
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(
        true_effect_per_domain=(0.6, 0.0, 0.0), noise=0.03, seed=9
    )
    ate = estimate_domain_causal_effect(
        features,
        masks,
        reliability,
        predictions,
        domain_index=0,
        domain_name="d0",
        score_index=0,
        n_splits=5,
        random_state=9,
    )
    cate = estimate_per_sample_cate(
        features,
        masks,
        reliability,
        predictions,
        domain_index=0,
        score_index=0,
        n_splits=5,
        random_state=9,
    )
    finite_cate = cate[np.isfinite(cate)]
    # CATE under Ridge regularisation will be shrunk toward zero, but the
    # sign of the mean CATE must match the ATE direction for a strong
    # positive true effect and the dispersion must be positive.
    assert ate.ate > 0.0
    assert float(finite_cate.mean()) > 0.0
    assert float(finite_cate.std()) > 0.0


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_recovered_effect_is_stable_across_seeds(seed: int):
    features, masks, reliability, predictions, _categories = _make_synthetic_panel(
        true_effect_per_domain=(0.7, 0.0, 0.0), noise=0.04, seed=seed
    )
    effect = estimate_domain_causal_effect(
        features,
        masks,
        reliability,
        predictions,
        domain_index=0,
        domain_name="d0",
        score_index=0,
        n_splits=5,
        random_state=seed,
    )
    assert 0.45 < effect.ate < 0.95
