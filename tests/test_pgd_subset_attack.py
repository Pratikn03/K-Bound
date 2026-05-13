"""Tests for gradient-aligned PGD attack over a subset of domains."""

from __future__ import annotations

import numpy as np
import torch

from uais.fusion.attention.adversarial_robustness import AdversarialPerturbationEngine
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel


def _make_model(num_domains: int = 3, input_dim: int = 2):
    torch.manual_seed(0)
    model = AttentionFusionModel(
        num_domains=num_domains,
        input_dim=input_dim,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_confidence=False,
        use_input_confidence=False,
    )
    model.eval()
    return model


def test_pgd_preserves_shapes():
    rng = np.random.default_rng(0)
    n, d, f = 16, 3, 2
    features = rng.uniform(0.0, 1.0, size=(n, d, f)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    domain_order = ["a", "b", "c"]
    model = _make_model(num_domains=d, input_dim=f)
    engine = AdversarialPerturbationEngine(domain_order, score_index=0, random_seed=0)

    perturbed, returned_masks = engine.pgd_attack_subset(
        model, features, masks, labels,
        target_domains=["b"], epsilon=0.1, step_size=0.02, n_steps=3,
    )
    assert perturbed.shape == features.shape
    assert returned_masks is masks


def test_pgd_only_perturbs_target_domain_score():
    rng = np.random.default_rng(1)
    n, d, f = 8, 3, 2
    features = rng.uniform(0.3, 0.7, size=(n, d, f)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = np.ones(n, dtype=np.float32)
    domain_order = ["a", "b", "c"]
    model = _make_model(num_domains=d, input_dim=f)
    engine = AdversarialPerturbationEngine(domain_order, score_index=0, random_seed=0)

    perturbed, _ = engine.pgd_attack_subset(
        model, features, masks, labels,
        target_domains=["b"], epsilon=0.2, step_size=0.05, n_steps=5,
    )
    # Domain b score channel must differ
    assert not np.allclose(perturbed[:, 1, 0], features[:, 1, 0])
    # Domains a and c score channels must be unchanged
    np.testing.assert_allclose(perturbed[:, 0, 0], features[:, 0, 0])
    np.testing.assert_allclose(perturbed[:, 2, 0], features[:, 2, 0])
    # Non-score channels must be unchanged across all domains
    np.testing.assert_allclose(perturbed[:, :, 1], features[:, :, 1])


def test_pgd_respects_epsilon_budget():
    rng = np.random.default_rng(2)
    n, d, f = 20, 2, 2
    features = rng.uniform(0.3, 0.7, size=(n, d, f)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = (features[:, 0, 0] > 0.5).astype(np.float32)
    domain_order = ["a", "b"]
    model = _make_model(num_domains=d, input_dim=f)
    engine = AdversarialPerturbationEngine(domain_order, score_index=0, random_seed=0)

    epsilon = 0.15
    perturbed, _ = engine.pgd_attack_subset(
        model, features, masks, labels,
        target_domains=["a"], epsilon=epsilon, step_size=0.03, n_steps=10,
    )
    # Linf budget on score channel of domain a
    delta = perturbed[:, 0, 0] - features[:, 0, 0]
    # Allow tiny float slack from clipping to [0, 1]
    assert np.max(np.abs(delta)) <= epsilon + 1e-5


def test_pgd_respects_missing_mask():
    rng = np.random.default_rng(3)
    n, d, f = 10, 2, 2
    features = rng.uniform(0.0, 1.0, size=(n, d, f)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    masks[:5, 0] = True  # First half: domain 0 missing
    labels = np.ones(n, dtype=np.float32)
    domain_order = ["a", "b"]
    model = _make_model(num_domains=d, input_dim=f)
    engine = AdversarialPerturbationEngine(domain_order, score_index=0, random_seed=0)

    perturbed, _ = engine.pgd_attack_subset(
        model, features, masks, labels,
        target_domains=["a"], epsilon=0.2, step_size=0.05, n_steps=3,
    )
    # Masked-out entries should not be perturbed
    np.testing.assert_allclose(perturbed[:5, 0, 0], features[:5, 0, 0])
    # Available entries may be perturbed (no exact-equality guarantee but they
    # should be in the epsilon-ball)
    delta_avail = perturbed[5:, 0, 0] - features[5:, 0, 0]
    assert np.max(np.abs(delta_avail)) <= 0.2 + 1e-5


def test_pgd_empty_target_returns_unchanged():
    rng = np.random.default_rng(4)
    n, d, f = 4, 2, 2
    features = rng.uniform(0.0, 1.0, size=(n, d, f)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = np.ones(n, dtype=np.float32)
    domain_order = ["a", "b"]
    model = _make_model(num_domains=d, input_dim=f)
    engine = AdversarialPerturbationEngine(domain_order, score_index=0, random_seed=0)
    perturbed, _ = engine.pgd_attack_subset(
        model, features, masks, labels,
        target_domains=[], epsilon=0.1, step_size=0.02, n_steps=3,
    )
    np.testing.assert_allclose(perturbed, features)
