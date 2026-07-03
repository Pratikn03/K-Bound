"""Tests for target-label-light micro-probe certificate."""

from __future__ import annotations

import numpy as np

from kga import KGA, Decision


# placements / placement_benefits vendored inline from the former ELARA
# src/uais/kbound/multimodal_guard during the ELARA separation, so this test
# keeps KGA probe-certificate coverage without importing the ELARA tree.
def placements(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.array([])
    ns = np.sort(neg)
    below = np.searchsorted(ns, pos, side="left")
    ties = np.searchsorted(ns, pos, side="right") - below
    return (below + 0.5 * ties) / len(neg)


def placement_benefits(y: np.ndarray, s_frozen: np.ndarray, s_adapt: np.ndarray) -> np.ndarray:
    return placements(y, s_adapt) - placements(y, s_frozen)


def test_certify_probe_subsample():
    rng = np.random.default_rng(0)
    benefits = rng.normal(0.35, 0.04, size=200)
    kga = KGA(alpha=0.1)
    cert_full = kga.certify_probe(benefits, k=None, benefit_range=2.0)
    cert_k8 = kga.certify_probe(benefits, k=8, seed=0, benefit_range=2.0)
    cert_k64 = kga.certify_probe(benefits, k=64, seed=0, benefit_range=2.0)
    assert cert_full.n == 200
    assert cert_k8.n == 8
    assert cert_k64.n == 64
    assert cert_k8.epsilon > cert_k64.epsilon
    assert kga.decide(cert_full) == Decision.ADAPT


def test_probe_adapts_on_tight_positive_probe():
    benefits = np.full(24, 0.25) + np.linspace(-0.01, 0.01, 24)
    kga = KGA(alpha=0.1)
    cert = kga.certify_probe(benefits, k=24, benefit_range=0.05)
    assert kga.decide(cert) == Decision.ADAPT


def test_probe_radius_shrinks_with_k_on_positive_benefits():
    rng = np.random.default_rng(1)
    benefits = rng.normal(0.2, 0.08, size=500)
    kga = KGA(alpha=0.1)
    eps = {}
    for k in (8, 16, 32, 64, 500):
        cert = kga.certify_probe(benefits, k=k, seed=0, benefit_range=2.0)
        eps[k] = cert.epsilon
    assert eps[8] > eps[64] > eps[500]


def test_label_free_abstains_probe_commits_bias_limited_scenario():
    """Wide label-free pool brackets zero; tight probe commits adapt."""
    rng = np.random.default_rng(42)
    pool = rng.normal(0.02, 0.22, size=400)
    kga = KGA(alpha=0.1)
    cert_lf = kga.certify(scores=pool, benefit_range=2.0)
    assert kga.decide(cert_lf) == Decision.ABSTAIN
    probe = np.full(32, 0.18) + rng.normal(0, 0.005, size=32)
    cert_probe = kga.certify_probe(probe, k=32, benefit_range=0.1)
    assert kga.decide(cert_probe) == Decision.ADAPT


def test_placement_benefits_positive_when_fusion_improves():
    y = np.array([0, 0, 0, 1, 1])
    s0 = np.array([0.1, 0.2, 0.3, 0.25, 0.28])
    s1 = np.array([0.1, 0.2, 0.3, 0.55, 0.7])
    pb = placement_benefits(y, s0, s1)
    assert pb.size == 2
    assert float(np.mean(pb)) > 0.0


# NOTE: the two MultimodalGuard tests were removed during the ELARA separation —
# MultimodalGuard lived in the ELARA src/uais tree. The certify_probe coverage
# above (the K-Bound feature under test) is unchanged.
