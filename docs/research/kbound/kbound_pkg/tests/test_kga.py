"""Tests for kbound.kga (KGA).

Covers the no-torch path:
  - KGA.decide() abstaining without an authorized benefit interval
  - KGA.evidence() returning the evidence vector
  - KGA.decide_from_batch() abstaining without requiring torch or model access
  - KGA constructor with/without router
"""

import numpy as np
import pytest
from kbound.evidence import EVIDENCE_NAMES
from kbound.kga import KGA
from kbound.router import BenefitRouter


def make_probs(n=64, C=10, seed=0, concentration=1.0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(C) * concentration, size=n)
    return p


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestKGAConstructor:
    def test_default_constructor(self):
        kga = KGA()
        assert kga.alpha == 0.1
        assert kga.f0 is None
        assert kga.fa is None
        assert isinstance(kga.router, BenefitRouter)

    def test_custom_alpha(self):
        kga = KGA(alpha=0.05)
        assert kga.alpha == 0.05

    def test_custom_router(self):
        r = BenefitRouter(n_estimators=10)
        kga = KGA(router=r)
        assert kga.router is r

    def test_f0_fa_stored(self):
        sentinel = object()
        kga = KGA(f0=sentinel, fa=sentinel)
        assert kga.f0 is sentinel
        assert kga.fa is sentinel


# ---------------------------------------------------------------------------
# evidence() method
# ---------------------------------------------------------------------------


class TestKGAEvidence:
    def test_evidence_shape(self):
        kga = KGA()
        p0 = make_probs(seed=0)
        pa = make_probs(seed=1)
        z = kga.evidence(p0, pa)
        assert z.shape == (11,)
        assert len(EVIDENCE_NAMES) == 11

    def test_evidence_with_upd_norm(self):
        kga = KGA()
        p0 = make_probs(seed=2)
        pa = make_probs(seed=3)
        z = kga.evidence(p0, pa, upd_norm=0.77)
        assert z[-1] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# decide() — unavailable benefit authority (no-torch)
# ---------------------------------------------------------------------------


class TestKGADecide:
    def test_returns_valid_string(self):
        kga = KGA()
        p0 = make_probs(seed=10)
        pa = make_probs(seed=11)
        d = kga.decide(p0, pa)
        assert d == "abstain"

    def test_collapse_does_not_certify_negative_benefit(self):
        """Collapse alone supports no certified direction."""
        n, C = 64, 10
        pa = np.full((n, C), 1e-4)
        pa[:, 0] = 1.0 - (C - 1) * 1e-4
        p0 = make_probs(n=n, C=C, seed=20)
        kga = KGA()
        d = kga.decide(p0, pa)
        assert d == "abstain", f"Uncalibrated collapse must abstain, got {d}"

    def test_high_marginal_kl_does_not_certify_negative_benefit(self):
        """Large marginal KL alone supports no certified direction."""
        n, C = 64, 10
        # p0: uniform-ish
        p0 = np.full((n, C), 1.0 / C)
        # pa: all mass on class 0 (extreme shift)
        pa = np.full((n, C), 1e-6)
        pa[:, 0] = 1.0 - (C - 1) * 1e-6
        kga = KGA()
        d = kga.decide(p0, pa)
        assert d == "abstain", f"Uncalibrated KL shift must abstain, got {d}"

    def test_entropy_drop_does_not_certify_positive_benefit(self):
        """Entropy reduction without a calibrated estimator still abstains."""
        rng = np.random.default_rng(30)
        n, C = 64, 10
        # p0: diffuse (high entropy)
        p0 = rng.dirichlet(np.ones(C) * 0.3, size=n)
        # pa: peaked but not collapsing (moderate confidence 0.6-0.8 per sample)
        pa_raw = rng.dirichlet(np.ones(C) * 0.05, size=n)
        # Scale so max is around 0.75 — well below 0.9 threshold
        pa_raw = pa_raw / pa_raw.max(axis=1, keepdims=True) * 0.75
        # Re-normalise each row
        pa_raw = np.clip(pa_raw, 1e-6, None)
        pa = pa_raw / pa_raw.sum(axis=1, keepdims=True)
        kga = KGA()
        d = kga.decide(p0, pa)
        assert d == "abstain"

    def test_abstain_when_uncertain(self):
        """Near-identical p0 and pa with small changes -> ABSTAIN."""
        rng = np.random.default_rng(40)
        n, C = 64, 10
        base = rng.dirichlet(np.ones(C), size=n)
        # Small perturbation
        noise = rng.dirichlet(np.ones(C) * 10, size=n)
        pa = 0.98 * base + 0.02 * noise
        pa /= pa.sum(axis=1, keepdims=True)
        kga = KGA()
        d = kga.decide(base, pa)
        assert d == "abstain"

    def test_decide_with_upd_norm(self):
        kga = KGA()
        p0 = make_probs(seed=50)
        pa = make_probs(seed=51)
        d = kga.decide(p0, pa, upd_norm=1.5)
        assert d == "abstain"


# ---------------------------------------------------------------------------
# decide_from_batch() — authority is unavailable, so no inference is attempted
# ---------------------------------------------------------------------------


class TestKGADecideFromBatch:
    def test_without_models_remains_unavailable(self):
        """Missing models/authority do not justify inference or strict actions."""
        kga = KGA()
        assert kga.decide_from_batch(None) == "abstain"
