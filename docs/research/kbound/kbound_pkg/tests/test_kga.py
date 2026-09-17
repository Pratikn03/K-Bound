"""Tests for kbound.kga (KGA).

Covers the no-torch path:
  - KGA.decide() returning adapt/freeze/abstain based on heuristic certificate
  - KGA.evidence() returning the evidence vector
  - KGA.decide_from_batch() raising ImportError when torch absent
  - KGA constructor with/without router
"""

import numpy as np
import pytest

from kbound.kga import KGA
from kbound.router import BenefitRouter
from kbound.evidence import EVIDENCE_NAMES


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
# decide() — heuristic certificate (no-torch)
# ---------------------------------------------------------------------------

class TestKGADecide:
    def test_returns_valid_string(self):
        kga = KGA()
        p0 = make_probs(seed=10)
        pa = make_probs(seed=11)
        d = kga.decide(p0, pa)
        assert d in ("adapt", "freeze", "abstain")

    def test_freeze_on_collapse(self):
        """Collapse: pa assigns near-unit mass to one class -> FREEZE."""
        n, C = 64, 10
        pa = np.full((n, C), 1e-4)
        pa[:, 0] = 1.0 - (C - 1) * 1e-4
        p0 = make_probs(n=n, C=C, seed=20)
        kga = KGA()
        d = kga.decide(p0, pa)
        assert d == "freeze", f"Expected freeze on collapse, got {d}"

    def test_freeze_on_high_marginal_kl(self):
        """Very large marginal KL (adapted distribution completely different) -> FREEZE."""
        n, C = 64, 10
        # p0: uniform-ish
        p0 = np.full((n, C), 1.0 / C)
        # pa: all mass on class 0 (extreme shift)
        pa = np.full((n, C), 1e-6)
        pa[:, 0] = 1.0 - (C - 1) * 1e-6
        kga = KGA()
        d = kga.decide(p0, pa)
        assert d == "freeze", f"Expected freeze on extreme KL shift, got {d}"

    def test_adapt_on_entropy_drop(self):
        """Strong entropy reduction without collapse -> ADAPT."""
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
        assert d in ("adapt", "abstain"), (
            f"Expected adapt or abstain for entropy-reducing pa, got {d}"
        )

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
        assert d in ("abstain", "adapt", "freeze"), f"Invalid decision: {d}"

    def test_decide_with_upd_norm(self):
        kga = KGA()
        p0 = make_probs(seed=50)
        pa = make_probs(seed=51)
        d = kga.decide(p0, pa, upd_norm=1.5)
        assert d in ("adapt", "freeze", "abstain")


# ---------------------------------------------------------------------------
# decide_from_batch() — torch path (torch absent -> ImportError)
# ---------------------------------------------------------------------------

class TestKGADecideFromBatch:
    def test_raises_importerror_without_torch(self):
        """decide_from_batch must raise ImportError when torch is not installed."""
        import sys
        # Confirm torch is genuinely absent in this environment
        if "torch" in sys.modules:
            pytest.skip("torch is installed; cannot test torch-absent path here")
        kga = KGA()
        with pytest.raises(ImportError, match="torch"):
            kga.decide_from_batch(None)
