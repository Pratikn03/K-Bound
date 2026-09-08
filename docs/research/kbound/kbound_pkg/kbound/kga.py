"""Legacy reproduction facade with evidence extraction and unavailable decisions.

This interface has no calibrated, externally authorized benefit-estimator input.
Consequently its decision methods always ABSTAIN. Entropy, confidence, and
collapse diagnostics alone do not justify a strict benefit-direction claim.

Use :meth:`KGA.evidence` for descriptive features. The stored ``router`` remains
available for explicitly retrospective analyses; fitting it does not authorize
this facade to make a deployment decision. The maintained ``kga`` package has a
separate schema-bound estimator interface with explicit coverage assumptions.
"""

from __future__ import annotations

import numpy as np

from kbound.evidence import evidence_vector
from kbound.router import BenefitRouter


class KGA:
    """Compatibility facade that retains the frozen model without certification.

    ``f0``, ``fa``, ``alpha``, and ``router`` are retained for existing callers.
    Neither a router object nor unlabelled probability arrays establish an
    authorized predictor/calibration identity. No strict direction is inferred.
    """

    def __init__(
        self,
        f0=None,
        fa=None,
        alpha: float = 0.1,
        router: BenefitRouter | None = None,
    ) -> None:
        self.f0 = f0
        self.fa = fa
        self.alpha = alpha
        self.router = router if router is not None else BenefitRouter()

    def decide(
        self,
        p0: np.ndarray,
        pa: np.ndarray,
        upd_norm: float = 0.0,
    ) -> str:
        """Return ABSTAIN: this legacy API has no authorized benefit interval.

        Inputs and model objects are not mutated. This is unavailable evidence
        for a direction, not a certified FREEZE or a fitted-heuristic fallback.
        Call :meth:`evidence` separately when descriptive features are wanted.
        """
        return "abstain"

    def evidence(
        self,
        p0: np.ndarray,
        pa: np.ndarray,
        upd_norm: float = 0.0,
    ) -> np.ndarray:
        """Compute the legacy 11-feature descriptive vector, not a certificate."""
        return evidence_vector(p0, pa, upd_norm)

    def decide_from_batch(self, x, upd_norm: float = 0.0) -> str:
        """ABSTAIN without inference or model-mode changes when authority is absent.

        This unavailable path does not require torch or evaluate the candidate.
        The caller retains ``f0``; the function makes no negative-benefit claim.
        """
        return "abstain"
