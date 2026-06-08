"""kbound.kga -- KGA: full adapt/freeze/abstain gate.

KGA (K-Bound Gated Adaptation) is the main decision module that combines
the label-free evidence vector with the BenefitRouter and conformal certificate
to produce a per-condition adapt/freeze/abstain decision.

Torch dependency:
    The original KGA (cifar_tent_mps_v2.py) computes evidence Z from live
    model forward passes (requires torch).  This class accepts PRECOMPUTED
    probability arrays p0 and pa so that the core logic is fully testable
    without torch.  Torch is only needed when calling from a training loop
    with actual model objects.

Usage (no-torch path -- always works):
    kga = KGA(alpha=0.10)
    # Pass probability arrays from your own inference code:
    decision = kga.decide(p0, pa, upd_norm=0.0)

Usage (torch path -- requires a router fitted on training conditions):
    kga = KGA(f0=frozen_model, fa=adapted_model, alpha=0.10)
    kga.router.fit(Z_train, B_train)
    decision = kga.decide_from_batch(x_batch, upd_norm)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from kbound.certificate import decide as cert_decide
from kbound.evidence import evidence_vector
from kbound.router import BenefitRouter


class KGA:
    """K-Bound Gated Adaptation: combines evidence + BenefitRouter + certificate.

    For the no-torch path, ``decide(p0, pa)`` accepts precomputed probability
    arrays and returns a single adapt/freeze/abstain string without a fitted
    router (uses a simple conformal rule based on the evidence alone).

    For a full batch decision with a trained router, call ``router.decide_all``.

    Parameters
    ----------
    f0 : optional torch.nn.Module
        Frozen base model.  Only needed for the torch forward-pass path.
    fa : optional torch.nn.Module
        Adapted model.  Only needed for the torch forward-pass path.
    alpha : float, default=0.1
        Miscoverage level for the conformal certificate.
    router : BenefitRouter, optional
        Pre-configured router.  Defaults to ``BenefitRouter()`` with default
        hyperparameters.

    Attributes
    ----------
    router : BenefitRouter
        The underlying benefit router.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p0 = rng.dirichlet(np.ones(10), size=32)
    >>> pa = rng.dirichlet(np.ones(10), size=32)
    >>> kga = KGA(alpha=0.1)
    >>> d = kga.decide(p0, pa, upd_norm=0.05)
    >>> d in ('adapt', 'freeze', 'abstain')
    True
    """

    def __init__(
        self,
        f0=None,
        fa=None,
        alpha: float = 0.1,
        router: Optional[BenefitRouter] = None,
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
        """Per-decision gate from precomputed probability arrays (no-torch path).

        Computes the evidence vector Z from p0 and pa, then applies a simple
        heuristic certificate based on entropy drop and marginal KL.

        For a full router-based decision on a COLLECTION of conditions, use
        ``router.decide_all(Z_matrix, B_vector)``.

        Parameters
        ----------
        p0 : np.ndarray of shape (n_samples, n_classes)
            Frozen model softmax probabilities.
        pa : np.ndarray of shape (n_samples, n_classes)
            Adapted model softmax probabilities.
        upd_norm : float, default=0.0
            Parameter update norm.

        Returns
        -------
        str : ``'adapt'``, ``'freeze'``, or ``'abstain'``.

        Notes
        -----
        The no-torch heuristic:
            - Entropy drop > 0.15 * log(n_classes)  AND  frac_highconf < 0.5
              AND marginal_KL < 1.0 -> ADAPT (strong adaptation signal, no collapse)
            - marginal_KL > 1.5  OR  frac_highconf > 0.8  -> FREEZE (collapse detected)
            - Otherwise ABSTAIN.

        This heuristic is a stand-in for the full router; for the paper's
        certified guarantee use ``router.decide_all`` with labelled conditions.
        """
        z = evidence_vector(p0, pa, upd_norm)
        return self._heuristic_decide(z, p0.shape[1])

    def evidence(
        self,
        p0: np.ndarray,
        pa: np.ndarray,
        upd_norm: float = 0.0,
    ) -> np.ndarray:
        """Compute and return the label-free evidence vector Z.

        Parameters
        ----------
        p0 : np.ndarray of shape (n_samples, n_classes)
        pa : np.ndarray of shape (n_samples, n_classes)
        upd_norm : float, default=0.0

        Returns
        -------
        z : np.ndarray of shape (11,)
        """
        return evidence_vector(p0, pa, upd_norm)

    def decide_from_batch(self, x, upd_norm: float = 0.0) -> str:
        """Compute evidence from live model forward passes and gate (torch path).

        Requires ``self.f0`` and ``self.fa`` to be torch.nn.Module instances.
        Raises RuntimeError if torch is not available or models are not set.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input batch on the appropriate device.
        upd_norm : float
            Parameter update L2 norm.

        Returns
        -------
        str : ``'adapt'``, ``'freeze'``, or ``'abstain'``.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torch is required for decide_from_batch. "
                "Install it with: pip install kbound[torch]"
            ) from exc

        if self.f0 is None or self.fa is None:
            raise RuntimeError(
                "f0 and fa must be set to use decide_from_batch. "
                "Use decide(p0, pa) for the no-torch path."
            )

        import torch

        self.f0.eval()
        self.fa.eval()
        with torch.no_grad():
            p0 = self.f0(x).softmax(dim=1).cpu().numpy()
            pa = self.fa(x).softmax(dim=1).cpu().numpy()

        return self.decide(p0, pa, upd_norm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_decide(self, z: np.ndarray, n_classes: int) -> str:
        """Simple heuristic certificate for single-condition no-router use."""
        import math

        # z indices: pre_entropy=0, pre_conf=1, pre_pbal=2,
        #            post_entropy=3, post_conf=4, post_pbal=5,
        #            pbal_drop=6, entropy_drop=7,
        #            frac_highconf=8, marginal_KL=9, update_norm=10

        entropy_drop = z[7]          # positive = entropy decreased after adapt
        frac_highconf = z[8]         # near 1 = collapse
        marginal_kl = z[9]           # large = prediction distribution shifted a lot

        log_nc = math.log(max(n_classes, 2))
        entropy_drop_normed = entropy_drop / log_nc if log_nc > 0 else 0.0

        # Collapse: high fraction of overconfident predictions or large KL
        if frac_highconf > 0.8 or marginal_kl > 1.5:
            return "freeze"

        # Adaptation benefit: entropy reduced without collapse
        if entropy_drop_normed > 0.15 and frac_highconf < 0.5 and marginal_kl < 1.0:
            return "adapt"

        return "abstain"
