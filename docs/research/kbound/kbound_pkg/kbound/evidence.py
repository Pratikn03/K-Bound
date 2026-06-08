"""kbound.evidence -- Label-free disagreement evidence vector.

Theorem thm:disagree (K-Bound paper):
    The benefit of adaptation Delta = E[l(f0,y) - l(fa,y)] can be bounded
    using label-free statistics derived from the pre- and post-adaptation
    softmax distributions p0 and pa.  These statistics form the feature vector
    Z fed to the BenefitRouter.

The 11 components mirror exactly the EVIDENCE_NAMES list in
    docs/research/kbound/scripts/cifar_tent_mps_v2.py
but are computed from precomputed probability arrays (no torch required):

    [pre_entropy, pre_conf, pre_pbal,
     post_entropy, post_conf, post_pbal,
     pbal_drop, entropy_drop,
     frac_highconf, marginal_KL, update_norm]
"""

from __future__ import annotations

import math
import numpy as np


#: Names of the 11 components of the evidence vector (matches cifar_tent_mps_v2.py).
EVIDENCE_NAMES: list[str] = [
    "pre_entropy",
    "pre_conf",
    "pre_pbal",
    "post_entropy",
    "post_conf",
    "post_pbal",
    "pbal_drop",
    "entropy_drop",
    "frac_highconf",
    "marginal_KL",
    "update_norm",
]


def _entropy_mean(p: np.ndarray) -> float:
    """Mean per-sample entropy of a probability matrix."""
    # p: (n_samples, n_classes)
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log(p)).sum(axis=1).mean())



def evidence_vector(
    p0: np.ndarray,
    pa: np.ndarray,
    upd_norm: float = 0.0,
) -> np.ndarray:
    """Label-free evidence vector from pre/post-adaptation softmax distributions.

    Implements Thm thm:disagree: computes 11 label-free statistics that serve
    as features Z for the BenefitRouter.  The statistics quantify how much the
    model's predictive distribution has shifted after adaptation.

    This is the pure-numpy (no-torch) counterpart of ``evidence_vector`` in
    ``cifar_tent_mps_v2.py``, which requires live model forward passes.

    Parameters
    ----------
    p0 : np.ndarray of shape (n_samples, n_classes)
        Softmax probabilities from the *frozen* model.
    pa : np.ndarray of shape (n_samples, n_classes)
        Softmax probabilities from the *adapted* model.
    upd_norm : float, default=0.0
        L2 norm of the parameter update (e.g. norm of BN-affine delta).
        Pass 0.0 if unknown / not applicable.

    Returns
    -------
    z : np.ndarray of shape (11,)
        Evidence vector with components named by :data:`EVIDENCE_NAMES`.

    Notes
    -----
    Collapse detection:
      - ``frac_highconf`` near 1 indicates entropy collapse (model always
        predicts one class at high confidence after adaptation).
      - ``marginal_KL`` large means the adapted marginal prediction has shifted
        far from the frozen marginal.
      - ``pbal_drop`` large means the adapted batch predictions are more
        class-imbalanced than the frozen ones.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> p0 = rng.dirichlet(np.ones(10), size=32)
    >>> pa = rng.dirichlet(np.ones(10), size=32)
    >>> z = evidence_vector(p0, pa, upd_norm=0.1)
    >>> z.shape
    (11,)
    """
    p0 = np.asarray(p0, dtype=float)
    pa = np.asarray(pa, dtype=float)
    if p0.ndim != 2 or pa.ndim != 2:
        raise ValueError("p0 and pa must be 2-D arrays (n_samples, n_classes)")
    if p0.shape != pa.shape:
        raise ValueError(f"p0 and pa must have the same shape; got {p0.shape} vs {pa.shape}")
    if p0.shape[0] == 0:
        raise ValueError("p0/pa must have at least one sample")

    p0 = np.clip(p0, 1e-12, None)
    pa = np.clip(pa, 1e-12, None)

    # Normalize rows to sum to 1 (guard against near-zero sums)
    p0 = p0 / p0.sum(axis=1, keepdims=True)
    pa = pa / pa.sum(axis=1, keepdims=True)

    n_classes = p0.shape[1]
    log_nc = math.log(n_classes) if n_classes > 1 else 1.0

    # Pre-adaptation stats
    e0 = _entropy_mean(p0)                          # mean per-sample entropy
    conf0 = float(p0.max(axis=1).mean())            # mean max-prob
    mb0 = p0.mean(axis=0)
    mb0 = np.clip(mb0, 1e-12, None)
    pbal0 = float(-(mb0 * np.log(mb0)).sum()) / log_nc   # normalised marginal entropy

    # Post-adaptation stats
    ea = _entropy_mean(pa)
    confa = float(pa.max(axis=1).mean())
    mba = pa.mean(axis=0)
    mba = np.clip(mba, 1e-12, None)
    pbala = float(-(mba * np.log(mba)).sum()) / log_nc

    # Derived statistics
    pbal_drop = pbal0 - pbala          # positive -> adapted is more imbalanced
    entropy_drop = e0 - ea             # positive -> adapted has lower entropy

    frac_highconf = float((pa.max(axis=1) > 0.9).mean())  # collapse indicator

    # Marginal-prediction KL(adapted || frozen)  -- spikes on collapse
    klm = float((mba * (np.log(mba) - np.log(mb0))).sum())

    z = np.array([
        e0, conf0, pbal0,
        ea, confa, pbala,
        pbal_drop, entropy_drop,
        frac_highconf, klm,
        float(upd_norm),
    ], dtype=float)
    return z
