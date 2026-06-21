"""reliability.py — label-free reliability features for the ELARA-Opt gate.

Every feature is computed from the frozen model f0, the current model f_t and the
*unlabeled* batch x (plus optimizer-side scalars: update norm, inter-objective
gradient conflict).  No target labels are read.  These features (a) drive the gate
g_phi that maps to objective weights and (b) set the reliability-dependent
trust-region radius.  The frozen-vs-current divergence / BN-drift / class-balance
blocks reuse the validated K-Bound evidence helpers.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

from ._compat import bn_running_stats, bn_batch_stats, bn_stat_kl_drift

#: canonical, FROZEN order of reliability features fed to the gate.
RELIABILITY_NAMES = [
    "ent_mean",            # mean predictive entropy / ln C   (high -> unreliable)
    "ent_var",             # spread of entropy / ln C
    "conf_mean",           # mean top-1 prob                  (high -> reliable)
    "conf_var",            # spread of top-1 prob
    "pred_balance",        # marginal entropy / ln C          (low -> class collapse)
    "pred_balance_drift",  # KL(mean p_t || mean p_0)         (high -> distribution shift)
    "aug_disagreement",    # frac argmax changes under aug    (high -> unstable)
    "frozen_div",          # mean KL(p_0 || p_t)              (high -> drifted from f0)
    "bn_drift",            # BN running(f0) vs batch(f_t) KL  (high -> covariate shift)
    "grad_conflict",       # min pairwise cos of objective grads (neg -> conflicting)
    "update_norm",         # cumulative L2 of affine-param delta
]
FEATURE_DIM = len(RELIABILITY_NAMES)


@torch.no_grad()
def _marginal(p: torch.Tensor) -> torch.Tensor:
    return p.mean(0)


@torch.no_grad()
def compute_features(
    f0: torch.nn.Module,
    ft: torch.nn.Module,
    x: torch.Tensor,
    num_classes: int,
    *,
    update_norm: float = 0.0,
    grad_conflict: float = 0.0,
    aug=None,
) -> Dict[str, float]:
    """Return the named, label-free reliability feature dict for batch x."""
    f0.eval()
    ft.eval()
    lnC = math.log(max(num_classes, 2))

    p0 = f0(x).softmax(1)
    pt = ft(x).softmax(1)

    ent = -(pt * (pt + 1e-9).log()).sum(1)
    ent_mean = float(ent.mean()) / lnC
    ent_var = float(ent.var(unbiased=False)) / (lnC ** 2)
    conf = pt.max(1).values
    conf_mean = float(conf.mean())
    conf_var = float(conf.var(unbiased=False))

    mt = _marginal(pt)
    m0 = _marginal(p0)
    pred_balance = float(-(mt * (mt + 1e-9).log()).sum()) / lnC
    pred_balance_drift = float((mt * ((mt + 1e-9).log() - (m0 + 1e-9).log())).sum())

    if aug is None:
        aug = torch.flip(x, dims=[3]) if x.dim() == 4 else x
    pt_aug = ft(aug).softmax(1)
    aug_disagreement = float((pt.argmax(1) != pt_aug.argmax(1)).float().mean())

    frozen_div = float((p0 * ((p0 + 1e-9).log() - (pt + 1e-9).log())).sum(1).mean())

    rm, rv = bn_running_stats(f0)
    bm, bv = bn_batch_stats(ft, x)
    bn = bn_stat_kl_drift(rm, rv, bm, bv) if rm is not None else 0.0
    bn_drift = float(bn / (1.0 + bn))  # squash to [0,1)

    feats = {
        "ent_mean": ent_mean,
        "ent_var": ent_var,
        "conf_mean": conf_mean,
        "conf_var": conf_var,
        "pred_balance": pred_balance,
        "pred_balance_drift": float(abs(pred_balance_drift)),
        "aug_disagreement": aug_disagreement,
        "frozen_div": frozen_div,
        "bn_drift": bn_drift,
        "grad_conflict": float(grad_conflict),
        "update_norm": float(update_norm),
    }
    return feats


def to_vector(feats: Dict[str, float]) -> np.ndarray:
    """Pack the feature dict into the canonical fixed-order vector."""
    return np.array([float(feats[k]) for k in RELIABILITY_NAMES], dtype=np.float64)


def reliability_score(feats: Dict[str, float], coeffs: Dict[str, float]) -> float:
    """Monotone reliability scalar in (0,1): high when confident, low-entropy,
    low-drift, low-BN-shift, stable under augmentation. Constants come from the
    lock (deterministic). Used to set the trust-region radius."""
    z = (
        coeffs.get("bias", 0.0)
        + coeffs.get("conf_mean", 0.0) * feats["conf_mean"]
        - coeffs.get("ent_mean", 0.0) * feats["ent_mean"]
        - coeffs.get("frozen_div", 0.0) * feats["frozen_div"]
        - coeffs.get("bn_drift", 0.0) * feats["bn_drift"]
        - coeffs.get("aug_disagreement", 0.0) * feats["aug_disagreement"]
    )
    return float(1.0 / (1.0 + math.exp(-z)))
