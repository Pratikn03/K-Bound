"""kbound_edge.evidence -- Label-free evidence features for edge inspection.

FIXED SCHEMA.  The edge evidence vector is the paper's 11 label-free
disagreement features (reused verbatim from :func:`kbound.evidence.evidence_vector`)
plus 3 edge-specific label-free features, for 14 total.  The order and names are
frozen in :data:`EDGE_EVIDENCE_NAMES`; tests assert the schema is stable and
complete.  Every feature is a deterministic function of the pre-/post-adaptation
softmax matrices ``p0`` and ``pa`` (and the optional scalar update norm) -- there
is NO dependence on ground-truth labels, so this module is safe to call on the
online path.

Paper features (indices 0..10), see kbound.evidence.EVIDENCE_NAMES:
    pre_entropy, pre_conf, pre_pbal, post_entropy, post_conf, post_pbal,
    pbal_drop, entropy_drop, frac_highconf, marginal_KL, update_norm

Edge features (indices 11..13):
    mean_js_div        mean per-sample Jensen-Shannon divergence JS(p0_i || pa_i)
                       -- symmetric, bounded disagreement magnitude in [0, ln2].
    pred_flip_rate     fraction of samples whose argmax prediction changed
                       (argmax(p0_i) != argmax(pa_i)) -- prediction churn.
    post_top2_margin   mean (top1 - top2) probability gap of pa -- post-adapt
                       sharpness/separation; collapses toward ~1 on over-confidence.
"""

from __future__ import annotations

import numpy as np

from kbound_edge._bridge import (
    evidence_vector as _paper_evidence_vector,
    PAPER_EVIDENCE_NAMES,
)

#: The 3 edge-specific label-free feature names (appended after the paper's 11).
EDGE_EXTRA_NAMES: tuple[str, ...] = (
    "mean_js_div",
    "pred_flip_rate",
    "post_top2_margin",
)

#: Frozen full schema: 11 paper features + 3 edge features = 14.
EDGE_EVIDENCE_NAMES: tuple[str, ...] = tuple(PAPER_EVIDENCE_NAMES) + EDGE_EXTRA_NAMES

#: Number of features in the edge evidence vector.
N_EDGE_FEATURES: int = len(EDGE_EVIDENCE_NAMES)


def _normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def _mean_js_divergence(p0: np.ndarray, pa: np.ndarray) -> float:
    """Mean per-sample Jensen-Shannon divergence (natural log, in [0, ln 2])."""
    p0 = _normalize_rows(p0)
    pa = _normalize_rows(pa)
    m = 0.5 * (p0 + pa)
    m = np.clip(m, 1e-12, None)

    def _kl(a, b):
        a = np.clip(a, 1e-12, None)
        return (a * (np.log(a) - np.log(b))).sum(axis=1)

    js = 0.5 * _kl(p0, m) + 0.5 * _kl(pa, m)
    return float(np.mean(js))


def _pred_flip_rate(p0: np.ndarray, pa: np.ndarray) -> float:
    """Fraction of samples whose top-1 class changed after adaptation."""
    a0 = np.asarray(p0, dtype=float).argmax(axis=1)
    aa = np.asarray(pa, dtype=float).argmax(axis=1)
    return float(np.mean(a0 != aa))


def _post_top2_margin(pa: np.ndarray) -> float:
    """Mean top1-minus-top2 probability gap of the adapted predictions."""
    pa = _normalize_rows(pa)
    if pa.shape[1] < 2:
        return float(np.mean(pa.max(axis=1)))
    part = np.partition(pa, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    return float(np.mean(top1 - top2))


def edge_evidence_vector(
    p0: np.ndarray,
    pa: np.ndarray,
    upd_norm: float = 0.0,
) -> np.ndarray:
    """Compute the 14-D label-free edge evidence vector.

    Reuses the paper's :func:`kbound.evidence.evidence_vector` for the first 11
    components (so the certificate-facing features are identical to the paper),
    then appends the 3 edge features.

    Parameters
    ----------
    p0 : np.ndarray of shape (n_samples, n_classes)
        Frozen-model softmax probabilities.
    pa : np.ndarray of shape (n_samples, n_classes)
        Adapted-candidate softmax probabilities.
    upd_norm : float, default=0.0
        L2 norm of the BN-affine update (0.0 if unknown).

    Returns
    -------
    z : np.ndarray of shape (14,)
        Ordered as :data:`EDGE_EVIDENCE_NAMES`.  Never contains NaN/inf for
        valid probability inputs.
    """
    p0 = np.asarray(p0, dtype=float)
    pa = np.asarray(pa, dtype=float)
    z11 = _paper_evidence_vector(p0, pa, upd_norm)  # reused, validated upstream
    extra = np.array(
        [
            _mean_js_divergence(p0, pa),
            _pred_flip_rate(p0, pa),
            _post_top2_margin(pa),
        ],
        dtype=float,
    )
    z = np.concatenate([np.asarray(z11, dtype=float), extra])
    return z


def evidence_dict(p0: np.ndarray, pa: np.ndarray, upd_norm: float = 0.0) -> dict:
    """Same as :func:`edge_evidence_vector` but returned as a name->value dict."""
    z = edge_evidence_vector(p0, pa, upd_norm)
    return {name: float(val) for name, val in zip(EDGE_EVIDENCE_NAMES, z)}
