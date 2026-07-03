"""Gap B — scale-invariant, source-calibrated CEI gate (self-normalized tau).

Frozen per PROTOCOL_GAPCLOSE_WAVE5_v1.md.

Problem: the rank-one residual tau_hat has a null scale that depends on panel
size K, sample size m, and |b| magnitudes, so a single fixed threshold
(tau* = 0.52, fit on synthetic panels) does not transfer to natural panels
(Camelyon tau_bar = 0.89, ImageNet-R tau_bar = 2.60).

Fix: self-normalization by a label-free parametric-bootstrap null under fitted H.
All inputs are label-free observables (agreement matrix C, panel size m, class
balance proxy pi). tau' = tau_obs / Q_{1-alpha}(tau_null); reject H iff tau' > 1.
"""
from __future__ import annotations

import numpy as np


def tau_residual(C: np.ndarray, b: np.ndarray) -> float:
    """Normalized off-diagonal Frobenius residual of the rank-one fit."""
    K = C.shape[0]
    mask = ~np.eye(K, dtype=bool)
    num = np.linalg.norm((C - np.outer(b, b))[mask])
    den = np.linalg.norm(C[mask]) + 1e-12
    return float(num / den)


def fit_rank_one(C: np.ndarray, return_se: bool = False):
    """|b_i| via median triple products, signs anchored to candidate 0.

    b_i^2 = c_ik c_il / c_kl (all distinct k, l != i). Global flip is
    irrelevant (tau is flip-invariant), so anchor b_0 >= 0 and set
    sign(b_i) = sign(c_{0i}) relative to it. With return_se, also returns the
    empirical SE of b_i from the dispersion of the per-pair estimates
    (delta method through the sqrt), used to widen the bootstrap null by the
    fit's own uncertainty (smoothed parametric bootstrap).
    """
    K = C.shape[0]
    b2 = np.zeros(K)
    se_b = np.zeros(K)
    for i in range(K):
        vals = []
        for k in range(K):
            for l in range(k + 1, K):
                if i in (k, l) or abs(C[k, l]) < 1e-9:
                    continue
                vals.append(C[i, k] * C[i, l] / C[k, l])
        vals = np.asarray(vals, dtype=float)
        if vals.size:
            b2[i] = np.median(vals)
            mad = 1.4826 * np.median(np.abs(vals - b2[i]))
            se_b2 = 1.2533 * mad / np.sqrt(vals.size)  # se of a median
            root = np.sqrt(max(b2[i], 1e-4))
            se_b[i] = se_b2 / (2.0 * root)              # delta method
    b = np.sqrt(np.clip(b2, 0.0, 1.0))
    sign = np.ones(K)
    for i in range(1, K):
        sign[i] = np.sign(C[0, i]) if C[0, i] != 0 else 1.0
    if return_se:
        return b * sign, np.clip(se_b, 0.0, 0.5)
    return b * sign


def simulate_H_panel(b: np.ndarray, pi: float, m: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Empirical agreement matrix from m draws of an exact-H panel.

    Under H: correctness s_i in {0,1} with P(s_i=1|y) = a_i = (1+|b_i|)/2 for
    both classes (symmetric accuracies), conditionally independent given y.
    Agreement in prediction space: i,j agree iff s_i == s_j (binary output).
    c_ij = 2*A_ij - 1 with A_ij the empirical agreement rate.
    """
    K = b.size
    a = (1.0 + np.abs(b)) / 2.0
    y = rng.random(m) < pi  # noqa: F841  (class draw; symmetric H => unused)
    s = rng.random((m, K)) < a[None, :]
    agree = np.einsum("mi,mj->ij", s.astype(float), s.astype(float))
    agree += np.einsum("mi,mj->ij", (~s).astype(float), (~s).astype(float))
    A = agree / m
    C = 2.0 * A - 1.0
    np.fill_diagonal(C, 1.0)
    return C


def _pair_index(K: int):
    pairs = [(k, l) for k in range(K) for l in range(k + 1, K)]
    return np.array(pairs, dtype=int)


def fit_rank_one_batch(C: np.ndarray) -> np.ndarray:
    """Vectorized fit_rank_one over a batch of agreement matrices (S, K, K)."""
    S, K, _ = C.shape
    P = _pair_index(K)
    b2 = np.zeros((S, K))
    for i in range(K):
        mask = (P[:, 0] != i) & (P[:, 1] != i)
        k, l = P[mask, 0], P[mask, 1]
        den = C[:, k, l]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(den) < 1e-9, np.nan,
                             C[:, i, k] * C[:, i, l] / den)
        b2[:, i] = np.nanmedian(ratio, axis=1)
    b = np.sqrt(np.clip(np.nan_to_num(b2), 0.0, 1.0))
    sign = np.sign(C[:, 0, :])
    sign[sign == 0] = 1.0
    sign[:, 0] = 1.0
    return b * sign


def simulate_H_batch(b: np.ndarray, m: int, n_sim: int,
                     rng: np.random.Generator) -> np.ndarray:
    """n_sim empirical agreement matrices from exact-H panels (vectorized)."""
    K = b.size
    a = (1.0 + np.abs(b)) / 2.0
    s = (rng.random((n_sim, m, K)) < a[None, None, :]).astype(np.float32)
    agree = np.einsum("smi,smj->sij", s, s) + np.einsum(
        "smi,smj->sij", 1.0 - s, 1.0 - s)
    C = 2.0 * agree / m - 1.0
    idx = np.arange(K)
    C[:, idx, idx] = 1.0
    return C


def tau_residual_batch(C: np.ndarray, b: np.ndarray) -> np.ndarray:
    K = C.shape[1]
    mask = ~np.eye(K, dtype=bool)
    R = C - np.einsum("si,sj->sij", b, b)
    num = np.linalg.norm(R[:, mask], axis=1)
    den = np.linalg.norm(C[:, mask], axis=1) + 1e-12
    return num / den


def tau_selfnorm(C_obs: np.ndarray, m: int, pi: float = 0.5, alpha: float = 0.05,
                 n_sim: int = 400, seed: int = 0) -> dict:
    """Self-normalized CEI statistic. Reject H iff tau_prime > 1.

    Returns dict with tau_obs, tau_star_local (the (1-alpha) null quantile at
    this panel's K, m, |b| scale), tau_prime, and the fitted |b|.
    """
    rng = np.random.default_rng(seed)
    b_hat, se_b = fit_rank_one(C_obs, return_se=True)
    t_obs = tau_residual(C_obs, b_hat)
    # smoothed parametric bootstrap: each null panel simulated at b perturbed
    # by the fit's own SE, so weak-agreement panels get properly wider nulls.
    B_sim = np.clip(b_hat[None, :]
                    + se_b[None, :] * rng.standard_normal((n_sim, b_hat.size)),
                    -0.999, 0.999)
    a_sim = (1.0 + np.abs(B_sim)) / 2.0
    s = (rng.random((n_sim, m, b_hat.size))
         < a_sim[:, None, :]).astype(np.float32)
    agree = np.einsum("smi,smj->sij", s, s) + np.einsum(
        "smi,smj->sij", 1.0 - s, 1.0 - s)
    C_sim = 2.0 * agree / m - 1.0
    idx = np.arange(b_hat.size)
    C_sim[:, idx, idx] = 1.0
    t_null = tau_residual_batch(C_sim, fit_rank_one_batch(C_sim))
    k = int(np.ceil((n_sim + 1) * (1 - alpha)))
    t_star = float(np.sort(t_null)[min(k, n_sim) - 1])
    return dict(tau_obs=t_obs, tau_star_local=t_star,
                tau_prime=t_obs / (t_star + 1e-12),
                reject_H=bool(t_obs > t_star),
                b_hat=b_hat.tolist(), m=m, alpha=alpha, n_sim=n_sim)
