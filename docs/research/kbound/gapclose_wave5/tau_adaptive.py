"""Arm C (WIN_HUNT_v4) — drift-conditioned self-normalized tau' null.

Frozen per research_lock/WIN_HUNT_v4_PROTOCOL.yaml (arm_C_adaptive_tau). Extends
tau_selfnorm: the per-panel self-normalization is kept, but the DECISION
THRESHOLD is calibrated PER DRIFT TERCILE on DEV panels only and frozen — the
global null is replaced by a per-tercile null.

Drift statistic (label-free): d(C) = mean(|off-diagonal C|) — panel consensus /
disagreement proxy (lower agreement => more drift). Two tercile cutpoints are
frozen on DEV d-values. Within each tercile the pooled SELF-NORMALIZED bootstrap
null (null / tau_star_local, made scale-free per panel) has its Q_{1-alpha} taken
as a threshold multiplier c[tercile].

Decision: reject H iff  tau' = tau_obs / tau_star_local  >  c[tercile]
(c collapses to 1.0 -> identical to tau_selfnorm when calib is None, so rerun
scripts can swap tau_adaptive in for tau_selfnorm transparently).

API mirrors tau_selfnorm: same call signature (+ optional `calib`) and the same
return keys (tau_obs, tau_star_local, tau_prime, reject_H, b_hat, m, alpha,
n_sim) plus drift_stat, tercile, threshold_mult.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_selfnorm import (fit_rank_one, fit_rank_one_batch,  # noqa: E402
                          tau_residual, tau_residual_batch)


def drift_stat(C) -> float:
    """Label-free drift proxy: mean absolute off-diagonal agreement."""
    C = np.asarray(C, float)
    K = C.shape[0]
    mask = ~np.eye(K, dtype=bool)
    return float(np.mean(np.abs(C[mask])))


def _selfnorm_null(b_hat, se_b, m, n_sim, rng):
    """Smoothed parametric-bootstrap null tau array (identical law to tau_selfnorm)."""
    K = b_hat.size
    B_sim = np.clip(b_hat[None, :]
                    + se_b[None, :] * rng.standard_normal((n_sim, K)),
                    -0.999, 0.999)
    a_sim = (1.0 + np.abs(B_sim)) / 2.0
    s = (rng.random((n_sim, m, K)) < a_sim[:, None, :]).astype(np.float32)
    agree = np.einsum("smi,smj->sij", s, s) + np.einsum(
        "smi,smj->sij", 1.0 - s, 1.0 - s)
    C_sim = 2.0 * agree / m - 1.0
    idx = np.arange(K)
    C_sim[:, idx, idx] = 1.0
    return tau_residual_batch(C_sim, fit_rank_one_batch(C_sim))


def _selfnorm_stats(C_obs, m, alpha, n_sim, rng) -> dict:
    b_hat, se_b = fit_rank_one(C_obs, return_se=True)
    t_obs = tau_residual(C_obs, b_hat)
    null = _selfnorm_null(b_hat, se_b, m, n_sim, rng)
    k = int(np.ceil((n_sim + 1) * (1 - alpha)))
    t_star = float(np.sort(null)[min(k, n_sim) - 1])
    return dict(tau_obs=t_obs, tau_star_local=t_star,
                tau_prime=t_obs / (t_star + 1e-12), b_hat=b_hat, null=null)


def assign_tercile(d, cuts) -> int:
    return int(np.searchsorted(np.asarray(cuts, float), d, side="right"))


def calibrate_dev_terciles(dev_Cs, dev_ms, alpha=0.05, n_sim=400, seed=0) -> dict:
    """Freeze tercile cutpoints + per-tercile self-norm threshold on DEV panels.

    dev_Cs / dev_ms: agreement matrices and panel sizes from DEV panels ONLY.
    Empty tercile -> multiplier defaults to 1.0 (falls back to plain self-norm).
    """
    rng = np.random.default_rng(seed)
    ds, null_primes = [], []
    for C, m in zip(dev_Cs, dev_ms):
        st = _selfnorm_stats(np.asarray(C, float), int(m), alpha, n_sim, rng)
        ds.append(drift_stat(C))
        null_primes.append(st["null"] / (st["tau_star_local"] + 1e-12))
    ds = np.asarray(ds)
    cuts = [float(np.quantile(ds, 1.0 / 3.0)), float(np.quantile(ds, 2.0 / 3.0))]
    terc = np.array([assign_tercile(d, cuts) for d in ds])
    mult = {}
    for t in (0, 1, 2):
        members = [null_primes[i] for i in range(len(ds)) if terc[i] == t]
        if members:
            mult[t] = float(np.quantile(np.concatenate(members), 1.0 - alpha))
        else:
            mult[t] = 1.0
    return dict(cuts=cuts, mult=mult, alpha=float(alpha), n_dev=int(len(ds)),
                n_sim=int(n_sim))


def tau_adaptive(C_obs, m, pi: float = 0.5, alpha: float = 0.05,
                 n_sim: int = 400, seed: int = 0, calib: dict | None = None) -> dict:
    """Drift-conditioned self-normalized CEI gate. Reject H iff tau' > c[tercile].

    With calib=None this is EXACTLY tau_selfnorm (c = 1.0). `pi` is retained for
    signature parity (vacuous under symmetric-accuracy H, as in tau_selfnorm).
    """
    rng = np.random.default_rng(seed)
    C_obs = np.asarray(C_obs, float)
    st = _selfnorm_stats(C_obs, int(m), alpha, n_sim, rng)
    d = drift_stat(C_obs)
    if calib is not None:
        terc = assign_tercile(d, calib["cuts"])
        mult = calib["mult"]
        c = float(mult.get(terc, mult.get(str(terc), 1.0)))
    else:
        terc, c = None, 1.0
    return dict(tau_obs=st["tau_obs"], tau_star_local=st["tau_star_local"],
                tau_prime=st["tau_prime"], threshold_mult=c,
                reject_H=bool(st["tau_prime"] > c),
                b_hat=st["b_hat"].tolist(), m=int(m), alpha=alpha, n_sim=n_sim,
                drift_stat=d, tercile=terc)
