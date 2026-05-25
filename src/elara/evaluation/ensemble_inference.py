"""Phase 2 — seed-ensemble audited inference.

Implements the locked Phase-2 inference rule:

  1. Load per-seed test predictions for RGA+ and validation-frozen
     comparator from the archive.
  2. Verify sample-ID pairing across seeds and across methods.
  3. Compute seed-averaged ensemble prediction vectors per method.
  4. Compute paired DeLong p-value on the ensemble vectors.
  5. Compute paired bootstrap over test samples (10 000 iterations,
     fixed seed 0) for the 95% AUROC delta CI.
  6. Apply Holm-Bonferroni inside the named family.

The result is an audited inferential statement about the seed-
ensemble predictor; not a typical-single-trained-model claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# DeLong's paired ROC test (Sun & Xu 2014 fast-DeLong implementation)
# ---------------------------------------------------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Tied-rank vector for DeLong's algorithm."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(N, dtype=np.float64)
    out[J] = T
    return out


def _delong_compute(predictions: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute V_a, V_b matrices used by DeLong.

    `predictions`: (k_methods, n_samples).
    `ground_truth`: (n_samples,) ∈ {0, 1}.

    Returns:
      aucs:     (k,)
      cov:      (k, k)
    """
    order = ground_truth == 1
    n_pos = int(order.sum())
    n_neg = int((~order).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ground_truth must contain both classes")
    pos_idx = np.where(order)[0]
    neg_idx = np.where(~order)[0]
    k = predictions.shape[0]

    Tx = np.empty((k, n_pos))
    Ty = np.empty((k, n_neg))
    Tz = np.empty((k, n_pos + n_neg))
    for r in range(k):
        Tx[r] = _compute_midrank(predictions[r, pos_idx])
        Ty[r] = _compute_midrank(predictions[r, neg_idx])
        Tz[r] = _compute_midrank(predictions[r])
    aucs = (Tz[:, pos_idx].sum(axis=1) / n_pos - (n_pos + 1.0) / 2.0) / n_neg
    V01 = (Tz[:, pos_idx] - Tx) / n_neg
    V10 = 1.0 - (Tz[:, neg_idx] - Ty) / n_pos
    S01 = np.cov(V01)
    S10 = np.cov(V10)
    if k == 1:
        S01 = np.atleast_2d(S01)
        S10 = np.atleast_2d(S10)
    cov = S01 / n_pos + S10 / n_neg
    return aucs, cov


def _delong_two_sided_p(auc_a: float, auc_b: float, var: float) -> float:
    if var <= 0:
        return 1.0
    z = (auc_a - auc_b) / math.sqrt(var)
    # two-sided p
    from scipy.special import erf
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / math.sqrt(2.0))))


def seed_averaged_delong(
    *,
    rga_ensemble_scores: np.ndarray,
    comparator_ensemble_scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Paired DeLong on the seed-averaged ensemble prediction vectors."""
    preds = np.stack([rga_ensemble_scores, comparator_ensemble_scores], axis=0)
    aucs, cov = _delong_compute(preds, labels)
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    p = _delong_two_sided_p(aucs[0], aucs[1], var)
    return {
        "rga_ensemble_auc": float(aucs[0]),
        "comparator_ensemble_auc": float(aucs[1]),
        "delta_auc": float(aucs[0] - aucs[1]),
        "variance_of_delta": float(var),
        "delong_p_value": float(p),
    }


# ---------------------------------------------------------------------------
# Paired bootstrap over test samples (10 000 iterations, fixed seed)
# ---------------------------------------------------------------------------


def paired_sample_bootstrap_ci(
    *,
    rga_scores: np.ndarray,
    comparator_scores: np.ndarray,
    labels: np.ndarray,
    n_iter: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap over TEST SAMPLES (not over seeds) on the
    AUROC delta between rga and comparator. Both scores must be
    aligned with the labels vector."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if not (len(rga_scores) == len(comparator_scores) == n):
        raise ValueError("score and label vectors must align by sample")
    boot_deltas = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            boot_deltas[b] = np.nan
            continue
        a_r = roc_auc_score(y, rga_scores[idx])
        a_c = roc_auc_score(y, comparator_scores[idx])
        boot_deltas[b] = a_r - a_c
    finite = boot_deltas[np.isfinite(boot_deltas)]
    if finite.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_finite": 0}
    low = float(np.quantile(finite, alpha / 2))
    high = float(np.quantile(finite, 1.0 - alpha / 2))
    return {"ci_low": low, "ci_high": high, "n_finite": int(finite.size),
            "n_iter": int(n_iter), "alpha": float(alpha)}


# ---------------------------------------------------------------------------
# Practical-effect-size bands (Phase 2 statistical policy §5)
# ---------------------------------------------------------------------------


def practical_effect_band(delta_auc: float) -> str:
    a = abs(float(delta_auc))
    if a < 0.001:
        return "negligible"
    if a < 0.005:
        return "very small"
    if a < 0.01:
        return "small"
    if a < 0.03:
        return "moderate"
    return "large"


# ---------------------------------------------------------------------------
# Holm-Bonferroni within a single named family
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: dict[str, float], K: int | None = None) -> dict[str, float]:
    """Standard Holm-Bonferroni with explicit family size K (defaults to
    the number of finite p-values supplied)."""
    finite = [(k, float(p)) for k, p in p_values.items()
              if p is not None and isinstance(p, (int, float)) and math.isfinite(float(p))]
    finite.sort(key=lambda kv: kv[1])
    K = K or len(finite)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (k, raw_p) in enumerate(finite, start=1):
        cand = min(1.0, raw_p * (K - rank + 1))
        running = max(running, cand)
        out[k] = running
    return out


# ---------------------------------------------------------------------------
# High-level audited analysis driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedEnsembleAuditedAnalysis:
    cell_id: str
    benchmark: str
    protocol: str
    rga_method: str
    comparator_method: str
    n_seeds: int
    n_test_samples: int
    per_seed_rga_aucs: tuple[float, ...]
    per_seed_comp_aucs: tuple[float, ...]
    per_seed_deltas: tuple[float, ...]
    sign_consistent_seeds: int
    ensemble_rga_auc: float
    ensemble_comparator_auc: float
    ensemble_delta_auc: float
    delong_p_value: float
    delong_p_holm: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_n_iter: int
    practical_effect_band: str
    inference_label: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def audited_analysis(
    *,
    cell_id: str,
    benchmark: str,
    protocol: str,
    rga_method: str,
    comparator_method: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    per_seed_rga_scores: dict[int, np.ndarray],
    per_seed_comp_scores: dict[int, np.ndarray],
    holm_input: dict[str, float] | None = None,
    holm_K: int | None = None,
    bootstrap_n_iter: int = 10_000,
    bootstrap_seed: int = 0,
) -> SeedEnsembleAuditedAnalysis:
    """Run the full seed-ensemble audited analysis for one cell."""
    seeds = sorted(set(per_seed_rga_scores) & set(per_seed_comp_scores))
    if not seeds:
        raise ValueError("no seeds shared between RGA+ and comparator archives")
    rga_per_seed_aucs = []
    comp_per_seed_aucs = []
    deltas = []
    rga_score_stack = []
    comp_score_stack = []
    for s in seeds:
        rga = per_seed_rga_scores[s]
        comp = per_seed_comp_scores[s]
        if rga.shape[0] != labels.shape[0] or comp.shape[0] != labels.shape[0]:
            raise ValueError(f"seed {s}: prediction length mismatch")
        rga_score_stack.append(rga)
        comp_score_stack.append(comp)
        a_r = float(roc_auc_score(labels, rga))
        a_c = float(roc_auc_score(labels, comp))
        rga_per_seed_aucs.append(a_r)
        comp_per_seed_aucs.append(a_c)
        deltas.append(a_r - a_c)
    rga_ensemble = np.mean(np.stack(rga_score_stack, axis=0), axis=0)
    comp_ensemble = np.mean(np.stack(comp_score_stack, axis=0), axis=0)
    delong = seed_averaged_delong(
        rga_ensemble_scores=rga_ensemble,
        comparator_ensemble_scores=comp_ensemble,
        labels=labels,
    )
    boot = paired_sample_bootstrap_ci(
        rga_scores=rga_ensemble,
        comparator_scores=comp_ensemble,
        labels=labels,
        n_iter=bootstrap_n_iter,
        seed=bootstrap_seed,
    )
    # Holm correction: if holm_input is None, this cell stands alone (Holm-adjusted = raw).
    p_holm_map = holm_bonferroni(
        {cell_id: delong["delong_p_value"], **(holm_input or {})},
        K=holm_K,
    )
    p_holm = p_holm_map.get(cell_id, delong["delong_p_value"])
    band = practical_effect_band(delong["delta_auc"])
    sign_consistent = sum(1 for d in deltas if d * delong["delta_auc"] >= 0)
    return SeedEnsembleAuditedAnalysis(
        cell_id=cell_id,
        benchmark=benchmark,
        protocol=protocol,
        rga_method=rga_method,
        comparator_method=comparator_method,
        n_seeds=len(seeds),
        n_test_samples=int(len(labels)),
        per_seed_rga_aucs=tuple(rga_per_seed_aucs),
        per_seed_comp_aucs=tuple(comp_per_seed_aucs),
        per_seed_deltas=tuple(deltas),
        sign_consistent_seeds=int(sign_consistent),
        ensemble_rga_auc=float(delong["rga_ensemble_auc"]),
        ensemble_comparator_auc=float(delong["comparator_ensemble_auc"]),
        ensemble_delta_auc=float(delong["delta_auc"]),
        delong_p_value=float(delong["delong_p_value"]),
        delong_p_holm=float(p_holm),
        bootstrap_ci_low=float(boot["ci_low"]),
        bootstrap_ci_high=float(boot["ci_high"]),
        bootstrap_n_iter=int(boot.get("n_iter", bootstrap_n_iter)),
        practical_effect_band=band,
        inference_label=(
            "ensemble audited analysis (seed-averaged predictor): DeLong paired test on "
            "seed-averaged ensemble predictions + paired sample bootstrap CI over test "
            "rows; not independent confirmatory replication"
        ),
    )


__all__ = [
    "SeedEnsembleAuditedAnalysis",
    "audited_analysis",
    "paired_sample_bootstrap_ci",
    "seed_averaged_delong",
    "practical_effect_band",
    "holm_bonferroni",
]
