"""Phase 2.G — finite-sample switching certificate.

For fired samples F = {i : g_i = 1}, define the paired loss benefit
   X_i = L_static(i) - L_gated(i).

A deployment window certifies the switch only if a paired-bootstrap
lower confidence bound

   LCB_alpha = mean_F - margin_alpha

is positive at the chosen significance alpha.

This is a retrospective evaluation certificate under the defined stress
protocol. It is NOT a production safety guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SwitchingCertificate:
    gate_id: str
    scenario_id: str
    n_fired_samples: int
    mean_paired_benefit: float
    bootstrap_lcb: float
    alpha: float
    n_iter: int
    certified: bool
    notes: str


def _loss_proxy(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """`|p - y|` bounded surrogate loss."""
    return np.abs(y_prob.astype(np.float64) - y_true.astype(np.float64))


def paired_bootstrap_lcb(
    paired_benefits: Sequence[float],
    *,
    alpha: float = 0.05,
    n_iter: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (mean_paired_benefit, LCB_alpha) by paired bootstrap over
    the fired-sample paired-benefit vector."""
    arr = np.asarray(list(paired_benefits), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boots[b] = arr[idx].mean()
    lcb = float(np.quantile(boots, alpha))
    return float(arr.mean()), lcb


def fired_subset_certificate(
    *,
    gate_id: str,
    scenario_id: str,
    static_scores: np.ndarray,
    gated_scores: np.ndarray,
    labels: np.ndarray,
    gate_fired: np.ndarray,
    alpha: float = 0.05,
    n_iter: int = 10_000,
    seed: int = 0,
    notes: str = "",
) -> SwitchingCertificate:
    """Build the certificate from per-sample prediction + gate-fire vectors."""
    fired = gate_fired.astype(bool)
    if not fired.any():
        return SwitchingCertificate(
            gate_id=gate_id,
            scenario_id=scenario_id,
            n_fired_samples=0,
            mean_paired_benefit=float("nan"),
            bootstrap_lcb=float("nan"),
            alpha=alpha,
            n_iter=n_iter,
            certified=False,
            notes=notes + " (no fired samples)",
        )
    l_static = _loss_proxy(labels[fired], static_scores[fired])
    l_gated = _loss_proxy(labels[fired], gated_scores[fired])
    X = l_static - l_gated
    mean, lcb = paired_bootstrap_lcb(X, alpha=alpha, n_iter=n_iter, seed=seed)
    return SwitchingCertificate(
        gate_id=gate_id,
        scenario_id=scenario_id,
        n_fired_samples=int(fired.sum()),
        mean_paired_benefit=float(mean),
        bootstrap_lcb=float(lcb),
        alpha=float(alpha),
        n_iter=int(n_iter),
        certified=bool(lcb > 0.0),
        notes=notes,
    )
