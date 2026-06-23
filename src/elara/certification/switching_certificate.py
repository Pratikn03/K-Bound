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

# ELARA + KGA merge: the empirical-Bernstein LCB formula now lives in exactly one
# place -- the canonical, typed, torch-free ``kga`` package (a top-level package,
# declared in pyproject ``package-dir``/``packages.find`` and already on the path
# wherever elara is). This module's ``empirical_bernstein_lcb`` delegates to it
# rather than re-deriving the Maurer-Pontil arithmetic. The bootstrap and the
# ``SwitchingCertificate`` gate/scenario machinery below have no kga equivalent
# and remain elara-specific. See docs/research/kbound/ELARA_KGA_MERGE_PLAN.md.
from kga.certificate import empirical_bernstein as _kga_empirical_bernstein


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
    # T5 Phase-2 additions: closed-form empirical-Bernstein LCB (no resampling).
    empirical_bernstein_lcb: float = float("nan")
    sample_variance: float = float("nan")
    benefit_range: float = float("nan")
    certified_eb: bool = False


def empirical_bernstein_lcb(
    paired_benefits: Sequence[float],
    *,
    alpha: float = 0.05,
    benefit_range: float | None = None,
) -> tuple[float, float, float]:
    """Closed-form Maurer-Pontil (2009) empirical-Bernstein lower confidence bound.

    For i.i.d. paired benefits X_i in [a, b] with empirical mean mu_hat and
    unbiased sample variance V_hat, with probability at least 1 - alpha:

        E[X] >= mu_hat - sqrt(2 V_hat ln(2/alpha) / n)
                       - 7 R ln(2/alpha) / (3 (n - 1))

    where R = b - a is the range of the benefit. This is tighter than the
    Hoeffding bound whenever V_hat << R^2, which is the typical regime for a
    fired-subset paired benefit (most fired samples agree in sign).

    Unlike the paired bootstrap, this LCB is deterministic (no resampling
    seed), streamable (mean + variance are sufficient statistics), and has a
    formal finite-sample guarantee rather than an asymptotic-percentile one.

    Returns (mean, lcb, sample_variance). If ``benefit_range`` is None it is
    estimated as max - min of the observed benefits (a valid range estimate
    when the loss surrogate is bounded; for |p - y| losses the exact range
    is 2.0, which the caller may pass explicitly).

    ELARA + KGA merge (single source of truth): on the normal path
    (``n >= 2``, finite inputs) the Maurer-Pontil arithmetic is computed by the
    canonical :func:`kga.certificate.empirical_bernstein`, of which this is a
    thin wrapper that adapts the canonical ``Certificate`` back to the
    ``(mean, lcb, sample_variance)`` tuple this module's callers expect. The
    two implementations are numerically identical on their shared domain
    (verified over 3000 randomized inputs, max abs diff 1.8e-15); see
    ``docs/research/kbound/ELARA_KGA_MERGE_PLAN.md``. The degenerate
    ``n == 0`` / ``n == 1`` returns below are kept here unchanged, since
    elara's callers (``fired_subset_certificate`` and the T5 audit scripts)
    depend on the nan / -inf return path rather than the ValueError that the
    canonical function raises for empty/non-finite input.
    """
    arr = np.asarray(list(paired_benefits), dtype=np.float64)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        # No variance estimate possible; fall back to a degenerate (mean only).
        return mean, float("-inf"), 0.0
    var = float(arr.var(ddof=1))
    if not np.all(np.isfinite(arr)):
        # Preserve this module's historical (pre-merge) behavior for non-finite
        # multi-element input: compute through and return NaNs rather than
        # raising. The canonical kga function validates finiteness and raises a
        # ValueError, so we only delegate on the finite path (where the two are
        # numerically identical). Real callers build a finite paired-benefit
        # vector by construction, so this branch is a defensive parity shim.
        mean_nf = float(arr.mean())
        R = float(benefit_range) if benefit_range is not None else float(arr.max() - arr.min())
        R = max(R, 1e-12)
        import math

        ln_term = math.log(2.0 / alpha)
        var_term = math.sqrt(2.0 * var * ln_term / n)
        range_term = 7.0 * R * ln_term / (3.0 * (n - 1))
        return mean_nf, float(mean_nf - var_term - range_term), var
    # Delegate the core LCB to the canonical kga implementation. The sample
    # variance is computed identically (ddof=1) and returned to preserve this
    # module's 3-tuple contract. cert.lower == delta_hat - epsilon reproduces
    # the Maurer-Pontil LCB above exactly.
    cert = _kga_empirical_bernstein(arr, alpha=alpha, benefit_range=benefit_range)
    return float(cert.delta_hat), float(cert.lower), var


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
    # |p - y| losses each lie in [0, 1], so the paired benefit lies in [-1, 1]
    # => exact range R = 2.0 for the empirical-Bernstein bound.
    eb_mean, eb_lcb, eb_var = empirical_bernstein_lcb(X, alpha=alpha, benefit_range=2.0)
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
        empirical_bernstein_lcb=float(eb_lcb),
        sample_variance=float(eb_var),
        benefit_range=2.0,
        certified_eb=bool(eb_lcb > 0.0),
    )
