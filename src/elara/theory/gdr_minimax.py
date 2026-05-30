"""GDR — minimax optimality of the coherence-certified switching policy.

The coherence-certified gate decision rule (gate_decision_rule.py) switches to
the reliability path iff (a) drift is coherent across the batch AND (b) a
finite-sample switching certificate holds. This module gives the decision-
theoretic justification: that policy is MINIMAX over the union of two
adversarial regimes.

Two-regime adversary.
    Regime C (coherent shift): every domain's evidence degrades together.
        Switching to the reliability path REDUCES risk by b_C > 0.
        The batch reliability signal is tightly clustered -> high coherence.
    Regime H (heterogeneous mixture): categories differ but there is NO
        degradation. Switching FALSE-FIRES and INCREASES risk by c_H > 0.
        The reliability signal is dispersed -> low coherence.

Policy class.
    * always_switch:  fire on every batch.
    * never_switch:   keep static on every batch.
    * gdr(theta):     fire iff coherence(batch) >= theta.

Risk (regret vs the per-regime oracle).
    Let the adversary pick a mixture pi in [0,1] of regimes (pi = P[regime C]).
    For a policy p with fire-probabilities (f_C, f_H) in the two regimes:

        regret(p, pi) = pi * (1 - f_C) * b_C        (missed coherent benefit)
                      + (1 - pi) * f_H * c_H         (heterogeneous false-fire cost)

    The minimax value is  min_p max_pi regret(p, pi).

Theorem (GDR minimax).
    If the coherence signal separates the regimes with margin -- i.e. there is
    a threshold theta* such that coherence >= theta* w.p. >= 1 - eps in regime C
    and coherence < theta* w.p. >= 1 - eps in regime H -- then gdr(theta*) has

        max_pi regret(gdr) <= eps * max(b_C, c_H),

    whereas BOTH fixed policies have

        max_pi regret(always)  >= c_H        (pays full false-fire at pi -> 0)
        max_pi regret(never)   >= b_C        (pays full missed benefit at pi -> 1).

    Hence for eps < min(b_C, c_H) / max(b_C, c_H), gdr strictly minimaxes the
    fixed policies. As the coherence separation sharpens (eps -> 0), gdr's
    worst-case regret -> 0 while the fixed policies stay bounded away from 0.

This module provides:
    1. `regret(...)`                  -- closed-form regret of a policy at pi.
    2. `minimax_value(...)`           -- min-max regret over a policy set.
    3. `gdr_separation_eps(...)`      -- empirical regime-separation error of a
       coherence threshold on sampled batches.
    4. `validate_minimax(...)`        -- numerically confirm gdr minimaxes the
       fixed policies on a synthetic two-regime operator family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "regret",
    "minimax_value",
    "gdr_separation_eps",
    "GDRMinimaxResult",
    "validate_minimax",
]


def regret(f_C: float, f_H: float, pi: float, b_C: float, c_H: float) -> float:
    """Regret of a policy with fire-probs (f_C, f_H) at regime-mix pi."""
    return float(pi * (1.0 - f_C) * b_C + (1.0 - pi) * f_H * c_H)


def _max_regret_over_pi(f_C: float, f_H: float, b_C: float, c_H: float,
                        n_grid: int = 201) -> float:
    """max_pi regret -- regret is linear in pi so the max is at an endpoint."""
    return max(
        regret(f_C, f_H, 0.0, b_C, c_H),
        regret(f_C, f_H, 1.0, b_C, c_H),
    )


def minimax_value(policies: dict[str, tuple[float, float]], b_C: float, c_H: float) -> dict:
    """Return {policy: max_regret} and the minimax (argmin) policy."""
    maxr = {name: _max_regret_over_pi(fc, fh, b_C, c_H) for name, (fc, fh) in policies.items()}
    best = min(maxr, key=maxr.get)
    return {"max_regret": maxr, "minimax_policy": best, "minimax_value": maxr[best]}


def gdr_separation_eps(
    coherence_regime_C: np.ndarray,
    coherence_regime_H: np.ndarray,
    theta: float,
) -> dict:
    """Empirical regime-separation error of threshold theta.

    eps_C = P(coherence < theta | regime C)   (missed switches)
    eps_H = P(coherence >= theta | regime H)   (false fires)
    Returns both plus eps = max(eps_C, eps_H) and the implied (f_C, f_H).
    """
    cC = np.asarray(coherence_regime_C, dtype=float)
    cH = np.asarray(coherence_regime_H, dtype=float)
    f_C = float(np.mean(cC >= theta))   # fire prob in regime C
    f_H = float(np.mean(cH >= theta))   # fire prob in regime H
    eps_C = 1.0 - f_C
    eps_H = f_H
    return {"theta": float(theta), "f_C": f_C, "f_H": f_H,
            "eps_C": float(eps_C), "eps_H": float(eps_H),
            "eps": float(max(eps_C, eps_H))}


@dataclass
class GDRMinimaxResult:
    b_C: float
    c_H: float
    best_theta: float
    eps_at_best_theta: float
    gdr_max_regret: float
    always_max_regret: float
    never_max_regret: float
    gdr_is_minimax: bool
    per_theta: list[dict] = field(default_factory=list)


def validate_minimax(
    *,
    b_C: float = 0.10,
    c_H: float = 0.10,
    coherence_C_mean: float = 0.85,
    coherence_H_mean: float = 0.25,
    coherence_sd: float = 0.12,
    n_batches: int = 5000,
    theta_grid: tuple[float, ...] | None = None,
    seed: int = 0,
) -> GDRMinimaxResult:
    """Numerically confirm GDR minimaxes the fixed policies on a synthetic
    two-regime operator family.

    Regime C produces coherence ~ N(coherence_C_mean, sd) (clipped to [0,1]);
    regime H produces coherence ~ N(coherence_H_mean, sd). We sweep theta,
    compute the empirical (f_C, f_H) and thus the max-regret of gdr(theta),
    and compare against always_switch (1,1) and never_switch (0,0).
    """
    rng = np.random.default_rng(seed)
    cC = np.clip(rng.normal(coherence_C_mean, coherence_sd, n_batches), 0.0, 1.0)
    cH = np.clip(rng.normal(coherence_H_mean, coherence_sd, n_batches), 0.0, 1.0)

    if theta_grid is None:
        theta_grid = tuple(i / 100.0 for i in range(0, 101))

    always = _max_regret_over_pi(1.0, 1.0, b_C, c_H)
    never = _max_regret_over_pi(0.0, 0.0, b_C, c_H)

    per_theta = []
    best = None
    for theta in theta_grid:
        sep = gdr_separation_eps(cC, cH, theta)
        gdr_max = _max_regret_over_pi(sep["f_C"], sep["f_H"], b_C, c_H)
        row = {"theta": theta, "f_C": sep["f_C"], "f_H": sep["f_H"],
               "eps": sep["eps"], "gdr_max_regret": gdr_max}
        per_theta.append(row)
        if best is None or gdr_max < best["gdr_max_regret"]:
            best = row

    gdr_is_minimax = bool(best["gdr_max_regret"] < min(always, never) - 1e-9)
    return GDRMinimaxResult(
        b_C=b_C, c_H=c_H,
        best_theta=float(best["theta"]),
        eps_at_best_theta=float(best["eps"]),
        gdr_max_regret=float(best["gdr_max_regret"]),
        always_max_regret=float(always),
        never_max_regret=float(never),
        gdr_is_minimax=gdr_is_minimax,
        per_theta=per_theta,
    )
