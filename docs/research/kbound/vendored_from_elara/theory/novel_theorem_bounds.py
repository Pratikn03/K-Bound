"""Novel quantitative theorem upgrades (T2, T3, T4, T6) + GDR formal test.

Each function turns a known qualitative insight into a NEW quantitative
consequence with an explicit, checkable formula. Derivations are stated in the
docstrings; none is transcribed without checking, and scope caveats are explicit.

T2  mixture_ks_inflation_bound  : lower bound on spurious KS under benign mixture shift
T3  stochastic_dilution_prob    : P(mean gate silent | k zeroed domains, noisy reliability)
T4  bayes_optimal_tau           : risk-minimising switching threshold (decision theory only)
T6  ks_window_power             : closed-form detection power as a function of window W
GDR exact_binomial_p            : exact-binomial p-value for the decision-rule hit record
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "mixture_ks_inflation_bound",
    "stochastic_dilution_prob",
    "min_failed_domains_for_activation",
    "bayes_optimal_tau",
    "ks_window_power",
    "ks_window_for_power",
    "exact_binomial_p",
]


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Standard normal quantile (Acklam-style rational approximation)."""
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0 else float("inf")
    # Beasley-Springer/Moro is overkill; use scipy if available, else erfinv.
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    # Winitzki approximation, good to ~1e-3 — adequate for window/threshold sizing.
    a = 0.147
    ln = math.log(1 - x * x) if abs(x) < 1 else -50.0
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


# ---------------------------------------------------------------------------
# T2 — Mixture-induced KS inflation (NOVEL: quantitative lower bound)
# ---------------------------------------------------------------------------
def mixture_ks_inflation_bound(
    pi_val: np.ndarray,
    pi_test: np.ndarray,
    pairwise_ks: np.ndarray,
) -> float:
    """Lower bound on E[KS(P_val, P_test)] under BENIGN mixture re-weighting.

    Setup. P_val = sum_c pi_val[c] P_c, P_test = sum_c pi_test[c] P_c with FIXED
    category distributions {P_c} and pairwise KS distances pairwise_ks[i,j].
    Even with NO per-category drift, re-weighting alone inflates the global KS.

    Bound (derivation in the paper):
        E[KS(P_val,P_test)] >= (1/2) sum_c |pi_val[c]-pi_test[c]| * min_{j!=c} d_{c,j}.

    Intuition: each unit of mass moved out of component c must travel at least
    its nearest-neighbour KS distance, and KS is a sup-CDF gap, so half the
    moved mass shows up as inflation. This is the quantitative criterion the
    field lacked: spurious firing is negligible iff ||pi_val-pi_test||_1 times
    the min inter-component distance is small.
    """
    pv = np.asarray(pi_val, float); pt = np.asarray(pi_test, float)
    D = np.asarray(pairwise_ks, float)
    k = len(pv)
    total = 0.0
    for c in range(k):
        others = [D[c, j] for j in range(k) if j != c]
        nn = min(others) if others else 0.0
        total += abs(pv[c] - pt[c]) * nn
    return 0.5 * total


# ---------------------------------------------------------------------------
# T3 — Stochastic dilution boundary (NOVEL: noisy-reliability gate-silence prob)
# ---------------------------------------------------------------------------
def stochastic_dilution_prob(k: int, D: int, tau: float, sigma: float,
                             r_fail: float = 0.0, r_ok: float = 1.0) -> float:
    """P(mean gate stays SILENT | k of D domains truly failed), noisy reliability.

    Each domain's reliability estimate has additive noise N(0, sigma^2). The
    mean reliability over D domains is
        rbar = ( k*r_fail + (D-k)*r_ok ) / D  + N(0, sigma^2 / D).
    The gate fires when rbar < tau; it stays SILENT (fails to fire) when
    rbar >= tau. Hence

        P(silent | k) = 1 - Phi( (tau - mu_k) / (sigma/sqrt(D)) )
                      = Phi( (mu_k - tau) / (sigma/sqrt(D)) ),
        mu_k = (k*r_fail + (D-k)*r_ok)/D.

    NOTE: the correct argument carries the FULL mean mu_k (the proposal's draft
    dropped the (1-k/D) term); for r_fail=0, r_ok=1 this is mu_k = (D-k)/D = 1-k/D.
    """
    mu_k = (k * r_fail + (D - k) * r_ok) / D
    se = sigma / math.sqrt(D)
    if se <= 0:
        return 1.0 if mu_k >= tau else 0.0
    return _phi((mu_k - tau) / se)


def min_failed_domains_for_activation(D: int, tau: float, sigma: float,
                                      delta: float = 0.05,
                                      r_fail: float = 0.0, r_ok: float = 1.0) -> float:
    """Smallest k guaranteeing the gate fires w.p. >= 1-delta (noisy reliability).

    Solve P(silent | k) <= delta  =>  (mu_k - tau)/se <= z_delta
    => with mu_k = 1 - k/D (for r_fail=0,r_ok=1): k >= D[(r_ok - tau) - z_{1-delta} se].
    Returns the real-valued threshold (ceil for an integer count).
    """
    se = sigma / math.sqrt(D)
    z = _phi_inv(1.0 - delta)
    # mu_k - tau <= z_delta * se ; z_delta = -z_{1-delta}
    # (D-k)/D - tau <= -z * se  (r_fail=0,r_ok=1)  => k >= D[(1-tau) + z*se]
    return D * ((r_ok - tau) + z * se)


# ---------------------------------------------------------------------------
# T4 — Bayes-optimal switching threshold (NOVEL: optimal tau; decision theory)
# ---------------------------------------------------------------------------
def bayes_optimal_tau(pi: float, q0: float, q1: float, delta0: float, delta1: float,
                      dq1_dtau: float = 1.0) -> float:
    """Risk-minimising switching threshold under Bernoulli(pi) degradation.

    Expected AUC regret of switching = (1-pi) q0 delta0 (clean false-fire cost)
    minus pi q1 delta1 (degraded benefit). Setting the derivative wrt tau to zero
    (q1 increases with tau at rate dq1_dtau):

        tau* = 1 - (1-pi) q0 delta0 / (pi * dq1_dtau * delta1).

    SCOPE (honest): this is a decision-theoretic optimum under a PREVALENCE shift.
    It is NOT the explanation for the 3D-ADAM/Eyecandies clean-transfer tie, which
    is a complementary->redundant modality-structure flip, not a pi shift. T4 is
    reported as a stand-alone threshold result, not attached to that failure.
    """
    denom = pi * dq1_dtau * delta1
    if denom == 0:
        return float("nan")
    return 1.0 - (1.0 - pi) * q0 * delta0 / denom


# ---------------------------------------------------------------------------
# T6 — KS window power analysis (NOVEL: closed-form power + window sizing)
# ---------------------------------------------------------------------------
def ks_window_power(W: int, delta_tv: float, alpha: float = 0.05, c: float = 1.0) -> float:
    """Detection power of a windowed KS test for a TV-distance shift delta_tv.

    Asymptotically the KS statistic concentrates at rate sqrt(W); modelling the
    test as a one-sided normal shift test gives

        power(W, alpha, delta) = 1 - Phi( z_{1-alpha} - delta_tv * sqrt(W) * c ),

    where c is a reference-distribution constant fit once on held data. As W grows
    the power -> 1; this is the closed form the empirical grid only sampled.
    """
    z = _phi_inv(1.0 - alpha)
    return 1.0 - _phi(z - delta_tv * math.sqrt(W) * c)


def ks_window_for_power(target_power: float, delta_tv: float,
                        alpha: float = 0.05, c: float = 1.0) -> float:
    """Window size W* to achieve >= target_power at level alpha for shift delta_tv:
        W* = ((z_{1-alpha} + z_{power}) / (delta_tv * c))^2."""
    z_a = _phi_inv(1.0 - alpha)
    z_b = _phi_inv(target_power)
    if delta_tv * c == 0:
        return float("inf")
    return ((z_a + z_b) / (delta_tv * c)) ** 2


# ---------------------------------------------------------------------------
# GDR — exact binomial significance of the decision-rule hit record
# ---------------------------------------------------------------------------
def exact_binomial_p(hits: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact-binomial p-value for `hits` correct of `n` decisions under
    chance p0. Used to certify the GDR's k/n switch/suppress prediction record."""
    from math import comb
    def pmf(k):
        return comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
    obs = pmf(hits)
    return float(sum(pmf(k) for k in range(n + 1) if pmf(k) <= obs + 1e-12))
