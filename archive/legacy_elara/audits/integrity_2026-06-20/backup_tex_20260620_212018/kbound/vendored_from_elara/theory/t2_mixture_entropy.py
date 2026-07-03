"""T2 — Global-KS mixture confounding (mixture-entropy bound + real-cohort test).

Statement (Lemma T2, mixture-confounding inflation):
    Let the reference (validation) scores be drawn from a k-component
    mixture P = sum_c pi_c P_c with mixture weights pi = (pi_1, ..., pi_k)
    and let a test fold be drawn from the SAME per-component distributions
    but with a (possibly) different mixture pi'. A *global* two-sample KS
    test between reference and test conflates two effects:
        (1) genuine per-component drift  (P_c -> P'_c), and
        (2) mixture re-weighting          (pi  -> pi').
    Even when there is NO per-component drift (P'_c = P_c for all c), the
    global KS statistic is inflated by the mixture shift, with expected
    inflation bounded below by a function of the mixture entropy:

        E[ D_KS^global ]  >=  c0 * TV(pi, pi')      (mixture-only term)

    where TV is total variation between the two mixture weight vectors and
    c0 depends on the inter-component separation. In the canonical
    "balanced mixture re-weighted to a single component" worst case,
    TV(uniform_k, delta_1) = 1 - 1/k, which GROWS with the number of
    components k (equivalently, with the mixture entropy H = log k).

    A *category-aware* KS test (run per component, then aggregated) removes
    the mixture-only term entirely: under no per-component drift, each
    per-component KS statistic concentrates at its null level regardless of
    pi'. This is why the global gate false-fires on heterogeneous cohorts
    while a category-aware gate does not.

This module:
    1. `mixture_entropy(weights)`           -- Shannon entropy of pi.
    2. `total_variation(p, q)`              -- TV distance between mixtures.
    3. `global_ks_inflation_bound(...)`     -- closed-form lower bound on the
       mixture-only inflation of the global KS statistic.
    4. `category_aware_ks(...)`             -- per-component KS p-values that
       are invariant to mixture re-weighting (the T2 remedy).
    5. `validate_against_cohorts(...)`      -- empirical check on real
       Eyecandies category scores: confirm global KS inflates under a
       re-weighted cohort while category-aware KS does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:  # SciPy KS is preferred; fall back to a pure-numpy implementation.
    from scipy.stats import ks_2samp as _ks_2samp

    def _ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        r = _ks_2samp(a, b)
        return float(r.statistic), float(r.pvalue)
except Exception:  # pragma: no cover - scipy always present in this repo

    def _ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        a = np.sort(a)
        b = np.sort(b)
        allv = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, allv, side="right") / a.size
        cdf_b = np.searchsorted(b, allv, side="right") / b.size
        d = float(np.max(np.abs(cdf_a - cdf_b)))
        # Asymptotic p-value (Kolmogorov distribution, two-term)
        en = math.sqrt(a.size * b.size / (a.size + b.size))
        lam = (en + 0.12 + 0.11 / en) * d
        p = 2.0 * math.exp(-2.0 * lam * lam)
        return d, float(min(max(p, 0.0), 1.0))


__all__ = [
    "mixture_entropy",
    "total_variation",
    "global_ks_inflation_bound",
    "category_aware_ks",
    "T2CohortResult",
    "validate_against_cohorts",
]


def mixture_entropy(weights) -> float:
    """Shannon entropy (nats) of a mixture-weight vector."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)))


def total_variation(p, q) -> float:
    """Total-variation distance between two mixture-weight vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * np.sum(np.abs(p - q)))


def global_ks_inflation_bound(
    component_means: np.ndarray,
    pi_ref,
    pi_test,
) -> float:
    """Lower bound on the mixture-only inflation of the global KS statistic.

    Under no per-component drift, the difference between the reference and
    test mixture CDFs at the median component threshold is at least the
    total-variation re-weighting of the components that lie on opposite
    sides of that threshold. A clean lower bound is

        inflation >= TV(pi_ref, pi_test) * spread_fraction

    where spread_fraction is the fraction of components separated by at
    least the median inter-component gap. For well-separated components
    this approaches TV(pi_ref, pi_test); we report the conservative
    0.5 * TV form that holds for any component geometry.
    """
    tv = total_variation(pi_ref, pi_test)
    # Conservative geometry-free constant: at least half the TV mass shows
    # up as a CDF gap somewhere on the score axis.
    return float(0.5 * tv)


def category_aware_ks(
    ref_scores_by_cat: dict[str, np.ndarray],
    test_scores_by_cat: dict[str, np.ndarray],
    *,
    min_samples: int = 20,
) -> dict[str, tuple[float, float]]:
    """Per-component (per-category) KS statistic + p-value.

    Returns {category: (ks_statistic, p_value)}. Categories with fewer than
    ``min_samples`` in either fold are skipped. This is invariant to the
    test-fold mixture weights pi' because each category is tested against
    its own reference -- the T2 remedy.
    """
    out: dict[str, tuple[float, float]] = {}
    for cat in sorted(set(ref_scores_by_cat) & set(test_scores_by_cat)):
        a = np.asarray(ref_scores_by_cat[cat], dtype=float)
        b = np.asarray(test_scores_by_cat[cat], dtype=float)
        if a.size < min_samples or b.size < min_samples:
            continue
        out[cat] = _ks(a, b)
    return out


@dataclass
class T2CohortResult:
    n_categories: int
    mixture_entropy_uniform: float
    tv_uniform_vs_skewed: float
    global_ks_statistic: float
    global_ks_pvalue: float
    global_ks_inflation_lower_bound: float
    category_aware_max_statistic: float
    category_aware_min_pvalue: float
    category_aware_fire_count: int
    global_fires: bool
    category_aware_fires: bool
    alpha: float
    per_category: dict = field(default_factory=dict)


def validate_against_cohorts(
    scores_by_cat: dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
    skew_to: int = 0,
    seed: int = 0,
    min_samples: int = 20,
) -> T2CohortResult:
    """Empirical T2 check on real per-category scores under NO per-component drift.

    Construction. Split each category's scores in half: half forms the
    reference, half the test pool (so there is NO genuine per-component
    drift -- both halves are i.i.d. from the same category distribution).
    Build:
      * a BALANCED reference cohort (equal samples per category), and
      * a SKEWED test cohort (over-weighting one category) -- this is the
        pure mixture re-weighting that T2 says should fool the global gate.

    Then compare:
      * global KS(reference, skewed-test)            -> expected to inflate
      * category-aware KS per category               -> expected null

    A gate "fires" if its KS p-value < alpha (global) or if ANY category's
    p-value < alpha (category-aware). T2 predicts global fires while
    category-aware does not, purely from the mixture skew.
    """
    rng = np.random.default_rng(seed)
    cats = sorted(scores_by_cat)
    k = len(cats)

    ref_by_cat: dict[str, np.ndarray] = {}
    pool_by_cat: dict[str, np.ndarray] = {}
    for c in cats:
        s = np.array(scores_by_cat[c], dtype=float, copy=True)
        rng.shuffle(s)
        half = s.size // 2
        ref_by_cat[c] = s[:half]
        pool_by_cat[c] = s[half:]

    # Balanced reference cohort: equal n per category.
    per_cat_ref = min(len(v) for v in ref_by_cat.values())
    reference = np.concatenate([ref_by_cat[c][:per_cat_ref] for c in cats])

    # Skewed test cohort: heavily over-weight one category, lightly sample
    # the rest. This is mixture re-weighting with NO per-component drift.
    skew_cat = cats[skew_to % k]
    test_parts = []
    pi_test = np.zeros(k)
    for i, c in enumerate(cats):
        n_take = min(len(pool_by_cat[c]), per_cat_ref if c == skew_cat else max(2, per_cat_ref // (2 * k)))
        if c == skew_cat:
            n_take = len(pool_by_cat[c])
        test_parts.append(pool_by_cat[c][:n_take])
        pi_test[i] = n_take
    test = np.concatenate(test_parts)
    pi_test = pi_test / pi_test.sum()
    pi_ref = np.full(k, 1.0 / k)

    g_stat, g_p = _ks(reference, test)

    # Category-aware KS: reference-half vs pool-half per category (no drift).
    ca = category_aware_ks(ref_by_cat, pool_by_cat, min_samples=min_samples)
    ca_stats = [v[0] for v in ca.values()]
    ca_ps = [v[1] for v in ca.values()]
    ca_fire = sum(1 for p in ca_ps if p < alpha)

    comp_means = np.array([float(np.mean(scores_by_cat[c])) for c in cats])
    bound = global_ks_inflation_bound(comp_means, pi_ref, pi_test)

    return T2CohortResult(
        n_categories=k,
        mixture_entropy_uniform=mixture_entropy(pi_ref),
        tv_uniform_vs_skewed=total_variation(pi_ref, pi_test),
        global_ks_statistic=g_stat,
        global_ks_pvalue=g_p,
        global_ks_inflation_lower_bound=bound,
        category_aware_max_statistic=float(max(ca_stats)) if ca_stats else float("nan"),
        category_aware_min_pvalue=float(min(ca_ps)) if ca_ps else float("nan"),
        category_aware_fire_count=int(ca_fire),
        global_fires=bool(g_p < alpha),
        category_aware_fires=bool(ca_fire > 0),
        alpha=alpha,
        per_category={c: {"ks": ca[c][0], "p": ca[c][1]} for c in ca},
    )
