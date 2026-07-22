"""kbound_repro.metrics -- the single canonical K-Bound decision-metric library.

This module is the ONE authoritative implementation of the offline decision
metrics used across the K-Bound project.  It is deliberately **torch-independent**
(standard library + numpy only) so it can be imported and unit-tested in any
environment, including a CPU-only CI box without the deep-learning stack.

Historically these metrics were re-implemented independently in many scripts
(``edge/src/kbound_edge/metrics.py``, ``scripts/build_results_source.py``,
``scripts/g8_*``, the ``gapclose_wave5`` runners, ...).  Those copies drifted --
most importantly some used a strict ``B < 0`` false-adapt boundary while the
certificate semantics require ``Delta <= 0`` (a tie is *unsafe* for a strict
adapt decision).  This module fixes the definition in one place; legacy copies
are retained for historical reproduction and pinned by equivalence tests.

Decision / action semantics (offline evaluation only)
-----------------------------------------------------
Each *condition* (one evaluated cell: a dataset x corruption x severity x seed
slice, etc.) carries a true benefit ``Delta`` (a.k.a. ``B``):

    Delta = acc(adapted candidate) - acc(frozen model)

and a realized decision in {ADAPT, FREEZE, ABSTAIN}.  The realized benefit of a
decision is::

    ADAPT   -> Delta            (the candidate is served)
    FREEZE  -> 0                (keep frozen)
    ABSTAIN -> 0                (documented frozen fallback: serve the frozen model)

The oracle adapts iff ``Delta > 0``, so oracle benefit is ``max(Delta, 0)`` and
per-condition realized regret is::

    regret_i = max(Delta_i, 0) - realized_i   >= 0

Canonical boundary (do not change without an audit note)
--------------------------------------------------------
* ``FALSE_ADAPT_BOUNDARY = "delta_le_0"``.
* Unconditional false-adapt   ``FA_u = mean( decision == ADAPT and Delta <= 0 )``
  over **all** conditions.
* Conditional  false-adapt    ``FA_c = (#ADAPT with Delta <= 0) / (#ADAPT)``
  over **ADAPT** conditions only.  ``FA_c`` has a *different name and a
  different denominator* from ``FA_u`` and is reported descriptively only
  (KB-CLAIM-004 is withdrawn: there is no theorem bounding ``FA_c``).
* ``Delta == 0`` counts as a false adapt (a tie is unsafe for strict adaptation).
* ABSTAIN contributes ``realized = 0`` to regret (the frozen fallback).

Empirical vs theoretical coverage
----------------------------------
``empirical_coverage`` here is purely descriptive -- the *observed* fraction of
conditions whose realized quantity landed inside a supplied certified region.
It is NOT a proof of the theoretical coverage guarantee and must never be
described as such (see project constraint: keep empirical coverage separate from
theoretically justified coverage).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants / vocabulary
# ---------------------------------------------------------------------------
ADAPT = "adapt"
FREEZE = "freeze"
ABSTAIN = "abstain"
VALID_ACTIONS = frozenset({ADAPT, FREEZE, ABSTAIN})

#: The false-adapt boundary is inclusive of zero (a tie is unsafe for a strict
#: adapt decision).  Recorded in manifests so downstream code can assert it.
FALSE_ADAPT_BOUNDARY = "delta_le_0"

__all__ = [
    "ADAPT",
    "FREEZE",
    "ABSTAIN",
    "VALID_ACTIONS",
    "FALSE_ADAPT_BOUNDARY",
    "realized_benefit",
    "regret_vector",
    "always_adapt_regret",
    "always_freeze_regret",
    "policy_regret",
    "regret_to_oracle",
    "false_adapt_unconditional",
    "false_adapt_conditional",
    "action_counts",
    "action_rates",
    "empirical_coverage",
    "DecisionSummary",
    "decision_summary",
    "wilson_interval",
    "paired_bootstrap_diff_ci",
    "holm_correction",
    "beats_both",
]


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------
def _as_actions(decisions: Sequence[str]) -> list[str]:
    acts = [str(d).strip().lower() for d in decisions]
    bad = sorted({a for a in acts if a not in VALID_ACTIONS})
    if bad:
        raise ValueError(
            f"unknown action(s) {bad!r}; expected a subset of {sorted(VALID_ACTIONS)}"
        )
    return acts


def _as_delta(deltas: Sequence[float]) -> np.ndarray:
    arr = np.asarray(deltas, dtype=float)
    if arr.ndim != 1:
        raise ValueError("deltas must be a 1-D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError("deltas contains non-finite values (NaN/inf)")
    return arr


def _check_paired(decisions: Sequence[str], deltas: Sequence[float]) -> tuple[list[str], np.ndarray]:
    acts = _as_actions(decisions)
    d = _as_delta(deltas)
    if len(acts) != d.shape[0]:
        raise ValueError(
            f"decisions (n={len(acts)}) and deltas (n={d.shape[0]}) must be equal length"
        )
    if not acts:
        raise ValueError("empty input: need at least one condition")
    return acts, d


# ---------------------------------------------------------------------------
# Core per-condition quantities
# ---------------------------------------------------------------------------
def realized_benefit(decision: str, delta: float) -> float:
    """Benefit actually obtained by following ``decision`` given true ``delta``.

    ADAPT serves the candidate (``delta``); FREEZE and ABSTAIN both serve the
    frozen model (``0.0``).
    """
    a = str(decision).strip().lower()
    if a not in VALID_ACTIONS:
        raise ValueError(f"unknown action {decision!r}")
    return float(delta) if a == ADAPT else 0.0


def regret_vector(decisions: Sequence[str], deltas: Sequence[float]) -> np.ndarray:
    """Per-condition realized regret ``max(delta,0) - realized`` (>= 0)."""
    acts, d = _check_paired(decisions, deltas)
    oracle = np.maximum(d, 0.0)
    realized = np.where(np.asarray(acts) == ADAPT, d, 0.0)
    reg = oracle - realized
    # numerical guard: regret is provably non-negative
    reg[np.abs(reg) < 1e-15] = 0.0
    return reg


def always_adapt_regret(deltas: Sequence[float]) -> float:
    """Mean regret of the always-adapt fixed policy: ``mean(max(-delta, 0))``."""
    d = _as_delta(deltas)
    return float(np.mean(np.maximum(-d, 0.0)))


def always_freeze_regret(deltas: Sequence[float]) -> float:
    """Mean regret of the always-freeze fixed policy: ``mean(max(delta, 0))``."""
    d = _as_delta(deltas)
    return float(np.mean(np.maximum(d, 0.0)))


def policy_regret(decisions: Sequence[str], deltas: Sequence[float]) -> float:
    """Mean realized regret of the KGA (or any) decision policy."""
    return float(np.mean(regret_vector(decisions, deltas)))


#: ``regret_to_oracle`` is a synonym for the policy's mean realized regret.
regret_to_oracle = policy_regret


# ---------------------------------------------------------------------------
# False-adapt (the canonical boundary lives here)
# ---------------------------------------------------------------------------
def false_adapt_unconditional(decisions: Sequence[str], deltas: Sequence[float]) -> float:
    """``FA_u = mean(decision == ADAPT and delta <= 0)`` over ALL conditions."""
    acts, d = _check_paired(decisions, deltas)
    a = np.asarray(acts)
    fa = (a == ADAPT) & (d <= 0.0)
    return float(np.mean(fa))


def false_adapt_conditional(decisions: Sequence[str], deltas: Sequence[float]) -> float | None:
    """``FA_c = (#ADAPT with delta <= 0) / (#ADAPT)`` over ADAPT conditions.

    Returns ``None`` when there are no ADAPT decisions (the denominator is
    undefined).  Descriptive only -- there is no theorem bounding ``FA_c``.
    """
    acts, d = _check_paired(decisions, deltas)
    a = np.asarray(acts)
    n_adapt = int(np.sum(a == ADAPT))
    if n_adapt == 0:
        return None
    n_bad = int(np.sum((a == ADAPT) & (d <= 0.0)))
    return n_bad / n_adapt


# ---------------------------------------------------------------------------
# Action counts / rates  (integer counts are always retained)
# ---------------------------------------------------------------------------
def action_counts(decisions: Sequence[str]) -> dict[str, int]:
    """Integer counts of adapt/freeze/abstain (raw, never reconstructed)."""
    acts = _as_actions(decisions)
    return {
        ADAPT: int(acts.count(ADAPT)),
        FREEZE: int(acts.count(FREEZE)),
        ABSTAIN: int(acts.count(ABSTAIN)),
    }


def action_rates(decisions: Sequence[str]) -> dict[str, float]:
    """Rates of adapt/freeze/abstain (counts / n)."""
    counts = action_counts(decisions)
    n = sum(counts.values())
    if n == 0:
        raise ValueError("empty decisions")
    return {k: counts[k] / n for k in (ADAPT, FREEZE, ABSTAIN)}


def empirical_coverage(covered_flags: Sequence[bool]) -> float:
    """Observed fraction of conditions marked covered (DESCRIPTIVE ONLY).

    This is the *empirical* coverage -- the observed hit rate of a supplied
    certified region.  It is not, and must not be presented as, a proof of the
    theoretical coverage guarantee.
    """
    flags = np.asarray(covered_flags, dtype=bool)
    if flags.size == 0:
        raise ValueError("empty covered_flags")
    return float(np.mean(flags))


# ---------------------------------------------------------------------------
# One-shot summary that retains raw integer counts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionSummary:
    n: int
    counts: dict  # integer adapt/freeze/abstain
    rates: dict
    false_adapt_boundary: str
    fa_u: float
    fa_c: float | None
    regret_kga: float
    regret_always_adapt: float
    regret_always_freeze: float
    mean_realized_benefit: float

    def to_dict(self) -> dict:
        return asdict(self)


def decision_summary(decisions: Sequence[str], deltas: Sequence[float]) -> DecisionSummary:
    """Full canonical summary for one policy on one set of conditions.

    Retains **integer** action counts so downstream schemas never have to
    reconstruct counts from rounded rates.
    """
    acts, d = _check_paired(decisions, deltas)
    counts = action_counts(acts)
    n = len(acts)
    realized = np.where(np.asarray(acts) == ADAPT, d, 0.0)
    return DecisionSummary(
        n=n,
        counts=counts,
        rates={k: counts[k] / n for k in (ADAPT, FREEZE, ABSTAIN)},
        false_adapt_boundary=FALSE_ADAPT_BOUNDARY,
        fa_u=false_adapt_unconditional(acts, d),
        fa_c=false_adapt_conditional(acts, d),
        regret_kga=policy_regret(acts, d),
        regret_always_adapt=always_adapt_regret(d),
        regret_always_freeze=always_freeze_regret(d),
        mean_realized_benefit=float(np.mean(realized)),
    )


# ---------------------------------------------------------------------------
# Interval estimators
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Uses ``statistics.NormalDist`` for the z quantile (no scipy dependency).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= successes <= n):
        raise ValueError("require 0 <= successes <= n")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    # snap tiny floating-point residue at the {0, 1} boundaries
    if lo < 1e-12:
        lo = 0.0
    if hi > 1.0 - 1e-12:
        hi = 1.0
    return (lo, hi)


def paired_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 0,
    statistic: str = "mean",
) -> dict:
    """Paired bootstrap CI for the difference ``stat(a) - stat(b)``.

    ``a`` and ``b`` are paired per-condition arrays (e.g. two policies' regrets).
    Resampling is done on the paired index so the pairing is preserved.  Returns
    the point difference, the CI, and whether the CI excludes zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length")
    if a.size == 0:
        raise ValueError("empty input")
    if statistic != "mean":
        raise ValueError("only statistic='mean' is supported")
    rng = np.random.default_rng(seed)
    n = a.size
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    lo, hi = np.quantile(diffs, [lo_q, hi_q])
    point = float(a.mean() - b.mean())
    return {
        "diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "n_boot": n_boot,
        "seed": seed,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def holm_correction(pvalues: Mapping[str, float] | Sequence[float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction.

    Accepts a mapping ``{name: p}`` or a sequence of p-values.  Returns, per
    hypothesis, the Holm-adjusted p-value and its reject/accept decision at
    ``alpha``.  Order-independent and monotone (adjusted p-values are enforced
    non-decreasing along the sorted order).
    """
    if isinstance(pvalues, Mapping):
        names = list(pvalues.keys())
        ps = [float(pvalues[k]) for k in names]
    else:
        ps = [float(p) for p in pvalues]
        names = [str(i) for i in range(len(ps))]
    m = len(ps)
    if m == 0:
        return {}
    if not all(0.0 <= p <= 1.0 for p in ps):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * ps[i]
        running = max(running, val)  # enforce monotonicity
        adj[i] = min(1.0, running)
    return {
        names[i]: {"p": ps[i], "p_holm": adj[i], "reject": bool(adj[i] <= alpha)}
        for i in range(m)
    }


# ---------------------------------------------------------------------------
# "Beats both" convenience (pre-registered WIN criterion helper)
# ---------------------------------------------------------------------------
def beats_both(
    decisions: Sequence[str],
    deltas: Sequence[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired-bootstrap 'beats both' test for a KGA policy on given conditions.

    KGA "beats both" when its per-condition regret is lower than BOTH the
    always-adapt and always-freeze fixed policies with paired-bootstrap CIs on
    the regret differences excluding zero.  ``FA_u`` is reported alongside.
    """
    acts, d = _check_paired(decisions, deltas)
    kga_reg = regret_vector(acts, d)
    aa_reg = np.maximum(-d, 0.0)  # always-adapt per-condition regret
    af_reg = np.maximum(d, 0.0)   # always-freeze per-condition regret
    vs_adapt = paired_bootstrap_diff_ci(aa_reg, kga_reg, confidence=confidence, n_boot=n_boot, seed=seed)
    vs_freeze = paired_bootstrap_diff_ci(af_reg, kga_reg, confidence=confidence, n_boot=n_boot, seed=seed)
    return {
        "false_adapt_boundary": FALSE_ADAPT_BOUNDARY,
        "fa_u": false_adapt_unconditional(acts, d),
        "regret_kga": float(kga_reg.mean()),
        "regret_always_adapt": float(aa_reg.mean()),
        "regret_always_freeze": float(af_reg.mean()),
        "vs_always_adapt": vs_adapt,
        "vs_always_freeze": vs_freeze,
        "beats_both": bool(vs_adapt["excludes_zero"] and vs_freeze["excludes_zero"]
                           and vs_adapt["diff"] > 0 and vs_freeze["diff"] > 0),
    }
