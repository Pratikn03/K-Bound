#!/usr/bin/env python3
"""Shared primitives for the K-Bound recomputation pack.

Conventions are lifted verbatim from the shipped code so that the "old" column
of every table reproduces the paper, and only the named change moves the "new"
column.

  interpolated radius (archived rule) : np.quantile(|bhat - B|, 1 - alpha)
      -> docs/research/kbound/scripts/cifar_tent_mps_v2.py:162
         docs/research/kbound/scripts/run_wilds_camelyon17.py:56
  exact-rank radius (promoted rule)   : rho_(k), k = min(n, ceil((n+1)(1-alpha)))
      -> docs/research/kbound/scripts/g8_canonical_pooling.py:4

  decision rule (all forks identical) :
      ADAPT   if bhat - eps > 0
      FREEZE  if bhat + eps < 0
      ABSTAIN otherwise
      -> g8_canonical_pooling.py:12, cifar_tent_mps_v2.py:163

  regret        : oracle = max(a0, a_adapted); policy accuracy; regret = oracle - policy
                  KGA takes a_adapted iff ADAPT, else a0 (ABSTAIN -> safe freeze)
      -> _locked_analysis_script.py:37-42
  FA_u (marginal, the quantity thm:certificate bounds)
                : mean( is_adapt & (B <= 0) )   -> _locked_analysis_script.py:43
  FA_c (conditional, the field name used by the runners)
                : mean( B <= 0 | is_adapt )
"""
import json
import math
import os

import numpy as np

REPO = "/home/claude/kb"
ALPHA = 0.10


# --------------------------------------------------------------------------- io
def read_json(path):
    """Read a JSON artifact, raising a clear error for iCloud placeholders."""
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    raw = open(p, "rb").read()
    if len(raw) == 0 or b"\x00" in raw:
        raise IOError(f"PLACEHOLDER (NUL-filled or empty, {len(raw)} bytes): {path}")
    return json.loads(raw)


def records(path):
    d = read_json(path)
    if isinstance(d, dict) and "records" in d:
        return d["records"]
    return d


# ----------------------------------------------------------------- radius rules
def eps_interp(rho):
    """Archived rule: interpolated empirical quantile over the whole pool."""
    return float(np.quantile(np.asarray(rho, dtype=float), 1 - ALPHA))


def eps_exact(rho, alpha=ALPHA):
    """Promoted rule: exact conformal rank k = min(n, ceil((n+1)(1-alpha)))."""
    r = np.sort(np.abs(np.asarray(rho, dtype=float)))
    n = len(r)
    k = min(n, int(math.ceil((n + 1) * (1 - alpha))))
    return float(r[k - 1])


def eps_exact_strict(rho, alpha=ALPHA):
    """Exact rank WITHOUT the min(n, .) clamp: returns inf when k > n.

    This is the variant three sibling scripts already use
    (ablation_exactrank.py:57, official_baselines_headtohead.py:48,
    reproduce_headlines.py:32) and the one that is honest at small n
    (fix-queue item 25).
    """
    r = np.sort(np.abs(np.asarray(rho, dtype=float)))
    n = len(r)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(r[k - 1])


RADIUS_RULES = {"interp": eps_interp, "exact": eps_exact, "exact_strict": eps_exact_strict}


def radii_in_pool(bhat, B, rule="exact"):
    """One radius for the whole file, computed from ALL residuals (as shipped)."""
    rho = np.abs(np.asarray(bhat, float) - np.asarray(B, float))
    e = RADIUS_RULES[rule](rho)
    return np.full(len(rho), e, dtype=float)


def radii_loo(bhat, B, rule="exact"):
    """Leave-one-out-of-pool radius: cell i's radius excludes cell i's residual."""
    rho = np.abs(np.asarray(bhat, float) - np.asarray(B, float))
    n = len(rho)
    out = np.empty(n, dtype=float)
    for i in range(n):
        pool = np.delete(rho, i)
        out[i] = RADIUS_RULES[rule](pool)
    return out


# ---------------------------------------------------------------- decision rule
def decide(bhat, eps):
    bhat = np.asarray(bhat, float)
    eps = np.asarray(eps, float)
    out = np.full(len(bhat), "ABSTAIN", dtype=object)
    out[bhat - eps > 0] = "ADAPT"
    out[(bhat + eps < 0) & (out == "ABSTAIN")] = "FREEZE"
    return out


# ---------------------------------------------------------------------- scoring
def score(dec, B, a0, aad):
    """Regret triple + decision accounting for one set of decisions."""
    B = np.asarray(B, float)
    a0 = np.asarray(a0, float)
    aad = np.asarray(aad, float)
    dec = np.asarray(dec, dtype=object)
    is_adapt = dec == "ADAPT"
    orc = np.maximum(a0, aad)
    kga = np.where(is_adapt, aad, a0)
    n = len(B)
    n_adapt = int(is_adapt.sum())
    fa_num = int(np.sum(is_adapt & (B <= 0)))
    return {
        "n": n,
        "regret_kga": float(np.mean(orc - kga)),
        "regret_adapt": float(np.mean(orc - aad)),
        "regret_freeze": float(np.mean(orc - a0)),
        "n_adapt": n_adapt,
        "n_freeze": int(np.sum(dec == "FREEZE")),
        "n_abstain": int(np.sum(dec == "ABSTAIN")),
        "fa_num": fa_num,
        "fa_u": fa_num / n,
        "fa_c": (fa_num / n_adapt) if n_adapt else None,
        "harmful_frac": float(np.mean(B < 0)),
        "regret_kga_vec": (orc - kga),
        "regret_adapt_vec": (orc - aad),
        "regret_freeze_vec": (orc - a0),
    }


def score_abs_B(dec, B):
    """The |B|-weighted regret used by g8_canonical_pooling.py:15 (Method B).

    Equivalent to score() whenever a_oracle = max(a0, a_adapted) and
    |B| = |a_adapted - a0|; kept separate so the ImageNet-C headline can be
    reproduced with the exact code path that produced it.
    """
    B = np.asarray(B, float)
    dec = np.asarray(dec, dtype=object)
    act = np.where(dec == "ADAPT", "ADAPT", "FREEZE")
    orc = np.where(B > 0, "ADAPT", "FREEZE")
    rk = np.abs(B) * (act != orc)
    ra = np.abs(B) * ("ADAPT" != orc)
    rf = np.abs(B) * ("FREEZE" != orc)
    return rk, ra, rf


# ------------------------------------------------------------------- statistics
def clopper_pearson_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound on a binomial rate.

    Returns None when n == 0 (bound undefined -- the guarantee is untested).
    """
    if n == 0:
        return None
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(conf, k + 1, n - k))


def clopper_pearson(k, n, conf=0.95):
    """Two-sided Clopper-Pearson interval (equal-tailed)."""
    if n == 0:
        return (None, None)
    from scipy.stats import beta
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return (lo, hi)


def wilson(k, n, conf=0.95):
    if n == 0:
        return (None, None)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def paired_boot(diff, nboot=20000, seed=20260611, clusters=None):
    """Paired bootstrap of mean(diff).

    clusters: optional array of cluster labels; if given, resamples whole
    clusters with replacement (cluster-robust / block bootstrap).
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    obs = float(np.mean(diff))
    if clusters is None:
        n = len(diff)
        idx = rng.integers(0, n, size=(nboot, n))
        bs = diff[idx].mean(axis=1)
    else:
        clusters = np.asarray(clusters)
        uniq = np.unique(clusters)
        groups = [diff[clusters == u] for u in uniq]
        m = len(uniq)
        bs = np.empty(nboot)
        pick = rng.integers(0, m, size=(nboot, m))
        for b in range(nboot):
            bs[b] = np.concatenate([groups[j] for j in pick[b]]).mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    centered = bs - bs.mean()
    p = (np.sum(np.abs(centered) >= abs(obs)) + 1) / (nboot + 1)
    return {"obs": obs, "lo": float(lo), "hi": float(hi), "p": float(p),
            "excludes_zero": bool(hi < 0 or lo > 0), "nboot": nboot}
