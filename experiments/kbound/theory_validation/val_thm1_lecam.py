#!/usr/bin/env python3
r"""
val_thm1_lecam.py
=================

Numerical validation of the *quantitative* upgrade of Theorem 1 of the K-Bound
paper (``thm:imp``), from an existence/non-identifiability witness to a Le Cam
two-point minimax lower bound on the committal error of any label-free rule.

Theorem (quantitative form being validated)
-------------------------------------------
Setting (label-free adaptation). Inputs X, hidden label Y, frozen model f0,
adapted model fa. Evidence  Z = phi(X_{1:n}, f0, fa)  is any statistic computable
WITHOUT target labels. Benefit  Delta = R_T(f0) - R_T(fa);  adapting is correct
iff Delta > 0. A label-free rule g(Z) in {adapt, freeze, abstain}.

For a two-world family {P1, P2} with Delta_1 < 0 < Delta_2, the *mixed committal
error*
        M(g) = P1(g = adapt) + P2(g = freeze)
satisfies, for ANY label-free rule g based on n unlabeled samples,
        inf_g M(g)  >=  1 - TV( P1_Z^n , P2_Z^n ),
where TV is the total variation distance between the n-sample EVIDENCE laws.

Two regimes are validated:
  (A) the EXACT witness of the paper: X ~ N(0,1) in BOTH worlds, f0 = 1[x>0],
      fa = 1[x<0], world-1 label Y = 1[x>0] (so Delta_1 = -1), world-2 label
      Y = 1[x<0] (so Delta_2 = +1). Because Z depends on X only and P1(X)=P2(X),
      TV(P1_Z^n, P2_Z^n) = 0 EXACTLY, so the bound reads inf_g M(g) >= 1. Hence
      every committal rule errs with total probability >= 1 across the two worlds
      (i.e. is wrong with prob >= 1/2 in at least one world): abstention is forced.

  (B) a DETECTABLE variant: the worlds additionally differ in the covariate law,
      P1(X) = N(-mu, 1), P2(X) = N(+mu, 1). Now Z (e.g. the sample mean confidence
      / score location) carries information, TV -> 1 as mu and/or n grow, and the
      committal error floor 1 - TV -> 0 at the rate governed by sample size and
      separation. We trace the rate.

What the script measures
------------------------
  1. Per-sample TV between the evidence laws Law(Z|P1) and Law(Z|P2) for several
     label-free Z, estimated by a fine histogram / kernel proxy (~0 in the witness,
     > 0 in the detectable variant).
  2. The n-sample evidence-law TV, TV(P1_Z^n, P2_Z^n), for an evidence statistic
     Z_n = T(X_{1:n}) that aggregates the batch (here the standardized sample mean
     of a per-sample label-free feature). For the witness this is ~0 for all n; for
     the detectable variant it grows with n, with the *Le Cam two-point bound*
     1 - TV plotted against n.
  3. A Monte-Carlo estimate of inf_g M(g) over a broad family of label-free
     committal rules g(Z_n) (all thresholds + polarities on each candidate Z), plus
     the actual K-Bound conformal adapt/freeze/abstain rule restricted to its
     committal actions. The empirical minimum tracks the theoretical floor 1 - TV.

It also reports the abstention behaviour of the full 3-way K-Bound rule on the
witness (should abstain ~100%), the empirical realisation already in the paper.

Run:
    python val_thm1_lecam.py
    python val_thm1_lecam.py --n-mc 400000 --seed 0 --json out.json

Dependencies: numpy, scipy (both already in the repo env). No labels, no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.stats import norm, ks_2samp


# --------------------------------------------------------------------------- #
#  Per-sample observables (all label-free: functions of X and the fixed maps)  #
# --------------------------------------------------------------------------- #
# f0(x) = 1[x>0], fa(x) = 1[x<0]. For 0/1 loss the per-sample model outputs and
# the disagreement region D = {x : f0 != fa} = {x != 0} (prob 1) are functions of
# X alone. A label-free feature is ANY function of X (and f0, fa). We use a panel:
#   z_x      : the raw covariate x                       (most informative possible)
#   z_absx   : |x|  (distance to boundary / margin)      (a confidence proxy)
#   z_f0     : f0(x) = 1[x>0]  (predicted class of f0)
#   z_disagr : 1[f0(x) != fa(x)] = 1[x != 0]             (disagreement indicator)
#   z_margin : a smooth "confidence" sigma(|x|) in (0.5,1)
# Crucially, NONE of these can see Y. In the witness all share the SAME law across
# worlds because X has the same law; in the detectable variant z_x and z_absx shift.

def per_sample_features(x: np.ndarray) -> dict[str, np.ndarray]:
    f0 = (x > 0).astype(float)
    fa = (x < 0).astype(float)
    return {
        "z_x": x,
        "z_absx": np.abs(x),
        "z_f0": f0,
        "z_disagree": (f0 != fa).astype(float),
        "z_conf": 1.0 / (1.0 + np.exp(-np.abs(x))),  # in (0.5, 1)
    }


# --------------------------------------------------------------------------- #
#  World samplers                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class World:
    """A target world for the two-point family.

    mu shifts the covariate law P(X) = N(mu, 1). label_rule in {"f0", "fa"}
    selects Y = 1[x>0] (matches f0, so adapting to fa HURTS, Delta<0) or
    Y = 1[x<0] (matches fa, so adapting HELPS, Delta>0).
    """
    name: str
    mu: float
    label_rule: str  # "f0" -> Y=1[x>0];  "fa" -> Y=1[x<0]

    def sample_x(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.mu + rng.standard_normal(n)

    def labels(self, x: np.ndarray) -> np.ndarray:
        if self.label_rule == "f0":
            return (x > 0).astype(float)
        elif self.label_rule == "fa":
            return (x < 0).astype(float)
        raise ValueError(self.label_rule)

    def delta_closed_form(self) -> float:
        r"""Delta = R_T(f0) - R_T(fa) for 0/1 loss, computed in closed form.

        f0 = 1[x>0], fa = 1[x<0]. On x != 0 exactly one of f0, fa equals any fixed
        target; for the label rules used here f0 and fa are deterministic functions
        of x, so the risks are exact:
          label_rule == "f0": Y = 1[x>0] = f0  => R(f0)=0, R(fa)=P(x!=0)=1 => Delta=-1
          label_rule == "fa": Y = 1[x<0] = fa  => R(fa)=0, R(f0)=1          => Delta=+1
        (Independent of mu, since both events have probability 1 under N(mu,1).)
        """
        return -1.0 if self.label_rule == "f0" else +1.0

    def delta_mc(self, n: int, rng: np.random.Generator) -> float:
        x = self.sample_x(n, rng)
        y = self.labels(x)
        f0 = (x > 0).astype(float)
        fa = (x < 0).astype(float)
        r0 = float(np.mean(f0 != y))
        ra = float(np.mean(fa != y))
        return r0 - ra


# --------------------------------------------------------------------------- #
#  TV estimators                                                              #
# --------------------------------------------------------------------------- #
def tv_histogram(a: np.ndarray, b: np.ndarray, bins: int = 200,
                 lo: float | None = None, hi: float | None = None) -> float:
    """Estimate TV(P_a, P_b) = 0.5 * integral |p_a - p_b| via a shared fine
    histogram. Consistent as bins, n -> inf; for identical laws the estimate is a
    small positive bias ~ O(bins / n) from finite-sample histogram noise, which we
    report honestly (it shrinks as n grows). For 0/1-valued features this is exact
    (two bins suffice)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    # discrete feature: exact TV over the support
    ua = np.unique(a); ub = np.unique(b)
    support = np.unique(np.concatenate([ua, ub]))
    if support.size <= 12:
        pa = np.array([np.mean(a == v) for v in support])
        pb = np.array([np.mean(b == v) for v in support])
        return float(0.5 * np.sum(np.abs(pa - pb)))
    if lo is None:
        lo = min(a.min(), b.min())
    if hi is None:
        hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(0.5 * np.sum(np.abs(pa - pb)))


def tv_histogram_debiased(a: np.ndarray, b: np.ndarray, bins: int = 200,
                          rng: np.random.Generator | None = None) -> dict:
    r"""Plug-in histogram TV has a known *positive* finite-sample bias: even when
    P_a == P_b, the empirical histograms differ by O(sqrt(bins/n)), so the raw
    estimate is > 0. We quantify and subtract this bias by a permutation null:
    pool a,b, repeatedly split into two halves of the same sizes, and average the
    raw TV under the null (where the two halves are i.i.d. from the SAME pooled
    law). The debiased estimate is  max(0, raw - null_mean). For identical laws the
    raw and null means coincide, so the debiased TV is ~0 (the honest answer); for
    genuinely different laws the raw exceeds the null and the excess survives."""
    if rng is None:
        rng = np.random.default_rng(0)
    raw = tv_histogram(a, b, bins=bins)
    pooled = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    na = len(a)
    nulls = []
    for _ in range(40):
        perm = rng.permutation(pooled.size)
        h1 = pooled[perm[:na]]; h2 = pooled[perm[na:]]
        nulls.append(tv_histogram(h1, h2, bins=bins))
    null_mean = float(np.mean(nulls))
    null_sd = float(np.std(nulls))
    return {
        "tv_raw": raw,
        "tv_null_mean": null_mean,      # the same-distribution sampling floor
        "tv_null_sd": null_sd,
        "tv_debiased": float(max(0.0, raw - null_mean)),
        "z_excess": float((raw - null_mean) / (null_sd + 1e-12)),  # std-devs above null
    }


def tv_two_gaussians(mu1: float, mu2: float, s1: float = 1.0, s2: float = 1.0) -> float:
    """Exact TV between N(mu1, s1^2) and N(mu2, s2^2). For equal variances s,
    TV = 2*Phi(|mu1-mu2| / (2 s)) - 1.  Used to give the closed-form Le Cam floor
    1 - TV for the detectable variant's aggregated evidence statistic."""
    if abs(s1 - s2) < 1e-12:
        d = abs(mu1 - mu2) / (2.0 * s1)
        return float(2.0 * norm.cdf(d) - 1.0)
    # general case: numeric integration
    grid = np.linspace(min(mu1, mu2) - 12 * max(s1, s2),
                       max(mu1, mu2) + 12 * max(s1, s2), 200_001)
    p1 = norm.pdf(grid, mu1, s1)
    p2 = norm.pdf(grid, mu2, s2)
    return float(0.5 * np.trapz(np.abs(p1 - p2), grid))


# --------------------------------------------------------------------------- #
#  n-sample evidence statistic and its TV                                     #
# --------------------------------------------------------------------------- #
# The strongest label-free batch statistic an adversary-proof rule could use is a
# sufficient summary of X_{1:n}. We use the standardized sample mean of the most
# informative per-sample feature (z_x), Z_n = sqrt(n) * mean(x_i). Under N(mu,1),
# Z_n ~ N(sqrt(n)*mu, 1). For the witness mu is the SAME in both worlds so the two
# Z_n laws coincide exactly (TV=0 for every n). For the detectable variant the two
# Z_n laws are N(-sqrt(n)*mu, 1) and N(+sqrt(n)*mu, 1), whose exact TV -> 1.

def aggregate_Zn(x_batch: np.ndarray) -> float:
    """Z_n = sqrt(n) * sample mean of x.  (Standardized; a label-free statistic.)"""
    n = x_batch.shape[-1]
    return math.sqrt(n) * float(np.mean(x_batch))


# --------------------------------------------------------------------------- #
#  Committal-rule families and the minimax error inf_g M(g)                    #
# --------------------------------------------------------------------------- #
def mc_committal_error_threshold_family(
    w1: World, w2: World, n: int, n_batches: int, rng: np.random.Generator,
    feature: str = "Zn",
) -> dict:
    r"""Monte-Carlo estimate of inf over a threshold rule family of the mixed
    committal error  M(g) = P1(g=adapt) + P2(g=freeze).

    Rule family (committal, label-free): for a scalar evidence statistic s and any
    threshold t and polarity p in {+1,-1},
        g_{t,p}(s) = adapt   if p*s > t,  else freeze.
    This is the optimal *committal* (no-abstain) family for a 1-D summary, and the
    minimax-over-thresholds error equals (up to MC noise) the Le Cam two-point
    floor 1 - TV(Law(s|P1), Law(s|P2)).  We draw n_batches batches of size n from
    each world, form the evidence statistic, and minimize the empirical
    P1(adapt)+P2(freeze) over a dense threshold grid and both polarities.

    feature: "Zn" -> standardized sample mean of x; or any per-sample feature name
    aggregated by its batch mean (a label-free summary).
    """
    def evidence(world: World) -> np.ndarray:
        # Draw ALL batches at once as an (n_batches x n) array (vectorized).
        X = world.mu + rng.standard_normal((n_batches, n))
        if feature == "Zn":
            return math.sqrt(n) * X.mean(axis=1)
        feats_per = per_sample_features(X)          # works elementwise on 2-D X
        return feats_per[feature].mean(axis=1)

    s1 = evidence(w1)  # evidence under world 1
    s2 = evidence(w2)  # evidence under world 2

    # Dense threshold grid spanning both samples.
    lo = min(s1.min(), s2.min()); hi = max(s1.max(), s2.max())
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, 4001)

    best = math.inf
    best_t = None; best_p = None
    for p in (+1.0, -1.0):
        ps1 = p * s1; ps2 = p * s2
        # g=adapt iff p*s > t. P1(adapt) = mean(ps1 > t); P2(freeze)=mean(ps2 <= t).
        # Vectorize over grid via sorting / searchsorted.
        ps1s = np.sort(ps1); ps2s = np.sort(ps2)
        # P1(adapt) = fraction of ps1 strictly > t
        p1_adapt = 1.0 - np.searchsorted(ps1s, grid, side="right") / ps1s.size
        # P2(freeze) = fraction of ps2 <= t
        p2_freeze = np.searchsorted(ps2s, grid, side="right") / ps2s.size
        M = p1_adapt + p2_freeze
        j = int(np.argmin(M))
        if M[j] < best:
            best = float(M[j]); best_t = float(grid[j]); best_p = float(p)

    # Empirical TV between the two evidence samples (histogram proxy) for reference.
    tv_emp = tv_histogram(s1, s2, bins=200)
    return {
        "feature": feature,
        "n_per_batch": int(n),
        "n_batches": int(n_batches),
        "inf_M_threshold_family": best,
        "best_threshold": best_t,
        "best_polarity": best_p,
        "lecam_floor_1_minus_TVemp": float(1.0 - tv_emp),
        "TV_evidence_emp": float(tv_emp),
    }


def kbound_conformal_committal_error(
    w1: World, w2: World, n: int, n_batches: int, rng: np.random.Generator,
    alpha: float = 0.10, max_instances: int = 4000,
) -> dict:
    r"""Run the ACTUAL K-Bound conformal adapt/freeze/abstain rule (the paper's
    decide_kga machinery, reproduced here without the sklearn dependency using a
    1-NN-in-evidence benefit estimator + split-conformal radius) on a pooled
    two-world dataset, and report:
      - abstention fraction (should be ~1 in the witness: the certified band
        straddles 0 because Z is uninformative about the benefit sign),
      - the committal error M = P1(adapt) + P2(freeze) of its committal actions,
      - false-adapt / false-freeze counts.

    Each "instance" is a batch of size n drawn from one of the two worlds; its
    evidence is the standardized sample mean Z_n; its true benefit B is the world's
    Delta. KGA never sees B at decision time except via the leave-one-out estimator
    trained on (Z_n, B) pairs -- exactly as in the paper.
    """
    # Build pooled instances: half from each world. (KGA's decision behaviour is
    # fully determined by a few thousand instances; we cap N for the O(N log N) LOO
    # estimator while the threshold-family MC above keeps the full batch count.)
    half = min(n_batches, max_instances) // 2
    sn = math.sqrt(n)
    Z1 = sn * (w1.mu + rng.standard_normal((half, n))).mean(axis=1)
    Z2 = sn * (w2.mu + rng.standard_normal((half, n))).mean(axis=1)
    Z = np.concatenate([Z1, Z2])
    B = np.concatenate([np.full(half, w1.delta_closed_form()),
                        np.full(half, w2.delta_closed_form())])
    world_id = np.concatenate([np.full(half, 1, dtype=int), np.full(half, 2, dtype=int)])

    # Leave-one-out benefit estimate: k-NN-in-evidence mean of B (a simple, sklearn-
    # free, consistent regressor). With uninformative Z the neighbours mix both
    # worlds so Bhat ~ 0; with informative Z, Bhat -> the world's true sign.
    # Vectorized: sort by the 1-D evidence so each point's nearest neighbours are
    # a contiguous window; take the k nearest by a small local search around the
    # sorted position (O(N log N) overall instead of O(N^2)).
    N = len(B)
    k = max(5, int(round(math.sqrt(N))))
    order = np.argsort(Z, kind="stable")
    zf_s = Z[order]; B_s = B[order]
    Bhat_s = np.empty(N)
    for r in range(N):
        lo = max(0, r - k); hi = min(N, r + k + 1)
        idx = np.arange(lo, hi)
        idx = idx[idx != r]                      # leave-one-out
        d = np.abs(zf_s[idx] - zf_s[r])
        nn = idx[np.argsort(d)[:k]]
        Bhat_s[r] = float(np.mean(B_s[nn]))
    Bhat = np.empty(N); Bhat[order] = Bhat_s     # unsort
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))

    abstain_frac = float(np.mean(dec == "ABSTAIN"))
    # Committal error on the two-point family: among instances, P1(adapt)+P2(freeze)
    is1 = world_id == 1; is2 = world_id == 2
    p1_adapt = float(np.mean(dec[is1] == "ADAPT")) if is1.any() else 0.0
    p2_freeze = float(np.mean(dec[is2] == "FREEZE")) if is2.any() else 0.0
    # false-adapt = adapt when true Delta<0 (world1); false-freeze = freeze when Delta>0 (world2)
    false_adapt = float(np.mean((dec == "ADAPT") & (B < 0)))
    false_freeze = float(np.mean((dec == "FREEZE") & (B > 0)))
    return {
        "alpha": alpha,
        "eps_radius": eps,
        "abstain_frac": abstain_frac,
        "P1_adapt": p1_adapt,
        "P2_freeze": p2_freeze,
        "kbound_committal_M": p1_adapt + p2_freeze,
        "false_adapt_rate": false_adapt,
        "false_freeze_rate": false_freeze,
        "decision_counts": {d: int(np.sum(dec == d)) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
    }


# --------------------------------------------------------------------------- #
#  Drivers                                                                     #
# --------------------------------------------------------------------------- #
def validate_witness(n_per_sample: int, n_batches: int, rng: np.random.Generator) -> dict:
    """Regime (A): the exact paper witness. P1(X)=P2(X)=N(0,1); opposite benefit.
    Expect: per-sample and n-sample evidence TV ~ 0, Le Cam floor ~ 1, every
    committal family error ~ 1, KGA abstains ~100%."""
    w1 = World("witness_world1_Delta=-1", mu=0.0, label_rule="f0")  # Delta = -1 (freeze correct)
    w2 = World("witness_world2_Delta=+1", mu=0.0, label_rule="fa")  # Delta = +1 (adapt correct)

    # closed-form vs MC sanity on Delta
    d1_cf, d2_cf = w1.delta_closed_form(), w2.delta_closed_form()
    d1_mc = w1.delta_mc(200_000, rng); d2_mc = w2.delta_mc(200_000, rng)

    # Per-sample TV for every feature (draw a big common pool from each world).
    # We report BOTH the raw plug-in TV and the permutation-debiased TV; the latter
    # is the honest answer (~0) once the same-distribution histogram noise floor is
    # removed. For the witness the two worlds share P(X), so true TV = 0 exactly.
    big = 300_000
    x1 = w1.sample_x(big, rng); x2 = w2.sample_x(big, rng)
    f1 = per_sample_features(x1); f2 = per_sample_features(x2)
    per_sample_tv = {k: tv_histogram_debiased(f1[k], f2[k], bins=300, rng=rng) for k in f1}
    # KS p-values (the paper reports these >0.05 across features)
    ks = {k: float(ks_2samp(f1[k][:5000], f2[k][:5000]).pvalue) for k in f1}

    # n-sample evidence-law TV for a grid of n (standardized sample mean Z_n).
    n_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    nsample_tv = []
    for n in n_grid:
        s1 = math.sqrt(n) * (w1.mu + rng.standard_normal((20_000, n))).mean(axis=1)
        s2 = math.sqrt(n) * (w2.mu + rng.standard_normal((20_000, n))).mean(axis=1)
        db = tv_histogram_debiased(s1, s2, bins=150, rng=rng)
        nsample_tv.append({"n": n, "TV_Zn_raw": db["tv_raw"],
                           "TV_Zn_debiased": db["tv_debiased"],
                           "TV_Zn_null_mean": db["tv_null_mean"],
                           "lecam_floor_1_minus_TVdebiased": 1.0 - db["tv_debiased"],
                           "TV_Zn_exact": tv_two_gaussians(0.0, 0.0)})  # exact = 0

    # inf_g M(g) over threshold families on several label-free statistics.
    thr = {}
    for feat in ["Zn", "z_x", "z_absx", "z_disagree", "z_conf"]:
        thr[feat] = mc_committal_error_threshold_family(
            w1, w2, n=n_per_sample, n_batches=n_batches, rng=rng, feature=feat)

    # The actual K-Bound 3-way rule.
    kga = kbound_conformal_committal_error(w1, w2, n=n_per_sample, n_batches=n_batches, rng=rng)

    return {
        "regime": "A_exact_witness",
        "description": "X~N(0,1) both worlds; f0=1[x>0], fa=1[x<0]; opposite labels.",
        "Delta_world1": {"closed_form": d1_cf, "mc": d1_mc},
        "Delta_world2": {"closed_form": d2_cf, "mc": d2_mc},
        "per_sample_TV": per_sample_tv,
        "per_sample_KS_pvalue": ks,
        "nsample_evidence_TV": nsample_tv,
        "inf_M_threshold_families": thr,
        "kbound_rule": kga,
    }


def validate_detectable(mu_grid: list[float], n_per_sample: int, n_batches: int,
                        rng: np.random.Generator) -> dict:
    """Regime (B): detectable variant. P1(X)=N(-mu,1), P2(X)=N(+mu,1) AND opposite
    benefit (world1 label=f0 => Delta=-1; world2 label=fa => Delta=+1). Now Z carries
    info: TV grows, the Le Cam floor 1-TV drops, committal error drops, KGA commits
    more (abstains less). We also sweep n at fixed mu to show the sqrt(n) rate."""
    out_mu = []
    for mu in mu_grid:
        w1 = World(f"det_world1_mu=-{mu}", mu=-mu, label_rule="f0")  # Delta=-1
        w2 = World(f"det_world2_mu=+{mu}", mu=+mu, label_rule="fa")  # Delta=+1
        # exact per-sample TV between N(-mu,1) and N(+mu,1)
        tv_x_exact = tv_two_gaussians(-mu, +mu)
        # n-sample Z_n laws: N(-sqrt(n)mu,1) vs N(+sqrt(n)mu,1)
        sn = math.sqrt(n_per_sample)
        tv_Zn_exact = tv_two_gaussians(-sn * mu, +sn * mu)
        thr = mc_committal_error_threshold_family(
            w1, w2, n=n_per_sample, n_batches=n_batches, rng=rng, feature="Zn")
        kga = kbound_conformal_committal_error(w1, w2, n=n_per_sample, n_batches=n_batches, rng=rng)
        out_mu.append({
            "mu": mu,
            "n_per_sample": n_per_sample,
            "TV_per_sample_x_exact": tv_x_exact,
            "TV_Zn_exact": tv_Zn_exact,
            "lecam_floor_1_minus_TV_Zn": 1.0 - tv_Zn_exact,
            "inf_M_threshold_Zn": thr["inf_M_threshold_family"],
            "TV_evidence_emp": thr["TV_evidence_emp"],
            "kbound_abstain_frac": kga["abstain_frac"],
            "kbound_committal_M": kga["kbound_committal_M"],
            "kbound_false_adapt": kga["false_adapt_rate"],
            "kbound_false_freeze": kga["false_freeze_rate"],
        })

    # rate sweep: fix a moderate per-sample separation, grow n.
    mu_fixed = 0.25
    n_sweep = [1, 4, 16, 64, 256, 1024]
    rate = []
    for n in n_sweep:
        w1 = World("rate_w1", mu=-mu_fixed, label_rule="f0")
        w2 = World("rate_w2", mu=+mu_fixed, label_rule="fa")
        sn = math.sqrt(n)
        tv_Zn_exact = tv_two_gaussians(-sn * mu_fixed, +sn * mu_fixed)
        thr = mc_committal_error_threshold_family(
            w1, w2, n=n, n_batches=n_batches, rng=rng, feature="Zn")
        rate.append({
            "n": n, "mu_per_sample": mu_fixed,
            "TV_Zn_exact": tv_Zn_exact,
            "lecam_floor_1_minus_TV": 1.0 - tv_Zn_exact,
            "inf_M_threshold_Zn": thr["inf_M_threshold_family"],
        })

    return {
        "regime": "B_detectable",
        "description": "P1(X)=N(-mu,1), P2(X)=N(+mu,1); opposite benefit; Z informative.",
        "mu_sweep": out_mu,
        "rate_vs_n_at_mu=0.25": rate,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-per-sample", type=int, default=64,
                    help="batch size n of unlabeled samples per evidence instance")
    ap.add_argument("--n-mc", type=int, default=20000,
                    help="number of MC batches (instances) per world")
    ap.add_argument("--json", type=str, default=None,
                    help="optional path to dump full results as JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    witness = validate_witness(args.n_per_sample, args.n_mc, rng)
    detectable = validate_detectable(
        mu_grid=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        n_per_sample=args.n_per_sample, n_batches=args.n_mc, rng=rng)

    results = {
        "config": vars(args),
        "theorem": "inf_g [ P1(g=adapt) + P2(g=freeze) ] >= 1 - TV(P1_Z^n, P2_Z^n)",
        "witness": witness,
        "detectable": detectable,
    }

    # ---- pretty report ----
    print("=" * 78)
    print("VALIDATION: quantitative Theorem 1 (Le Cam two-point lower bound)")
    print("  inf_g [ P1(g=adapt) + P2(g=freeze) ]  >=  1 - TV(P1_Z^n, P2_Z^n)")
    print("=" * 78)

    print("\n[REGIME A] EXACT WITNESS  (X~N(0,1) both worlds; opposite benefit)")
    print(f"  Delta_world1 = {witness['Delta_world1']['closed_form']:+.3f} "
          f"(MC {witness['Delta_world1']['mc']:+.3f})   "
          f"Delta_world2 = {witness['Delta_world2']['closed_form']:+.3f} "
          f"(MC {witness['Delta_world2']['mc']:+.3f})")
    print("  Per-sample TV between evidence laws (raw plug-in vs permutation-debiased;")
    print("  debiased ~0 = honest answer once same-dist histogram noise floor removed):")
    for k, v in witness["per_sample_TV"].items():
        print(f"      TV[{k:<11s}] raw={v['tv_raw']:.5f}  null_floor={v['tv_null_mean']:.5f}  "
              f"debiased={v['tv_debiased']:.5f}  (excess {v['z_excess']:+.1f}sd; KS p="
              f"{witness['per_sample_KS_pvalue'][k]:.3f})")
    print("  n-sample evidence-law TV  TV(P1_Zn^n, P2_Zn^n)  (exact = 0 for all n):")
    for r in witness["nsample_evidence_TV"]:
        print(f"      n={r['n']:>4d} : TV_raw={r['TV_Zn_raw']:.5f} (null {r['TV_Zn_null_mean']:.5f})  "
              f"TV_debiased={r['TV_Zn_debiased']:.5f}  "
              f"floor(1-TV_deb)={r['lecam_floor_1_minus_TVdebiased']:.5f}")
    print("  inf_g M(g) over committal threshold families (should be ~1):")
    for feat, r in witness["inf_M_threshold_families"].items():
        print(f"      Z={feat:<11s}: inf_M={r['inf_M_threshold_family']:.4f}   "
              f"(1 - TV_emp = {r['lecam_floor_1_minus_TVemp']:.4f})")
    kg = witness["kbound_rule"]
    print(f"  K-Bound 3-way rule on witness: abstain={kg['abstain_frac']*100:.1f}%  "
          f"committal_M={kg['kbound_committal_M']:.4f}  "
          f"false_adapt={kg['false_adapt_rate']:.4f}  false_freeze={kg['false_freeze_rate']:.4f}")
    print(f"      decision counts: {kg['decision_counts']}")

    print("\n[REGIME B] DETECTABLE VARIANT  (P1(X)=N(-mu,1), P2(X)=N(+mu,1))")
    print(f"  per-instance batch size n = {args.n_per_sample}")
    print("   mu   TV(per-sample) TV(Zn,exact)  floor(1-TV)  inf_M(thr)  KGA-abstain  KGA-M")
    for r in detectable["mu_sweep"]:
        print(f"  {r['mu']:>4.2f}   {r['TV_per_sample_x_exact']:.4f}        "
              f"{r['TV_Zn_exact']:.4f}      {r['lecam_floor_1_minus_TV_Zn']:.4f}     "
              f"{r['inf_M_threshold_Zn']:.4f}      {r['kbound_abstain_frac']*100:5.1f}%    "
              f"{r['kbound_committal_M']:.4f}")
    print("  Rate vs n at fixed per-sample mu=0.25 (floor 1-TV -> 0, inf_M tracks it):")
    print("     n     TV(Zn,exact)  floor(1-TV)  inf_M(thr)")
    for r in detectable["rate_vs_n_at_mu=0.25"]:
        print(f"   {r['n']:>5d}    {r['TV_Zn_exact']:.4f}      "
              f"{r['lecam_floor_1_minus_TV']:.4f}     {r['inf_M_threshold_Zn']:.4f}")

    print("\nINTERPRETATION")
    print("  - Regime A: evidence laws coincide (TV=0 for all n) => floor=1 => every")
    print("    committal rule (incl. the optimal threshold family) has M ~ 1, i.e. is")
    print("    wrong with prob ~1/2 in one world. KGA abstains ~100%. Abstention forced.")
    print("  - Regime B: as worlds become detectable (mu, n grow) TV->1, the floor")
    print("    1-TV->0, the minimax committal error ->0, and KGA stops abstaining.")
    print("    The empirical inf_M tracks the closed-form Le Cam floor 1-TV throughout.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n[wrote full results -> {args.json}]")


if __name__ == "__main__":
    main()
