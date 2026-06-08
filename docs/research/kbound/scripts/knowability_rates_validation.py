"""Validation for the KNOWABILITY RATE theorem (matching upper/lower bounds).

Theorem R (validated here), for paired benefits X_i in [-R/2, R/2], sd sigma, mean Delta:
  UPPER (adaptive): the deployed empirical-Bernstein certificate (Maurer-Pontil; the
    same formula as vendored_from_elara/certification/switching_certificate.py, cross-
    checked below) certifies sign(Delta) with wrong-commit <= alpha and abstention <= beta
    whenever |Delta| >= kappa_n^UB := c1*sigma*sqrt(log(1/a')/n) + c2*R*log(1/a')/n,
    WITHOUT knowing sigma.
  LOWER: any rule with wrong-commit <= alpha in every world and abstention <= beta at
    margin kappa must have
      kappa >= kappa_n^LB := max( (sigma/2)*sqrt( ln(1/(4(alpha+beta))) / n ),
                                  (R/4) * ln(1/(2(alpha+beta))) / n ),
    via Bernoulli two-point + Bretagnolle-Huber (dense regime) and a spike two-point
    (Bernstein regime).
  => kappa_n^UB / kappa_n^LB = O(1): the certificate is rate-optimal and ADAPTIVE.

Checks performed (all saved to results/theory/knowability_rates_validation.json):
  1. EB cross-check: vectorized EB == vendored empirical_bernstein_lcb (100 cases).
  2. Empirical minimal certifiable margin kappa_hat(n): log-log slope ~ -1/2 in the
     dense regime and ~ -1 in the spike regime; one certificate, no sigma knowledge.
  3. kappa_hat(n) sits ABOVE the lower bound with bounded ratio (constant factor).
  4. Infeasibility below the bound: at kappa = 0.5*kappa^LB the LIKELIHOOD-RATIO-optimal
     test's achievable error sum (exact, computed from the two-point pair) already
     exceeds the (alpha+beta) budget -> no valid rule exists there, certificate or not.
"""
import os, sys, json, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(HERE)
OUTD = os.path.join(KB, "results", "theory"); os.makedirs(OUTD, exist_ok=True)
OUTJ = os.path.join(OUTD, "knowability_rates_validation.json")
FIGD = os.path.join(KB, "figures"); FIGF = os.path.join(FIGD, "final")
sys.path.insert(0, os.path.join(KB, "vendored_from_elara"))
rng = np.random.default_rng(0)

ALPHA, BETA = 0.05, 0.10          # validity and abstention budgets (alpha+beta < 1/4 so the BH constant is non-vacuous)
APB = ALPHA + BETA


def _load():
    return json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}


def _save(d):
    json.dump(d, open(OUTJ, "w"), indent=2)


# ---------------------------------------------------------------- EB (vectorized)
def eb_radius(X, alpha):
    """Maurer-Pontil empirical-Bernstein deviation radius, rows = trials."""
    n = X.shape[1]
    m = X.mean(1); v = X.var(1, ddof=1)
    rad = np.sqrt(2 * v * np.log(2 / alpha) / n) + 7 * (X.max(1) - X.min(1)) * np.log(2 / alpha) / (3 * (n - 1))
    return m, rad


def part_crosscheck():
    """Vectorized EB == vendored certificate formula (same constants)."""
    from certification.switching_certificate import empirical_bernstein_lcb
    ok, worst = True, 0.0
    for _ in range(100):
        n = int(rng.integers(20, 400))
        x = rng.normal(rng.uniform(-.2, .2), rng.uniform(.05, .5), n)
        mean_v, lcb_v, _ = empirical_bernstein_lcb(x, alpha=ALPHA, benefit_range=float(x.max() - x.min()))
        m, r = eb_radius(x[None, :], ALPHA)
        gap = abs((m[0] - r[0]) - lcb_v)
        worst = max(worst, gap); ok &= gap < 1e-9
    d = _load(); d["eb_crosscheck"] = dict(matches_vendored=bool(ok), worst_abs_gap=float(worst)); _save(d)
    print("[xcheck] vectorized EB == vendored:", ok, "worst gap", worst)


# ---------------------------------------------------------------- regimes
def draw(regime, n, trials, delta, R=2.0, sigma=0.3):
    """Benefit samples with mean +delta. dense: clipped normal sd sigma.
    spike: mass at -s plus rare +R/2 spikes (Bernstein regime, tiny variance)."""
    if regime == "dense":
        X = rng.normal(delta, sigma, (trials, n))
        return np.clip(X, -R / 2, R / 2)
    p = np.clip(delta / (R / 2), 0, 1)                       # mean = delta; variance ~ delta*R/2 (no floor -> true Bernstein regime)
    spike = rng.random((trials, n)) < p
    return np.where(spike, R / 2, 0.0)


def min_certifiable_margin(regime, n, lo=1e-4, hi=1.0, trials=400):
    """Smallest delta where the EB certificate commits-adapt w.p. >= 1-BETA
    (wrong-commit checked at -delta too)."""
    def commit_rate(delta, sign=+1):
        X = sign * draw(regime, n, trials, delta)
        m, r = eb_radius(X, ALPHA)
        return float(np.mean(m - r > 0)) if sign > 0 else float(np.mean(m + r < 0))
    # bisect on coverage at +delta (validity at -delta holds by EB symmetry; spot-check)
    for _ in range(22):
        mid = math.sqrt(lo * hi)
        ok = commit_rate(mid) >= 1 - BETA
        hi, lo = (mid, lo) if ok else (hi, mid)
    # wrong-commit spot check at the found margin
    Xm = draw(regime, n, trials, hi)
    m, r = eb_radius(-Xm, ALPHA)                # adverse world: mean -delta
    fa = float(np.mean(m - r > 0))
    return hi, fa


def lower_bound(n, R=2.0, sigma=0.3):
    dense = (sigma / 2) * math.sqrt(math.log(1 / (4 * APB)) / n)
    spike = (R / 4) * math.log(1 / (2 * APB)) / n
    return dense, spike


def eb_theory_ub(n, R=2.0, sigma=0.3, spike=False, kap=None):
    """Exact EB radius with population quantities (what Theorem R1 predicts)."""
    L = math.log(2 / ALPHA)
    var = (kap * R / 2) if spike and kap else sigma ** 2     # spike variance ~ p*(R/2)^2 ~ kap*R/2
    rng_eff = R / 2 if spike else 2 * sigma * math.sqrt(2 * math.log(n))
    return math.sqrt(2 * var * L / n) + 7 * rng_eff * L / (3 * (n - 1))


def part_rates():
    ns = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    out = {"alpha": ALPHA, "beta": BETA, "ns": ns, "dense": [], "spike": []}
    for regime in ("dense", "spike"):
        for n in ns:
            k, fa = min_certifiable_margin(regime, n)
            lb_d, lb_s = lower_bound(n)
            lb = lb_d if regime == "dense" else lb_s
            ub = eb_theory_ub(n, spike=(regime == "spike"), kap=k)
            out[regime].append(dict(n=n, kappa_hat=k, false_commit_at_margin=fa,
                                    lower_bound=lb, ratio_to_LB=k / lb, ratio_to_EBtheory=k / ub))
    # tail slopes (additive two-term rate -> fit on the asymptotic tail, last 4 points)
    for regime, want, tol in (("dense", -0.5, 0.15), ("spike", -1.0, 0.20)):
        K = np.log([r["kappa_hat"] for r in out[regime]][-4:]); N = np.log(ns[-4:])
        slope = float(np.polyfit(N, K, 1)[0])
        out[f"{regime}_tail_slope"] = slope
        out[f"{regime}_slope_matches"] = bool(abs(slope - want) < tol)
        # rate match = ratio to LB is CONSTANT in n (constants differ, n-dependence must not)
        rat = [r["ratio_to_LB"] for r in out[regime]][-4:]
        out[f"{regime}_ratio_constancy"] = float(max(rat) / min(rat))
        out[f"{regime}_rate_matches_LB"] = bool(max(rat) / min(rat) < 3.0)
    out["kappa_tracks_EB_theory"] = bool(all(0.5 < r["ratio_to_EBtheory"] < 4.0
                                             for reg in ("dense", "spike") for r in out[reg]))
    out["validity_holds"] = bool(max(r["false_commit_at_margin"] for reg in ("dense", "spike")
                                     for r in out[reg]) <= ALPHA + 0.03)
    out["constant_factor_LB_note"] = "EB constants are a bounded universal factor above the two-point optimum (reported per-n); the RATE (n-dependence) matches."
    d = _load(); d["rates"] = out; _save(d)
    print("[rates] dense tail slope", round(out["dense_tail_slope"], 3), "(want -0.5) ok:", out["dense_slope_matches"],
          "| ratio constancy x", round(out["dense_ratio_constancy"], 2), "ok:", out["dense_rate_matches_LB"])
    print("[rates] spike tail slope", round(out["spike_tail_slope"], 3), "(want -1.0) ok:", out["spike_slope_matches"],
          "| ratio constancy x", round(out["spike_ratio_constancy"], 2), "ok:", out["spike_rate_matches_LB"])
    print("[rates] kappa tracks exact EB theory:", out["kappa_tracks_EB_theory"],
          "| validity at margin:", out["validity_holds"])


def part_infeasible():
    """Below the lower bound NO rule is feasible: exact optimal error-sum of the
    two-point pair already exceeds the budget. Dense: Bernoulli pair via
    Bretagnolle-Huber e^{-KL}; spike: exact (1-p)^n no-spike probability."""
    rows = []
    for n in [100, 400, 1600, 6400]:
        lb_d, lb_s = lower_bound(n)
        # dense pair at kappa = lb/2: X in {-1,+1} w.p. q± = (1 ± 2*kappa)/2 -> sd~1? scale to sigma:
        sig, R = 0.3, 2.0
        kap = lb_d / 2
        q1, q2 = 0.5 + kap / (2 * sig), 0.5 - kap / (2 * sig)  # X=±sig valued: mean ±kap, sd ~ sig
        kl = n * (q1 * math.log(q1 / q2) + (1 - q1) * math.log((1 - q1) / (1 - q2)))
        errsum_dense = 0.5 * math.exp(-kl)                    # Bretagnolle-Huber
        # spike pair at kappa = lb_s/2: world-: const; world+: spike prob p
        kapS = lb_s / 2; p = 2 * kapS / R
        errsum_spike = (1 - p) ** n                           # P(no spike) = indistinguishable
        rows.append(dict(n=n,
                         dense=dict(kappa=kap, optimal_errsum=errsum_dense,
                                    infeasible=bool(errsum_dense > 2 * APB * 0.999)),
                         spike=dict(kappa=kapS, optimal_errsum=errsum_spike,
                                    infeasible=bool(errsum_spike > 2 * APB * 0.999))))
    d = _load()
    d["infeasibility_below_bound"] = dict(
        rows=rows, all_infeasible=bool(all(r["dense"]["infeasible"] and r["spike"]["infeasible"] for r in rows)),
        note="optimal-test error sum exceeds budget at kappa = LB/2 -> no rule (certificate or not) can be (alpha,beta)-reliable there")
    _save(d)
    print("[lower] infeasible below bound at all n:", d["infeasibility_below_bound"]["all_infeasible"])


def part_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = _load(); r = d["rates"]; ns = r["ns"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for ax, regime, slope_want in ((axes[0], "dense", -0.5), (axes[1], "spike", -1.0)):
        K = [x["kappa_hat"] for x in r[regime]]; LB = [x["lower_bound"] for x in r[regime]]
        ax.loglog(ns, K, "-o", color="#2a9d8f", label=r"certificate $\hat\kappa(n)$ (achieved)")
        ax.loglog(ns, LB, "--", color="#e76f51", label="lower bound (no rule below)")
        ax.set_xlabel("n (calibrated benefit samples)")
        ax.set_title(f"{regime} regime: tail slope {r[f'{regime}_tail_slope']:.2f} (theory {slope_want})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"certifiable margin $\kappa$")
    plt.suptitle("Knowability rate: the deployed certificate matches the two-point lower bound", y=1.02)
    plt.tight_layout()
    for p in (os.path.join(FIGD, "fig_krates.png"), os.path.join(FIGF, "fig_krates.png")):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    print("figure written: fig_krates.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["xcheck", "rates", "lower", "figure", "all"])
    a = ap.parse_args()
    if a.part in ("xcheck", "all"): part_crosscheck()
    if a.part in ("rates", "all"): part_rates()
    if a.part in ("lower", "all"): part_infeasible()
    if a.part in ("figure", "all"): part_figure()
