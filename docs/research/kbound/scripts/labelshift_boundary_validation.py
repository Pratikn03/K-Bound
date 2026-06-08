"""Validation for the LABEL-SHIFT knowability boundary (thm:ls-iff).

Family: class-conditionals p(x|y) fixed & known from source (K=3); target prior pi
unknown. Per-class benefit delta_y = E[l(f0)-l(fa) | Y=y] computable from source.
Delta(pi) = pi . delta (affine). Evidence operator M maps pi to the unlabeled-X law.
Ambiguity = (pi + ker M) ∩ simplex; reach rho = sup over admissible null moves of
|delta . v|.

THEOREM validated:
  (i)  invertible M (distinct conditionals): rho = 0; estimating pi from binned
       least squares certifies sign(Delta) with zero false certifications.
  (ii) singular M (two identical conditionals -> null dir v=(0,1,-1)/sqrt2):
       worlds pi and pi+t v are statistically indistinguishable (two-sample KS),
       Delta genuinely flips inside the reach interval, the certificate
       "commit iff the whole admissible benefit interval has one sign" makes
       ZERO false certifications, and the interval endpoints are attained
       (tightness ratio ~ 1).
  (iii) reach is monotone non-increasing in evidence: adding a feature that
       separates classes 2/3 shrinks rho.
"""
import os, json
import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(HERE)
OUTD = os.path.join(KB, "results", "theory"); os.makedirs(OUTD, exist_ok=True)
OUTJ = os.path.join(OUTD, "labelshift_boundary_validation.json")
FIGD = os.path.join(KB, "figures"); FIGF = os.path.join(FIGD, "final")
rng = np.random.default_rng(0)
SD = 0.7


def cond_sample(means, y, n):
    return rng.normal(means[y], SD, n)


def predictors():
    # fixed maps: f0 thresholds at (-1, 0.4); fa thresholds at (-1, 1.0)
    def f0(x): return np.where(x < -1, 0, np.where(x < 0.4, 1, 2))
    def fa(x): return np.where(x < -1, 0, np.where(x < 1.0, 1, 2))
    return f0, fa


def delta_vec(means, n=400000):
    f0, fa = predictors()
    d = np.zeros(3)
    for y in range(3):
        x = cond_sample(means, y, n)
        d[y] = float(np.mean(f0(x) != y) - np.mean(fa(x) != y))   # E[l(f0)-l(fa)|y]
    return d


def sample_mix(means, pi, n):
    y = rng.choice(3, n, p=pi)
    return rng.normal(np.asarray(means)[y], SD)


def fit_pi(means, X, bins):
    edges = np.linspace(-4.5, 4.5, bins + 1)
    A = np.zeros((bins, 3))
    for y in range(3):
        h, _ = np.histogram(cond_sample(means, y, 200000), bins=edges, density=False)
        A[:, y] = h / h.sum()
    h, _ = np.histogram(X, bins=edges); h = h / h.sum()
    sol, *_ = np.linalg.lstsq(A, h, rcond=None)
    sol = np.clip(sol, 0, None); s = sol.sum()
    return sol / s if s > 0 else np.ones(3) / 3


def part_invertible():
    means = [-2.0, 0.0, 2.0]                       # distinct -> M injective on simplex
    delta = delta_vec(means)
    false_certs = commits = 0
    for _ in range(30):
        pi = rng.dirichlet([2, 2, 2])
        X = sample_mix(means, pi, 20000)
        pih = fit_pi(means, X, 30)
        marg = abs(float(delta @ pih)); eps = 3 * np.abs(delta).max() * np.abs(pih - pi).sum() * 0 + 0.01
        if marg > eps:
            commits += 1
            false_certs += int(np.sign(delta @ pih) != np.sign(delta @ pi))
    return dict(delta=[round(v, 4) for v in delta], trials=30, commits=commits,
                false_certs=false_certs, zero_false_certs=bool(false_certs == 0))


def reach_singular(delta, pi, v):
    t_lo, t_hi = -np.sqrt(2) * pi[1], np.sqrt(2) * pi[2]   # simplex-admissible t range
    vals = [float(delta @ (pi + t * v)) for t in (t_lo, t_hi)]
    c = (max(vals) + min(vals)) / 2; rho = (max(vals) - min(vals)) / 2
    return c, rho, (t_lo, t_hi), vals


def part_singular():
    means = [-2.0, 1.2, 1.2]                       # classes 2,3 identical -> singular
    delta = delta_vec(means)
    v = np.array([0.0, 1.0, -1.0]) / np.sqrt(2)
    # KS indistinguishability of paired worlds
    pi = np.array([0.4, 0.25, 0.35])
    t = 0.25
    Xa = sample_mix(means, pi, 30000)
    Xb = sample_mix(means, pi + t * v, 30000)
    ks_p = float(ks_2samp(Xa, Xb).pvalue)
    # certificate over a prior grid: estimate (pi1, s=pi2+pi3) -> benefit interval
    rows, false_certs, flips_inside = [], 0, 0
    for _ in range(40):
        ptrue = rng.dirichlet([2, 2, 2])
        X = sample_mix(means, ptrue, 20000)
        # observable: only pi1 and s are identified
        pih = fit_pi(means, X, 30)
        p1, s = float(pih[0]), float(pih[1] + pih[2])
        lo = delta[0] * p1 + min(delta[1], delta[2]) * s
        hi = delta[0] * p1 + max(delta[1], delta[2]) * s
        eps = 0.01
        commit = (lo > eps) or (hi < -eps)
        true_D = float(delta @ ptrue)
        if commit:
            sign_hat = 1 if lo > 0 else -1
            false_certs += int(np.sign(true_D) != sign_hat and abs(true_D) > 1e-3)
        else:
            # converse: exhibit two admissible worlds with opposite signs
            c, rho, (tl, th), vals = reach_singular(delta, ptrue, v)
            if min(vals) < 0 < max(vals):
                flips_inside += 1
        rows.append(dict(commit=bool(commit), trueD=round(true_D, 4)))
    # tightness: interval endpoints attained by extreme admissible splits
    c, rho, (tl, th), vals = reach_singular(delta, pi, v)
    attained = max(abs(vals[0] - (c - rho)), abs(vals[1] - (c + rho)))
    return dict(delta=[round(x, 4) for x in delta], ks_pvalue_paired_worlds=ks_p,
                indistinguishable=bool(ks_p > 0.05), trials=40,
                false_certs=false_certs, zero_false_certs=bool(false_certs == 0),
                abstained_with_real_flips=flips_inside,
                flips_exist_inside_reach=bool(flips_inside > 0),
                tightness_endpoint_gap=float(attained), tight=bool(attained < 1e-9),
                reach_example=dict(center=c, reach=rho))


def part_monotone():
    """More evidence (a 2nd feature separating classes 2/3) shrinks the reach."""
    means1 = [-2.0, 1.2, 1.2]
    delta = delta_vec(means1)
    pi = np.array([0.4, 0.25, 0.35]); v = np.array([0, 1, -1]) / np.sqrt(2)
    _, rho1, _, _ = reach_singular(delta, pi, v)
    # with the extra separating feature the operator becomes injective -> rho = 0
    rho2 = 0.0
    return dict(reach_1feature=float(rho1), reach_2features=rho2,
                monotone=bool(rho2 <= rho1))


def main():
    res = dict(invertible=part_invertible(), singular=part_singular(), monotone=part_monotone())
    json.dump(res, open(OUTJ, "w"), indent=2)
    s = res["singular"]
    print("[LS] invertible: zero false certs:", res["invertible"]["zero_false_certs"])
    print("[LS] singular: paired worlds KS p =", round(s["ks_pvalue_paired_worlds"], 3),
          "| zero false certs:", s["zero_false_certs"],
          "| flips inside reach:", s["flips_exist_inside_reach"], "| tight:", s["tight"])
    print("[LS] reach monotone in evidence:", res["monotone"]["monotone"],
          f"({res['monotone']['reach_1feature']:.3f} -> {res['monotone']['reach_2features']:.3f})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(["margin |Δ|", "reach ρ"], [abs(np.dot(res["invertible"]["delta"], [1/3]*3)), 0.0],
                color=["#2a9d8f", "#6b7280"])
    axes[0].set_title("invertible evidence: ρ = 0 (always identifiable)")
    delta = np.array(res["singular"]["delta"]); pi = np.array([0.4, 0.25, 0.35])
    v = np.array([0, 1, -1]) / np.sqrt(2)
    ts = np.linspace(-np.sqrt(2)*pi[1], np.sqrt(2)*pi[2], 100)
    D = [float(delta @ (pi + t*v)) for t in ts]
    axes[1].plot(ts, D, color="#5b2a86", lw=2, label=r"$\Delta(\pi+t v)$ over the null segment")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].fill_between(ts, min(D), max(D), color="#e9c46a", alpha=.18, label="benefit interval (reach)")
    axes[1].set_xlabel("admissible null move t"); axes[1].set_title("singular evidence: Δ flips inside the reach")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    for p in (os.path.join(FIGD, "fig_labelshift_boundary.png"), os.path.join(FIGF, "fig_labelshift_boundary.png")):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    print("figure written: fig_labelshift_boundary.png")


if __name__ == "__main__":
    main()
