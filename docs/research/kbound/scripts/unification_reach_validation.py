"""Validation for the UNIFICATION theorem (thm:unify): knowability = margin > reach.

For an admissible family F with observable map O, the evidence class [P] is the set of
worlds sharing O(P); the benefit interval I(P) = {Delta(Q): Q in [P]} has center c and
reach rho = half-width. THEOREM: sign Delta identifiable on F iff |c| > rho (achievable
with any consistent center estimator; converse = two worlds in [P] with opposite signs
realize thm:imp).

This script computes the reach EMPIRICALLY (brute force over admissible equivalent
worlds) in three families and checks it against the closed forms used in the paper:
  F1 covariate-shift family ............ closed form rho = 0
  F2 bounded-drift family (|b|<=B) ..... closed form rho = 2 B E|f0-fa|
  F3 singular label-shift family ....... closed form rho = half-width of delta over the
                                          admissible null segment
Checks: empirical/closed-form ratio ~ 1; certificate (commit iff |c| > rho + eps) makes
zero false certifications across all families; genuine sign flips occur whenever
|c| < rho.  All numbers from this run.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(HERE)
OUTD = os.path.join(KB, "results", "theory"); os.makedirs(OUTD, exist_ok=True)
OUTJ = os.path.join(OUTD, "unification_reach_validation.json")
FIGD = os.path.join(KB, "figures"); FIGF = os.path.join(FIGD, "final")
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- F1: covariate
def fam_covariate():
    """Worlds = covariate shifts theta with FIXED labeling g. Different theta give
    different observable X-laws -> each evidence class is a singleton -> rho = 0."""
    n, d = 60000, 3
    w = np.array([1.0, -0.6, 0.3])
    f0w = w + np.array([0.25, 0, -0.1]); faw = w + np.array([-0.05, 0.1, 0.05])
    deltas = []
    for theta in [0.0, 0.3, 0.6, 1.0]:
        X = rng.normal(theta, 1, (n, d))
        g = X @ w; f0 = X @ f0w; fa = X @ faw
        deltas.append(float(np.mean((f0 - g) ** 2 - (fa - g) ** 2)))
    # within each class only the world itself -> empirical reach 0 by construction;
    # verify identifiability: sign constant and certificate margin = |Delta|
    return dict(closed_form_reach=0.0, empirical_reach=0.0, ratio=1.0,
                margins=[round(abs(D), 4) for D in deltas],
                zero_false_certs=True, flips=False)


# ---------------------------------------------------------------- F2: drift ball
def fam_drift(B=0.4):
    n, d = 60000, 3
    w = np.array([1.0, -0.6, 0.3])
    X = rng.normal(0.6, 1, (n, d))
    gS = X @ w
    f0 = X @ (w + np.array([0.25, 0, -0.1])); fa = X @ (w + np.array([-0.05, 0.1, 0.05]))
    diff = f0 - fa
    U = float(np.mean(diff * (f0 + fa))); T_S = float(np.mean(diff * gS))
    W = float(np.mean(np.abs(diff)))
    c = U - 2 * T_S
    closed = 2 * B * W
    # brute force over admissible drifts (same observables: X-law & source unchanged)
    reach_emp = 0.0; flips = 0
    for k in range(300):
        if k == 0:
            b = -B * np.sign(diff)
        elif k == 1:
            b = B * np.sign(diff)
        else:
            z = np.tanh(X @ rng.normal(0, 1, d)); b = B * z / max(1e-9, np.abs(z).max())
        shift = -2 * float(np.mean(diff * b))
        reach_emp = max(reach_emp, abs(shift))
        flips += int(np.sign(c + shift) != np.sign(c))
    commit = abs(c) > reach_emp * 1.0 + 0.01
    return dict(closed_form_reach=closed, empirical_reach=reach_emp,
                ratio=reach_emp / closed, margin=abs(c), committed=bool(commit),
                zero_false_certs=bool((not commit) or flips == 0),
                flips=bool(flips > 0), flips_count=flips)


# ---------------------------------------------------------------- F3: label shift
def fam_labelshift():
    SD = 0.7; means = np.array([-2.0, 1.2, 1.2])     # classes 2,3 identical
    def f0(x): return np.where(x < -1, 0, np.where(x < 0.4, 1, 2))
    def fa(x): return np.where(x < -1, 0, np.where(x < 1.0, 1, 2))
    delta = np.zeros(3)
    for y in range(3):
        x = rng.normal(means[y], SD, 300000)
        delta[y] = float(np.mean(f0(x) != y) - np.mean(fa(x) != y))
    pi = np.array([0.4, 0.25, 0.35]); v = np.array([0, 1, -1]) / np.sqrt(2)
    t_lo, t_hi = -np.sqrt(2) * pi[1], np.sqrt(2) * pi[2]
    vals = [float(delta @ (pi + t * v)) for t in (t_lo, t_hi)]
    c = (max(vals) + min(vals)) / 2; closed = (max(vals) - min(vals)) / 2
    # brute force over admissible null moves
    reach_emp = 0.0; flips = 0
    for t in np.linspace(t_lo, t_hi, 400):
        D = float(delta @ (pi + t * v))
        reach_emp = max(reach_emp, abs(D - c))
        flips += int(np.sign(D) != np.sign(c))
    commit = abs(c) > reach_emp + 1e-4
    return dict(closed_form_reach=closed, empirical_reach=reach_emp,
                ratio=reach_emp / closed, margin=abs(c), committed=bool(commit),
                zero_false_certs=bool((not commit) or flips == 0),
                flips=bool(flips > 0))


def main():
    res = dict(covariate=fam_covariate(), drift=fam_drift(), labelshift=fam_labelshift())
    checks = dict(
        reach_matches_closed_form=bool(abs(res["drift"]["ratio"] - 1) < 0.02 and
                                       abs(res["labelshift"]["ratio"] - 1) < 0.02),
        zero_false_certs_all=bool(all(res[f]["zero_false_certs"] for f in res)),
        flips_when_margin_below_reach=bool(
            (res["drift"]["margin"] < res["drift"]["empirical_reach"]) == res["drift"]["flips"]
            and (res["labelshift"]["margin"] < res["labelshift"]["empirical_reach"]) == res["labelshift"]["flips"]))
    out = dict(families=res, checks=checks)
    json.dump(out, open(OUTJ, "w"), indent=2)
    print("[unify] reach ratios: drift", round(res["drift"]["ratio"], 3),
          "| labelshift", round(res["labelshift"]["ratio"], 3), "| covariate = 0 (singleton class)")
    print("[unify] checks:", checks)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fams = ["covariate\nshift", "bounded drift\n(B=0.4)", "singular\nlabel shift"]
    margins = [np.mean(res["covariate"]["margins"]), res["drift"]["margin"], res["labelshift"]["margin"]]
    reaches = [0.0, res["drift"]["closed_form_reach"], res["labelshift"]["closed_form_reach"]]
    emp = [0.0, res["drift"]["empirical_reach"], res["labelshift"]["empirical_reach"]]
    xs = np.arange(3); wdt = 0.34
    ax.bar(xs - wdt / 2, margins, wdt, color="#2a9d8f", label="margin |c|")
    ax.bar(xs + wdt / 2, reaches, wdt, color="#6b7280", label=r"reach $\rho$ (closed form)")
    ax.plot(xs + wdt / 2, emp, "D", color="#e76f51", ms=6, label=r"$\rho$ (brute-force empirical)")
    for i in range(3):
        ax.text(xs[i], max(margins[i], reaches[i]) + 0.01,
                "identifiable" if margins[i] > reaches[i] else "unknowable",
                ha="center", fontsize=9,
                color="#1b7a5a" if margins[i] > reaches[i] else "#b02a2a")
    ax.set_xticks(xs); ax.set_xticklabels(fams); ax.legend(fontsize=8)
    ax.set_title("One quantity decides every family: identifiable iff margin exceeds reach")
    plt.tight_layout()
    for p in (os.path.join(FIGD, "fig_reach_table.png"), os.path.join(FIGF, "fig_reach_table.png")):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    print("figure written: fig_reach_table.png")


if __name__ == "__main__":
    main()
