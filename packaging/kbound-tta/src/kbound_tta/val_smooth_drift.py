"""Smooth source->target drift bracket -- numerical validation (Part 1B).

Refines Proposition ``thm:reg-iff`` (the bounded-drift knowability boundary,
docs/research/kbound/paper/sections/regression_conjecture.tex).  Does NOT modify
the paper or any SAR/GPU harness.  CPU only.

==============================================================================
SETTING (from thm:reg-noise / thm:reg-iff)
==============================================================================
Squared loss, Y = g(X) + eps with E[eps|X]=0 (noise-invariant decision,
Prop thm:reg-noise).  Fixed predictors f0, fa.  On the target,
    Delta = R_T(f0) - R_T(fa) = E_T[(f0-fa)(f0+fa-2 g_T)]
                              = U - 2 T_S - 2 E_T[(f0-fa)(g_T - g_S)],
    U   = E_T[(f0-fa)(f0+fa)]          (OBSERVABLE: unlabeled target X)
    T_S = E_T[(f0-fa) g_S]             (importance-weighted from LABELED source)
    W   = E_T|f0-fa|                   (OBSERVABLE).
Prop thm:reg-iff bounds the concept-drift term by Hoelder: with b = g_T - g_S
and a *known* radius ||b||_inf <= B,  |E_T[(f0-fa) b]| <= B W,  saturated by the
adversary b* = -B sign(f0-fa).  Boundary: commit sign(U-2T_S) iff |U-2T_S|>2BW.

==============================================================================
THE 1B REFINEMENT (make the drift radius OBSERVABLE)
==============================================================================
thm:reg-iff leaves B as an assumed free parameter.  We replace "known B" by a
DRIFT-SMOOTHNESS coupling: the labeling function does not change faster than the
inputs move,
        ||g_T - g_S||_{inf, D}  <=  L * d(P_S, P_T)                       (DS)
for a known modulus L and an OBSERVABLE covariate discrepancy d (here the
Gaussian W2 distance between source/target X, estimable from unlabeled data).
Then B := L*d is observable, and the boundary becomes FULLY computable:

    center  c    = U - 2 T_S            (observable)
    reach   rho  = 2 * (L*d) * W        (observable given L)
    COMMIT sign(c) iff |c| > rho + 2*eps_n ;  else ABSTAIN.

This slots into the reach table (thm:unify): covariate movement is priced as an
observable reach L*d*W instead of an a-priori BW.  (Online view: over a stream
P_0 -> ... -> P_K the reach accumulates as 2 L W * sum_k d(P_{k-1},P_k); abstain
once the budget is spent.  Validated here in the static two-point case.)

HONEST SCOPE.  The core Hoelder inequality and the center/U/T_S/W objects are
thm:reg-iff's; 1B's contribution is (i) making B observable via (DS), (ii) the
path/online accumulation remark, (iii) showing the boundary is necessary (over-
budget drifts flip the sign).  Incremental, not a new hard theorem -- flagged.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_smooth_drift.json")
FIG1 = os.path.join(HERE, "fig_smooth_drift_boundary.png")
FIG2 = os.path.join(HERE, "fig_smooth_drift_noise_and_baseline.png")
SEED = 20260609
RNG = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# model: scalar X, affine predictors, linear source concept g_S(x)=ws*x
# --------------------------------------------------------------------------- #
F0 = (1.0, 0.0)      # f0(x) = 1.0*x + 0.0
FA = (0.4, 0.3)      # fa(x) = 0.4*x + 0.3
WS = 0.8             # g_S(x) = 0.8 x


def f0(x): return F0[0] * x + F0[1]
def fa(x): return FA[0] * x + FA[1]
def gS(x): return WS * x
def diff(x): return f0(x) - fa(x)
def summ(x): return f0(x) + fa(x)


def w2_gaussian(mu_s, sd_s, mu_t, sd_t):
    """Observable covariate discrepancy: W2 between 1-D Gaussians."""
    return float(np.sqrt((mu_t - mu_s) ** 2 + (sd_t - sd_s) ** 2))


def make_target_X(mu_t, sd_t, n, rng):
    return mu_t + sd_t * rng.standard_normal(n)


def concept_drift(x, kind, B):
    """Concept drift b(x)=g_T-g_S with ||b||_inf <= B (within budget), or a
    deliberate OVER-budget variant for the converse."""
    d = diff(x)
    if kind == "adversarial":          # saturates the Hoelder bound
        return -B * np.sign(d)
    if kind == "aligned":              # helps; same magnitude, opposite sign
        return +B * np.sign(d)
    if kind == "random":               # random within budget
        return RNG.uniform(-B, B, size=x.shape)
    if kind == "smooth":               # smooth, within budget (Lipschitz bump)
        return B * np.tanh(0.5 * x)
    if kind == "over_budget":          # violates (DS): 1.8x the budget, adversarial
        return -1.8 * B * np.sign(d)
    return np.zeros_like(x)


# --------------------------------------------------------------------------- #
# estimators (all label-free on target; source used with labels)
# --------------------------------------------------------------------------- #
def estimate_TS(xt, n_src, mu_s, sd_s, noise_sd, rng):
    """T_S = E_T[(f0-fa) g_S], g_S fit by OLS-through-0 on a labeled SOURCE
    sample (no target labels).  Returns (T_S_hat, w_hat, SE_w) where SE_w is the
    standard error of the slope (drives the data-driven center radius eps_n)."""
    xs = mu_s + sd_s * rng.standard_normal(n_src)
    ys = gS(xs) + noise_sd * rng.standard_normal(n_src)      # source labels
    sxx = float(np.sum(xs * xs))
    w_hat = float(np.sum(xs * ys) / sxx)                     # OLS through 0
    sigma_hat = float(np.std(ys - w_hat * xs))               # residual scale
    se_w = sigma_hat / np.sqrt(sxx)                          # SE of slope
    return float(np.mean(diff(xt) * (w_hat * xt))), w_hat, se_w


def center_radius(xt, se_w, z=3.0):
    """Data-driven confidence radius of the center c=U-2T_S.  Since
    T_S_hat = w_hat * A with A=E_T[(f0-fa)X] observable, the center error is
    2|A|*|w_hat-w|; a z-sigma radius is eps_n = 2 z |A| SE_w."""
    A = float(np.mean(diff(xt) * xt))
    return 2.0 * z * abs(A) * se_w


def observables(xt):
    U = float(np.mean(diff(xt) * summ(xt)))
    W = float(np.mean(np.abs(diff(xt))))
    return U, W


def true_delta(xt, b_vals):
    """Ground-truth target benefit (uses the true target concept g_T=g_S+b)."""
    gT = gS(xt) + b_vals
    return float(np.mean(diff(xt) * (summ(xt) - 2.0 * gT)))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def exp_noise_invariance(n=120000, n_trials=8):
    """REG-1 analog: Delta is invariant to unknown label-noise scale."""
    xt = make_target_X(0.6, 1.1, n, RNG)
    b = concept_drift(xt, "random", 0.2)
    base = true_delta(xt, b)
    rows = []
    for sd in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        # absolute risks shift by E[eps^2]=sd^2; the DIFFERENCE Delta is unchanged
        ds = []
        for _ in range(n_trials):
            eps = sd * RNG.standard_normal(n)
            gT = gS(xt) + b
            r0 = np.mean((f0(xt) - gT - eps) ** 2)
            ra = np.mean((fa(xt) - gT - eps) ** 2)
            ds.append(float(r0 - ra))
        rows.append({"noise_sd": sd, "mean_Delta": float(np.mean(ds)),
                     "abs_risk_f0": float(np.mean((f0(xt) - gS(xt) - b) ** 2) + sd ** 2)})
    return {"Delta_ref": base, "rows": rows,
            "max_abs_dev_from_ref": float(max(abs(r["mean_Delta"] - base) for r in rows))}


def exp_boundary_and_coverage(L=0.6, n=30000, n_src=20000, noise_sd=0.7,
                              n_trials=120):
    """Sweep covariate shift (hence the observable reach 2 L d W); for each, test
    bracket coverage + false-commit over many within-budget drifts (incl.
    adversarial), and the converse with over-budget drifts."""
    shifts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8]
    within_kinds = ["zero", "random", "smooth", "aligned", "adversarial"]
    rows = []
    for mu_t in shifts:
        sd_t = 1.0 + 0.3 * mu_t
        d_obs = w2_gaussian(0.0, 1.0, mu_t, sd_t)
        B = L * d_obs
        cov_hits = cov_tot = 0
        commit = false_commit = cells = 0
        over_flips = over_tot = 0
        c_vals, reach_vals = [], []
        for _ in range(n_trials):
            xt = make_target_X(mu_t, sd_t, n, RNG)
            U, W = observables(xt)
            TS, _, se_w = estimate_TS(xt, n_src, 0.0, 1.0, noise_sd, RNG)
            c = U - 2.0 * TS
            reach = 2.0 * B * W
            eps_n = center_radius(xt, se_w)       # data-driven 3-sigma center CI
            c_vals.append(c); reach_vals.append(reach)
            # within-budget drifts: bracket must cover, commits must be correct
            for kind in within_kinds:
                b = concept_drift(xt, kind, B)
                D = true_delta(xt, b)
                cov_tot += 1
                if (c - reach - eps_n) <= D <= (c + reach + eps_n):
                    cov_hits += 1
                cells += 1
                if abs(c) > reach + eps_n:
                    commit += 1
                    if np.sign(c) != np.sign(D) and np.sign(D) != 0:
                        false_commit += 1
            # over-budget drift: smoothness (DS) violated -> sign may flip
            b_ob = concept_drift(xt, "over_budget", B)
            D_ob = true_delta(xt, b_ob)
            over_tot += 1
            if abs(c) > reach + eps_n and np.sign(c) != np.sign(D_ob) \
               and np.sign(D_ob) != 0:
                over_flips += 1
        rows.append({
            "mu_shift": mu_t, "d_obs": d_obs, "B_eff": B,
            "mean_center_c": float(np.mean(c_vals)),
            "mean_reach": float(np.mean(reach_vals)),
            "coverage_within_budget": cov_hits / cov_tot,
            "commit_rate": commit / cells,
            "false_commit_rate": false_commit / cells,
            "over_budget_flip_rate": over_flips / over_tot,
        })
    return rows


def exp_baseline_vs_guard(L=0.6, n=30000, n_src=20000, noise_sd=0.7, n_trials=200):
    """As concept drift grows, a no-drift-correction rule (commit sign(U-2T_S)
    ALWAYS) accrues wrong commits; the guarded rule abstains and stays safe."""
    mu_t, sd_t = 0.9, 1.27
    d_obs = w2_gaussian(0.0, 1.0, mu_t, sd_t)
    rows = []
    for scale in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:    # drift as multiple of budget
        B = L * d_obs
        guard_commit = guard_false = base_false = cells = 0
        for _ in range(n_trials):
            xt = make_target_X(mu_t, sd_t, n, RNG)
            U, W = observables(xt)
            TS, _, se_w = estimate_TS(xt, n_src, 0.0, 1.0, noise_sd, RNG)
            c = U - 2.0 * TS
            reach = 2.0 * B * W
            eps_n = center_radius(xt, se_w)
            b = concept_drift(xt, "adversarial", scale * B)   # may exceed budget
            D = true_delta(xt, b)
            cells += 1
            # baseline: always commit sign(c)
            if np.sign(c) != np.sign(D) and np.sign(D) != 0:
                base_false += 1
            # guarded: commit only if |c|>reach
            if abs(c) > reach + eps_n:
                guard_commit += 1
                if np.sign(c) != np.sign(D) and np.sign(D) != 0:
                    guard_false += 1
        rows.append({
            "drift_x_budget": scale,
            "baseline_false_commit_rate": base_false / cells,
            "guard_commit_rate": guard_commit / cells,
            "guard_false_commit_rate": guard_false / cells,
        })
    return rows


# --------------------------------------------------------------------------- #
def plot_boundary(rows, path):
    mu = [r["mu_shift"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(mu, [r["mean_center_c"] for r in rows], "o-", label="center |c|=|U-2T_S|")
    ax[0].plot(mu, [r["mean_reach"] for r in rows], "s--", label="reach 2 L d W")
    ax[0].set(title="observable boundary: commit iff |c|>reach",
              xlabel="covariate shift mu", ylabel="value"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(mu, [r["coverage_within_budget"] for r in rows], "o-", color="C2")
    ax[1].axhline(1, ls=":", c="gray")
    ax[1].set(title="bracket coverage of true Delta\n(within-budget drifts)",
              xlabel="covariate shift mu", ylabel="coverage"); ax[1].grid(alpha=.3)
    ax[2].plot(mu, [r["false_commit_rate"] for r in rows], "^-", color="red",
               label="FALSE commit (within budget)")
    ax[2].plot(mu, [r["over_budget_flip_rate"] for r in rows], "x--", color="purple",
               label="sign flip (OVER budget = (DS) violated)")
    ax[2].set(title="safety + necessity of (DS)", xlabel="covariate shift mu",
              ylabel="rate"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_noise_baseline(noise, base, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    nr = noise["rows"]
    ax[0].plot([r["noise_sd"] for r in nr], [r["mean_Delta"] for r in nr], "o-",
               label="Delta (decision)")
    ax[0].plot([r["noise_sd"] for r in nr], [r["abs_risk_f0"] for r in nr], "s--",
               label="abs risk R_T(f0)")
    ax[0].set(title="noise-invariance (Prop thm:reg-noise)",
              xlabel="unknown noise sd", ylabel="value"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot([r["drift_x_budget"] for r in base],
               [r["baseline_false_commit_rate"] for r in base], "^-", color="red",
               label="no-guard false-commit")
    ax[1].plot([r["drift_x_budget"] for r in base],
               [r["guard_false_commit_rate"] for r in base], "o-", color="C2",
               label="guarded false-commit")
    ax[1].plot([r["drift_x_budget"] for r in base],
               [r["guard_commit_rate"] for r in base], "s--", color="C0",
               label="guarded commit rate")
    ax[1].set(title="guard necessary: baseline fails as drift grows",
              xlabel="drift (x budget)", ylabel="rate"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    print("=" * 78)
    print("Smooth source->target drift bracket (Part 1B)  seed =", SEED, " CPU only")
    print("=" * 78)

    noise = exp_noise_invariance()
    boundary = exp_boundary_and_coverage()
    base = exp_baseline_vs_guard()

    cov_min = min(r["coverage_within_budget"] for r in boundary)
    false_max = max(r["false_commit_rate"] for r in boundary)
    # converse: over-budget drift DOES flip somewhere (necessity of (DS))
    over_max = max(r["over_budget_flip_rate"] for r in boundary)
    # abstention engages: commit rate falls as reach grows with shift
    commit_lo = boundary[0]["commit_rate"]
    commit_hi = boundary[-1]["commit_rate"]
    base_grows = (base[-1]["baseline_false_commit_rate"]
                  > base[0]["baseline_false_commit_rate"] + 0.1)
    guard_safe = max(r["guard_false_commit_rate"] for r in base)

    headline = {
        "noise_invariance_max_dev": noise["max_abs_dev_from_ref"],
        "min_coverage_within_budget": float(cov_min),
        "max_false_commit_within_budget": float(false_max),
        "max_over_budget_flip_rate": float(over_max),
        "commit_rate_low_shift": float(commit_lo),
        "commit_rate_high_shift": float(commit_hi),
        "baseline_false_commit_at_3x": float(base[-1]["baseline_false_commit_rate"]),
        "guard_false_commit_max": float(guard_safe),
        # ---- PASS flags ----
        "PASS_noise_invariance": bool(noise["max_abs_dev_from_ref"] < 0.02),
        "PASS_bracket_covers_truth": bool(cov_min >= 0.99),
        "PASS_zero_false_commit_within_budget": bool(false_max <= 0.005),
        "PASS_converse_over_budget_flips": bool(over_max > 0.02),
        "PASS_abstention_engages_with_shift": bool(commit_hi <= 0.75 * commit_lo + 1e-9),
        "PASS_guard_necessary_baseline_fails": bool(base_grows),
        "PASS_guard_stays_safe": bool(guard_safe <= 0.005),
    }
    headline["ALL_PASS"] = bool(all(v for k, v in headline.items()
                                    if k.startswith("PASS_")))

    results = {
        "description": "Smooth source->target drift bracket (Part 1B); refines "
                       "thm:reg-iff by making drift radius observable via (DS)",
        "seed": SEED, "model": {"f0": F0, "fa": FA, "gS_slope": WS, "L": 0.6},
        "noise_invariance": noise,
        "boundary_coverage": boundary,
        "baseline_vs_guard": base,
        "headline": headline,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    plot_boundary(boundary, FIG1)
    plot_noise_baseline(noise, base, FIG2)

    print("\n-- noise invariance (Delta should be flat) --")
    for r in noise["rows"]:
        print(f"   noise_sd={r['noise_sd']:.2f}  Delta={r['mean_Delta']:+.5f}  "
              f"abs_risk_f0={r['abs_risk_f0']:.3f}")
    print(f"   max |Delta - ref| = {noise['max_abs_dev_from_ref']:.2e}")

    print("\n-- boundary / coverage / safety vs covariate shift --")
    print(f"   {'mu':>4} {'d_obs':>6} {'|c|':>7} {'reach':>7} {'cover':>6} "
          f"{'commit':>7} {'FALSE':>7} {'OB_flip':>7}")
    for r in boundary:
        print(f"   {r['mu_shift']:4.1f} {r['d_obs']:6.3f} "
              f"{abs(r['mean_center_c']):7.3f} {r['mean_reach']:7.3f} "
              f"{r['coverage_within_budget']:6.3f} {r['commit_rate']:7.3f} "
              f"{r['false_commit_rate']:7.4f} {r['over_budget_flip_rate']:7.3f}")

    print("\n-- baseline (no guard) vs guarded, drift as multiple of budget --")
    for r in base:
        print(f"   drift={r['drift_x_budget']:.1f}x  baseline_false="
              f"{r['baseline_false_commit_rate']:.3f}  guard_commit="
              f"{r['guard_commit_rate']:.3f}  guard_false="
              f"{r['guard_false_commit_rate']:.4f}")

    print("\n" + "=" * 78)
    print(json.dumps(headline, indent=2))
    print(f"\nWrote {OUT_JSON}\nWrote {FIG1}\nWrote {FIG2}")
    print("\nALL_PASS =", headline["ALL_PASS"])
    return 0 if headline["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
