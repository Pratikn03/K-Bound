"""Validation for the regression slice of Conjecture 1 (squared loss).

REG-1 (noise-invariance of the decision):
    Y = g(X) + eps with E[eps|X]=0, arbitrary unknown (heteroscedastic) noise law.
    Then Delta = R(f0)-R(fa) = E[(f0-g)^2-(fa-g)^2] does NOT depend on the noise,
    while each absolute risk shifts by E[eps^2]. The adapt/freeze decision is
    ordinal-robust; cardinal risk estimation is confounded.

REG-2 (bounded-drift knowability boundary; an exact per-family IFF):
    Target conditional mean g_T = g_S + b with |b(x)| <= B (bounded concept drift),
    plus the covariate-shift machinery for the source-transfer term. Write
        U   = E_T[(f0-fa)(f0+fa)]      (unlabeled-observable)
        T_S = E_T[(f0-fa) g_S]         (importance-weighted estimable, radius eps_n)
        W   = E_T[|f0-fa|]             (unlabeled-observable)
    Then Delta = U - 2 T_S - 2 E_T[(f0-fa) b]  and  |E_T[(f0-fa) b]| <= B*W with
    EQUALITY at b* = -B*sign(f0-fa). Hence within the family:
      * commit sign(U - 2 T_S) iff |U - 2 T_S| > 2 B W + 2 eps_n  -> wrong-commit <= alpha
        for EVERY admissible drift (achievability);
      * if |U - 2 T_S| <= 2 B W, two admissible drifts flip the sign of Delta while all
        observables are unchanged -> unknowable (converse).
    So: sign Delta is identifiable in the bounded-drift family IFF |U-2T_S| > 2BW.

Checks here: (1) Delta invariant across noise scales while risks shift; (2) zero false
certifications across random AND adversarial drifts at every B; (3) the adversarial
drift saturates the bracket (tightness ~1.00); (4) coverage decays as B grows and the
empirical commit boundary tracks |U-2T_S| = 2BW.
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(HERE)
OUTD = os.path.join(KB, "results", "theory"); os.makedirs(OUTD, exist_ok=True)
OUTJ = os.path.join(OUTD, "regression_conjecture_validation.json")
FIGD = os.path.join(KB, "figures"); FIGF = os.path.join(FIGD, "final")
rng = np.random.default_rng(0)
ALPHA = 0.05


def _load():
    return json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}


def _save(d):
    json.dump(d, open(OUTJ, "w"), indent=2)


def make_world(n=200000, d=3, shift=0.6, seed=1):
    r = np.random.default_rng(seed)
    Xt = r.normal(shift, 1.0, (n, d))                  # target covariates (shifted)
    w_true = np.array([1.0, -0.6, 0.3])
    gS = Xt @ w_true                                   # source conditional mean at target X
    f0 = Xt @ (w_true + np.array([0.25, 0.0, -0.1]))   # frozen model (imperfect)
    fa = Xt @ (w_true + np.array([-0.05, 0.1, 0.05]))  # candidate (better aligned)
    return Xt, gS, f0, fa


def part_reg1():
    _, gS, f0, fa = make_world()
    rows = []
    base = None
    for s0 in [0.0, 0.5, 1.0, 2.0]:
        # heteroscedastic, zero-mean noise of UNKNOWN scale
        scale = s0 * (1.0 + 0.5 * np.abs(gS) / (1e-9 + np.abs(gS).mean()))
        eps = rng.normal(0, 1, gS.shape) * scale
        Y = gS + eps
        R0 = float(np.mean((f0 - Y) ** 2)); Ra = float(np.mean((fa - Y) ** 2))
        rows.append(dict(noise_scale=s0, Delta=R0 - Ra, R_f0=R0, R_fa=Ra,
                         sign=int(np.sign(R0 - Ra))))
        if base is None: base = R0 - Ra
    d = _load()
    d["REG1_noise_invariance"] = dict(
        rows=rows,
        Delta_max_abs_deviation=float(max(abs(r["Delta"] - base) for r in rows)),
        Delta_invariant=bool(max(abs(r["Delta"] - base) for r in rows) < 0.02 * abs(base) + 0.01),
        sign_preserved_all=bool(len({r["sign"] for r in rows}) == 1),
        absolute_risk_confounded=bool(rows[-1]["R_f0"] - rows[0]["R_f0"] > 1.0),
        identity="R(f) = E[(f-g)^2] + E[eps^2]: the noise term cancels in Delta, not in the levels")
    _save(d)
    print("[REG1] Delta invariant:", d["REG1_noise_invariance"]["Delta_invariant"],
          "| sign preserved:", d["REG1_noise_invariance"]["sign_preserved_all"],
          "| risks confounded:", d["REG1_noise_invariance"]["absolute_risk_confounded"])


def part_reg2():
    Xt, gS, f0, fa = make_world()
    diff = f0 - fa
    U = float(np.mean(diff * (f0 + fa)))
    T_S = float(np.mean(diff * gS))                    # population transfer term
    W = float(np.mean(np.abs(diff)))
    margin = abs(U - 2 * T_S)
    eps_n = 0.01 * margin                              # small estimation radius (population-scale demo)
    rows, viol = [], 0
    for B in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]:
        commit = margin > 2 * B * W + 2 * eps_n
        false_cert = 0; flips = 0; reach = 0.0
        for k in range(40):                            # random drifts within the class
            if k == 0:
                b = -B * np.sign(diff)                 # ADVERSARIAL: saturates the bracket
            else:
                z = np.tanh(Xt @ rng.normal(0, 1, Xt.shape[1]))
                b = B * z / max(1e-9, np.abs(z).max())
            shift_term = float(np.mean(diff * b))
            reach = max(reach, abs(shift_term) / (B * W + 1e-12)) if B > 0 else 0.0
            Delta_b = U - 2 * T_S - 2 * shift_term
            if commit and np.sign(Delta_b) != np.sign(U - 2 * T_S):
                false_cert += 1
            if not commit and B > 0:
                flips += int(np.sign(Delta_b) != np.sign(U - 2 * T_S))
        viol += false_cert
        rows.append(dict(B=B, committed=bool(commit), boundary_2BW=2 * B * W,
                         margin=margin, false_certifications=false_cert,
                         adversarial_bracket_saturation=round(reach, 4),
                         sign_flips_observed_when_uncommitted=flips))
    d = _load()
    boundary_B = margin / (2 * W)                      # predicted knowability boundary in B
    committed_Bs = [r["B"] for r in rows if r["committed"]]
    d["REG2_bounded_drift_iff"] = dict(
        U=U, T_S=T_S, W=W, margin=margin, predicted_boundary_B=boundary_B, rows=rows,
        zero_false_certifications=bool(viol == 0),
        bracket_tight_at_adversary=bool(all(abs(r["adversarial_bracket_saturation"] - 1.0) < 0.01
                                            for r in rows if r["B"] > 0)),
        coverage_decays=bool(max(committed_Bs) < boundary_B <= 1.2 or boundary_B > 1.2),
        flips_exist_beyond_boundary=bool(any(r["sign_flips_observed_when_uncommitted"] > 0
                                             for r in rows if not r["committed"])),
        statement="identifiable in the |b|<=B family IFF |U-2T_S| > 2BW; certificate never wrong below, sign genuinely flips above")
    _save(d)
    print("[REG2] zero false certs:", d["REG2_bounded_drift_iff"]["zero_false_certifications"],
          "| bracket tight:", d["REG2_bounded_drift_iff"]["bracket_tight_at_adversary"],
          "| boundary B* =", round(boundary_B, 3),
          "| flips beyond boundary:", d["REG2_bounded_drift_iff"]["flips_exist_beyond_boundary"])


def part_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = _load(); r1 = d["REG1_noise_invariance"]["rows"]; r2 = d["REG2_bounded_drift_iff"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    s = [x["noise_scale"] for x in r1]
    axes[0].plot(s, [x["Delta"] for x in r1], "-o", color="#2a9d8f", label=r"$\Delta$ (decision target)")
    axes[0].plot(s, [x["R_f0"] for x in r1], "--s", color="#e76f51", label=r"$R(f_0)$ (cardinal)")
    axes[0].plot(s, [x["R_fa"] for x in r1], "--^", color="#e9c46a", label=r"$R(f_a)$ (cardinal)")
    axes[0].set_xlabel("unknown noise scale"); axes[0].set_title("REG-1: decision invariant, levels confounded")
    axes[0].legend(fontsize=8)
    Bs = [x["B"] for x in r2["rows"]]
    axes[1].plot(Bs, [x["boundary_2BW"] for x in r2["rows"]], "-o", color="#6b7280", label=r"drift reach $2BW$ (tight)")
    axes[1].axhline(r2["margin"], color="#2a9d8f", lw=2, label=r"observable margin $|U-2T_S|$")
    axes[1].axvline(r2["predicted_boundary_B"], ls=":", color="#5b2a86")
    axes[1].text(r2["predicted_boundary_B"], 0.02, " knowability boundary $B^\\ast$", fontsize=8, color="#5b2a86")
    axes[1].set_xlabel("drift radius B"); axes[1].set_title("REG-2: identifiable iff margin > 2BW")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    for p in (os.path.join(FIGD, "fig_regression_boundary.png"), os.path.join(FIGF, "fig_regression_boundary.png")):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    print("figure written: fig_regression_boundary.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["reg1", "reg2", "figure", "all"])
    a = ap.parse_args()
    if a.part in ("reg1", "all"): part_reg1()
    if a.part in ("reg2", "all"): part_reg2()
    if a.part in ("figure", "all"): part_figure()
