# SUPERSEDED RULE -- EXPLORATORY v1 CODE (defect D10).
# This script computes its certificate radius as ``np.quantile(|Bhat - B|, 1 - alpha)``,
# numpy's *linearly interpolated* quantile.  That is NOT the rule the paper declares.
# The declared rule is the exact split-conformal rank quantile
# ``eps = r_(k)``, ``k = ceil((n + 1)(1 - alpha))``, leave-one-out-of-pool, with ``+inf``
# => ABSTAIN when ``k > n`` -- implemented once in ``kga/certificate.py`` and reached from
# ``docs/research/kbound/scripts/kbound_decide.py``.
#
# This file is retained unconverted on purpose: it is v1/exploratory code, no promoted
# number in the paper comes from it, and its archived JSON outputs were produced under the
# interpolated rule, so converting it in place would silently make those outputs
# irreproducible.  Do not cite any number it prints, and do not copy its radius line.
# It is on the named allowlist in ``tests/test_one_radius_rule.py``; adding a new
# interpolated radius anywhere else fails that test.

"""Numerical validation of the four K-Bound theory extensions.

Every claim below is CHECKED by a real computation here (no asserted numbers).
Outputs: results/theory/theory_extensions_validation.json + two figures.

T1  Le Cam minimax lower bound:  inf_g max-regret >= (delta/2)(1 - TV),
    achieved by the likelihood-ratio test; at TV=0 (the witness) it is delta/2.
T2  Forced abstention:           any rule with false-adapt<=a and false-freeze<=a
    on an indistinguishable pair must abstain w.p. >= 1 - 2a.
T3  Regret decomposition:        policy regret == false-adapt + false-freeze
    + abstention-coverage (exact identity).
T4  Multiclass disagreement:     Delta == P(D) * (a_a^D - a_0^D), so
    sign Delta = sign(a_a^D - a_0^D)  (generalizes binary Theorem 5).
"""
import os, json
import numpy as np
from math import erf, sqrt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # docs/research/kbound
OUT = os.path.join(HERE, "results", "theory"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures", "final"); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))          # standard normal CDF
report = {}

# ---------------------------------------------------------------- T1: Le Cam
# Two worlds, benefit +/- delta; Z ~ N(0,1) (world1, adapt-correct) vs N(mu,1)
# (world2, freeze-correct). TV(N0,Nmu)=2*Phi(mu/2)-1. Uniform prior.
# Bayes-optimal committal regret (closed form) = delta * Phi(-mu/2) = (delta/2)(1-TV).
delta = 1.0
mus = np.linspace(0.0, 4.0, 17)
t1_rows = []
for mu in mus:
    TV = 2 * Phi(mu / 2) - 1
    bound = (delta / 2) * (1 - TV)                     # Le Cam lower bound
    closed = delta * Phi(-mu / 2)                      # optimal threshold-rule regret
    # Monte-Carlo: optimal rule "adapt if Z < mu/2"; regret averaged over the two worlds
    n = 200000
    z1 = rng.normal(0, 1, n); z2 = rng.normal(mu, 1, n)
    err1 = np.mean(z1 >= mu / 2)        # world1: should adapt(Z<mu/2); freeze is wrong
    err2 = np.mean(z2 < mu / 2)         # world2: should freeze; adapt is wrong
    mc = (delta / 2) * (err1 + err2)
    t1_rows.append(dict(mu=float(mu), TV=float(TV), lecam_bound=float(bound),
                        closed_form=float(closed), monte_carlo=float(mc)))
t1_tv0 = t1_rows[0]
report["T1_le_cam"] = {
    "witness_TV0_min_worstcase_regret": t1_tv0["monte_carlo"],   # ~0.5
    "witness_matches_delta_over_2": abs(t1_tv0["monte_carlo"] - delta / 2) < 2e-3,
    "max_abs_gap_bound_vs_montecarlo": float(max(abs(r["lecam_bound"] - r["monte_carlo"]) for r in t1_rows)),
    "bound_is_tight_everywhere": bool(all(abs(r["lecam_bound"] - r["monte_carlo"]) < 5e-3 for r in t1_rows)),
    "curve": t1_rows,
}

# ---------------------------------------------------------------- T2: forced abstention
# On an indistinguishable pair the action law (a,f,s) is identical across worlds.
# Constraint false-adapt = a <= alpha, false-freeze = f <= alpha  =>  s = 1-a-f >= 1-2alpha.
alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]
t2_rows = []
violations = 0
for al in alphas:
    # sample many feasible action laws and check the floor on abstention
    A = rng.uniform(0, al, 5000); Fr = rng.uniform(0, al, 5000)
    S = 1 - A - Fr
    min_s = float(S.min())
    floor = 1 - 2 * al
    if min_s < floor - 1e-9:
        violations += 1
    t2_rows.append(dict(alpha=al, abstain_floor=float(floor), min_abstain_observed=min_s,
                        floor_achieved_at_a_eq_f_eq_alpha=float(1 - 2 * al)))
# cross-check against the real witness run if present
wit = os.path.join(HERE, "results", "witness", "witness_clean.json")
witness_abstain = None
if os.path.exists(wit):
    witness_abstain = json.load(open(wit)).get("abstain_rate")
report["T2_forced_abstention"] = {
    "floor_violations": violations,                       # must be 0
    "holds_for_all_alpha": violations == 0,
    "rows": t2_rows,
    "kga_abstain_rate_on_real_witness": witness_abstain,  # 1.0 expected
}

# ---------------------------------------------------------------- T3: regret decomposition
# Synthetic mixed instances. Benefit B = R_f0 - R_fa (B>0 => adapt better).
N = 4000
R_f0 = rng.uniform(0.1, 0.6, N)
B = rng.normal(0.0, 0.1, N)                 # mixed: some help, some harm
R_fa = R_f0 - B
oracle = np.minimum(R_f0, R_fa)
def regret_of(action):  # action in {'adapt','freeze','abstain'}; abstain->freeze default
    chosen = np.where(action == "adapt", R_fa, R_f0)
    return chosen - oracle
# a noisy trichotomy decision (estimate B with noise, conformal-style band)
Bhat = B + rng.normal(0, 0.05, N); eps = float(np.quantile(np.abs(Bhat - B), 0.9))
dec = np.where(Bhat - eps > 0, "adapt", np.where(Bhat + eps < 0, "freeze", "abstain"))
total = regret_of(dec).sum()
FA = np.sum(np.abs(B)[(dec == "adapt") & (B < 0)])      # adapted into harm
FF = np.sum(np.abs(B)[(dec == "freeze") & (B > 0)])     # froze through benefit
AC = np.sum(np.abs(B)[(dec == "abstain") & (B > 0)])    # abstained through benefit
report["T3_regret_decomposition"] = {
    "total_regret": float(total),
    "FA_plus_FF_plus_AC": float(FA + FF + AC),
    "identity_holds": bool(abs(total - (FA + FF + AC)) < 1e-9),
    "components": {"false_adapt": float(FA), "false_freeze": float(FF), "abstain_coverage": float(AC)},
    "baselines_regret": {
        "always_adapt": float(np.sum(np.abs(B)[B < 0])),
        "always_freeze": float(np.sum(np.abs(B)[B > 0])),
        "trichotomy": float(total),
    },
}

# ---------------------------------------------------------------- T4: multiclass disagreement
def multiclass_case(Kc, n, acc_a, acc_0, seed):
    r = np.random.default_rng(seed)
    y = r.integers(0, Kc, n)
    # f0, fa: each correct with given accuracy; wrong -> a (mostly) different random class
    def make(acc):
        correct = r.random(n) < acc
        pred = y.copy()
        wrongpick = (y + r.integers(1, Kc, n)) % Kc
        pred[~correct] = wrongpick[~correct]
        return pred
    f0 = make(acc_0); fa = make(acc_a)
    D = f0 != fa
    Delta_direct = np.mean(f0 != y) - np.mean(fa != y)       # R(f0)-R(fa)
    PD = D.mean()
    aD_a = np.mean((fa == y)[D]); aD_0 = np.mean((f0 == y)[D])
    Delta_formula = PD * (aD_a - aD_0)
    both_wrong_on_D = np.mean(((f0 != y) & (fa != y))[D])     # >0 distinguishes from binary
    return dict(K=Kc, P_D=float(PD), aD_a=float(aD_a), aD_0=float(aD_0),
                Delta_direct=float(Delta_direct), Delta_via_disagreement=float(Delta_formula),
                gap=float(abs(Delta_direct - Delta_formula)),
                sign_matches=bool(np.sign(Delta_direct) == np.sign(aD_a - aD_0)),
                both_wrong_fraction_on_D=float(both_wrong_on_D))
helpful = multiclass_case(5, 60000, acc_a=0.70, acc_0=0.55, seed=1)   # adapt helps
harmful = multiclass_case(5, 60000, acc_a=0.50, acc_0=0.65, seed=2)   # adapt hurts
report["T4_multiclass_disagreement"] = {
    "helpful_case": helpful, "harmful_case": harmful,
    "identity_holds": bool(helpful["gap"] < 2e-3 and harmful["gap"] < 2e-3),
    "sign_tracks_relative_accuracy": bool(helpful["sign_matches"] and harmful["sign_matches"]),
    "genuinely_multiclass_both_wrong_present": bool(helpful["both_wrong_fraction_on_D"] > 0.05),
    "note": "binary Theorem 5 is the special case a_0^D = 1 - a_a^D (both-wrong fraction = 0)",
}

json.dump(report, open(os.path.join(OUT, "theory_extensions_validation.json"), "w"), indent=2)

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tv = [r["TV"] for r in t1_rows]
plt.figure(figsize=(6, 4))
plt.plot(tv, [r["lecam_bound"] for r in t1_rows], "-", color="#2a9d8f", label="Le Cam bound (δ/2)(1−TV)")
plt.plot(tv, [r["monte_carlo"] for r in t1_rows], "o", color="#e76f51", ms=4, label="optimal rule (Monte-Carlo)")
plt.axhline(0.5, ls="--", lw=.8, color="#999"); plt.text(0.02, 0.51, "δ/2 at TV=0 (witness)", fontsize=8)
plt.xlabel("TV between evidence laws of the two worlds"); plt.ylabel("worst-case committal regret")
plt.title("T1: Le Cam lower bound is tight"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_lecam_bound.png"), dpi=130); plt.close()

plt.figure(figsize=(5.6, 4))
labs = ["always-adapt", "always-freeze", "K-Bound\n(trichotomy)"]
fa = [report["T3_regret_decomposition"]["baselines_regret"]["always_adapt"], 0, FA]
ff = [0, report["T3_regret_decomposition"]["baselines_regret"]["always_freeze"], FF]
ac = [0, 0, AC]
plt.bar(labs, fa, color="#e76f51", label="false-adapt")
plt.bar(labs, ff, bottom=fa, color="#457b9d", label="false-freeze")
plt.bar(labs, ac, bottom=np.array(fa) + np.array(ff), color="#e9c46a", label="abstain-coverage")
plt.ylabel("total regret vs oracle"); plt.title("T3: regret decomposition")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_regret_decomposition.png"), dpi=130); plt.close()

# ---------------------------------------------------------------- console summary
print("T1 Le Cam : witness worst-case regret =", round(report["T1_le_cam"]["witness_TV0_min_worstcase_regret"], 4),
      "| bound tight everywhere:", report["T1_le_cam"]["bound_is_tight_everywhere"])
print("T2 forced  : floor violations =", report["T2_forced_abstention"]["floor_violations"],
      "| KGA abstains on real witness =", report["T2_forced_abstention"]["kga_abstain_rate_on_real_witness"])
print("T3 decomp  : identity holds =", report["T3_regret_decomposition"]["identity_holds"],
      "(total", round(total, 4), "= FA+FF+AC", round(FA + FF + AC, 4), ")")
print("T4 multicls: identity holds =", report["T4_multiclass_disagreement"]["identity_holds"],
      "| sign tracks rel-acc =", report["T4_multiclass_disagreement"]["sign_tracks_relative_accuracy"],
      "| both-wrong on D (helpful) =", round(helpful["both_wrong_fraction_on_D"], 3))
print("saved:", os.path.join(OUT, "theory_extensions_validation.json"))
