"""KTC synthetic validation (paper-seed experiment; run in full, no fabrication).

Blocks:
  A  Witness: gold-world vs distractor-world with IDENTICAL label-free observables
     and opposite benefit -> certificate abstains, committal gates ~chance regret.
  B  Mixed-stream routing under calibration drift: KTC certificate vs
     never/always/entropy-gate(beta=0)/exchangeable-conformal. Metrics: regret
     to oracle, false-spend rate (alpha=0.10), beats-both.
  C  Prop-1 check: sign(Delta_lambda) == sign(M_lambda + gamma) exactly (machine
     precision) and the |M_lambda|>beta frontier classifies correctly whenever
     |gamma| <= beta.

Everything is condition-level (batches of queries), mirroring the K-Bound grids.
Outputs: ktc_results.json + fig_ktc_regret.png + fig_ktc_lambda_frontier.png.
Seeded; pure numpy + sklearn + matplotlib.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260702)
ALPHA = 0.10
N_QUERIES = 256  # per condition


# --------------------------------------------------------------------- world
def make_condition(rng, regime: str, drift: bool):
    """One deployment condition. Returns (Z_dict, truth_dict).

    regime: 'helpful' (gold retrieval), 'harmful' (distractor retrieval,
    overthinking), 'neutral' (parametric knowledge suffices).
    drift=True applies the deployment calibration drift: distractor retrieval
    scores match gold (similarity!=truth) and confidence temperature shifts.
    """
    n = N_QUERIES
    diff = rng.beta(2, 3, n)                     # query difficulty
    a0 = np.clip(0.95 - 0.75 * diff + rng.normal(0, 0.03, n), 0.05, 0.99)
    q0 = rng.random(n) < a0
    if regime == "helpful":
        fix = (rng.random(n) < 0.85) & ~q0        # gold passages fix wrong answers
        brk = (rng.random(n) < 0.02) & q0
    elif regime == "harmful":
        fix = (rng.random(n) < 0.05) & ~q0        # distractors rarely fix
        brk = (rng.random(n) < 0.55 * diff.mean() + 0.25) & q0  # lock-in breaks
    else:
        fix = (rng.random(n) < 0.10) & ~q0
        brk = (rng.random(n) < 0.05) & q0
    qa = q0.copy(); qa[fix] = True; qa[brk] = False
    answer_changed = fix | brk | (rng.random(n) < 0.05)   # some neutral rewrites
    muD = float(answer_changed.mean())
    B = float(qa.mean() - q0.mean())              # true benefit
    cost = float(rng.uniform(0.8, 1.2))           # mean extra compute (units)

    # ---- label-free observables (condition level)
    conf = a0 + rng.normal(0, 0.02, n)
    if drift:                                     # miscalibration at deployment
        conf = np.clip(conf * 1.12, 0, 1)         # over-confidence shift
    sc_direct = np.clip(a0 + rng.normal(0, 0.06, n), 0, 1)   # self-consistency
    # retrieval similarity: high for gold AND (under drift) for distractors
    base_ret = {"helpful": 0.82, "harmful": 0.80 if drift else 0.55,
                "neutral": 0.45}[regime]
    ret = np.clip(base_ret + rng.normal(0, 0.05, n), 0, 1)
    # divergence of augmented answer from direct high-confidence answers:
    # in harmful regimes the augmented answer contradicts confident directs more
    contradict_conf = float(np.mean(answer_changed & (conf > 0.75)))
    Z = dict(mean_conf=float(conf.mean()), sc_direct=float(sc_direct.mean()),
             retrieval_score=float(ret.mean()), muD=muD,
             flip_rate=float(answer_changed.mean()),
             contradict_conf=contradict_conf,
             entropy=float(1.0 - conf.mean()))
    # split-observable pieces for Block C: s = calibrated correctness scores
    s0 = conf                                       # proxy score for direct
    sa = np.clip(conf + (ret - 0.5) * 0.3, 0, 1)    # proxy score for augmented
    onD = answer_changed
    M = float(np.mean(sa[onD] - s0[onD])) if onD.any() else 0.0
    p_diff = float(np.mean(qa[onD].astype(float) - q0[onD].astype(float))) if onD.any() else 0.0
    gamma = p_diff - M
    return Z, dict(B=B, muD=muD, cost=cost, M=M, gamma=gamma, regime=regime)


def build_panel(rng, n_cond, weights, drift):
    regs = rng.choice(["helpful", "harmful", "neutral"], size=n_cond, p=weights)
    Zs, Ts = [], []
    for r in regs:
        z, t = make_condition(rng, r, drift)
        Zs.append(z); Ts.append(t)
    names = sorted(Zs[0])
    Z = np.array([[z[k] for k in names] for z in Zs])
    return Z, Ts, names


# ------------------------------------------------------------------ policies
def signed_bounds(res, alpha):
    n = len(res)
    lo = np.sort(res)[min(max(int(np.ceil((n + 1) * alpha)), 1), n) - 1]
    hi = np.sort(res)[min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n) - 1]
    return float(lo), float(hi)


def run_block_B(lam=0.0):
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(11)
    # calibration: diverse corpus qualities, includes drift-like conditions
    Zc, Tc, names = build_panel(rng, 240, [0.4, 0.3, 0.3], drift=True)
    Bc = np.array([t["B"] - lam * t["cost"] for t in Tc])
    # deployment: mixed stream, drifted
    Zd, Td, _ = build_panel(rng, 200, [0.35, 0.4, 0.25], drift=True)
    Bd = np.array([t["B"] - lam * t["cost"] for t in Td])
    # exchangeable-conformal baseline calibrated on NO-DRIFT conditions only
    Zc0, Tc0, _ = build_panel(rng, 240, [0.4, 0.3, 0.3], drift=False)
    Bc0 = np.array([t["B"] - lam * t["cost"] for t in Tc0])

    def fit_predict(Ztr, Btr, Zte):
        m = GradientBoostingRegressor(n_estimators=250, max_depth=2,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=0).fit(Ztr, Btr)
        return m.predict(Zte), m

    # KTC certificate: 2-fold cross-fitted signed conformal on drift-diverse cal
    half = len(Zc) // 2
    pred_cal = np.empty(len(Zc))
    p1, m1 = fit_predict(Zc[:half], Bc[:half], Zc[half:])
    p2, m2 = fit_predict(Zc[half:], Bc[half:], Zc[:half])
    pred_cal[half:], pred_cal[:half] = p1, p2
    lo, hi = signed_bounds(Bc - pred_cal, ALPHA)
    _, m_full = fit_predict(Zc, Bc, Zc)
    pred_d = m_full.predict(Zd)
    dec_ktc = np.where(pred_d + lo > 0, 1, np.where(pred_d + hi < 0, -1, 0))

    # exchangeable conformal (no drift in its calibration -> radius too small)
    pred_cal0 = np.empty(len(Zc0))
    q1, n1 = fit_predict(Zc0[:half], Bc0[:half], Zc0[half:])
    q2, n2 = fit_predict(Zc0[half:], Bc0[half:], Zc0[:half])
    pred_cal0[half:], pred_cal0[:half] = q1, q2
    lo0, hi0 = signed_bounds(Bc0 - pred_cal0, ALPHA)
    _, m0_full = fit_predict(Zc0, Bc0, Zc0)
    pred_d0 = m0_full.predict(Zd)
    dec_cp = np.where(pred_d0 + lo0 > 0, 1, np.where(pred_d0 + hi0 < 0, -1, 0))

    # entropy-gate (TARG-style beta=0): threshold tuned on cal for best accuracy
    ent_c = Zc[:, names.index("entropy")]
    ths = np.quantile(ent_c, np.linspace(0.05, 0.95, 19))
    best_th, best_val = None, -1e9
    for th in ths:
        d = (ent_c > th).astype(int)              # uncertain -> spend
        val = np.mean(np.where(d == 1, Bc, 0.0))  # realized gain on cal
        if val > best_val:
            best_val, best_th = val, th
    dec_ent = (Zd[:, names.index("entropy")] > best_th).astype(int)

    oracle = np.maximum(Bd, 0.0)                  # spend iff true Delta_lam>0

    def score(dec, spendlike=(1,)):
        gain = np.where(np.isin(dec, spendlike), Bd, 0.0)
        regret = float(np.mean(oracle - gain))
        fa = float(np.mean(np.isin(dec, spendlike) & (Bd <= 0)))
        return regret, fa

    out = {}
    out["never"] = dict(zip(("regret", "FA"), score(np.zeros(len(Bd)))))
    out["always"] = dict(zip(("regret", "FA"), score(np.ones(len(Bd)))))
    out["entropy_gate_b0"] = dict(zip(("regret", "FA"), score(dec_ent)))
    out["conformal_exch"] = dict(zip(("regret", "FA"), score(dec_cp)))
    out["KTC"] = dict(zip(("regret", "FA"), score(dec_ktc)))
    out["KTC_rates"] = dict(spend=float((dec_ktc == 1).mean()),
                            direct=float((dec_ktc == -1).mean()),
                            abstain=float((dec_ktc == 0).mean()))
    r = out["KTC"]["regret"]
    out["beats_both"] = bool(r < out["never"]["regret"] and r < out["always"]["regret"])
    out["lambda"] = lam
    return out, (Zd, Td, names)


def run_block_A():
    """Witness: identical observable law, opposite benefit."""
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(7)
    n_each = 150
    Zs, Bs = [], []
    for world in (+1, -1):
        for _ in range(n_each):
            # identical observable draws for both worlds
            z = dict(mean_conf=rng.normal(0.78, 0.03), sc_direct=rng.normal(0.75, 0.04),
                     retrieval_score=rng.normal(0.81, 0.04), muD=rng.uniform(0.15, 0.25),
                     flip_rate=rng.uniform(0.15, 0.25),
                     contradict_conf=rng.uniform(0.05, 0.12),
                     entropy=rng.normal(0.22, 0.03))
            Zs.append([z[k] for k in sorted(z)])
            Bs.append(world * rng.uniform(0.10, 0.20))   # opposite benefit
    Z, B = np.array(Zs), np.array(Bs)
    idx = rng.permutation(len(B))
    Z, B = Z[idx], B[idx]
    cal, te = np.arange(len(B)) < 200, np.arange(len(B)) >= 200
    m = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                  subsample=0.8, random_state=0).fit(Z[cal], B[cal])
    res = B[cal] - m.predict(Z[cal])
    lo, hi = signed_bounds(res, ALPHA)
    pred = m.predict(Z[te])
    dec = np.where(pred + lo > 0, 1, np.where(pred + hi < 0, -1, 0))
    abstain = float((dec == 0).mean())
    committal = np.sign(pred)                      # forced-commit baseline
    wrong = float((committal != np.sign(B[te])).mean())
    return dict(abstain_rate_certificate=abstain,
                forced_commit_wrong_rate=wrong,
                mean_abs_true_benefit=float(np.abs(B[te]).mean()))


def run_block_C():
    """Prop-1: sign identity exact; frontier classifies when |gamma|<=beta."""
    rng = np.random.default_rng(3)
    Z, Ts, _ = build_panel(rng, 300, [0.34, 0.33, 0.33], drift=True)
    ident_err = 0
    checked = 0
    frontier_correct = 0
    frontier_total = 0
    beta = 0.06
    for lam in (0.0, 0.05, 0.1):
        for t in Ts:
            if t["muD"] <= 0:
                continue
            M_lam = t["M"] - lam * t["cost"] / t["muD"]
            lhs = np.sign(t["B"] - lam * t["cost"])
            rhs = np.sign(t["muD"] * (M_lam + t["gamma"]))
            checked += 1
            if lhs != rhs and abs(t["B"] - lam * t["cost"]) > 1e-12:
                ident_err += 1
            if abs(t["gamma"]) <= beta and abs(M_lam) > beta:
                frontier_total += 1
                if np.sign(M_lam) == lhs or lhs == 0:
                    frontier_correct += 1
    return dict(identity_checked=checked, identity_violations=ident_err,
                frontier_beta=beta, frontier_cells=frontier_total,
                frontier_correct=frontier_correct)


def main():
    results = {}
    results["A_witness"] = run_block_A()
    b0, _ = run_block_B(lam=0.0)
    b1, (Zd, Td, names) = run_block_B(lam=0.08)
    results["B_routing_lam0"] = b0
    results["B_routing_lam008"] = b1
    results["C_prop1"] = run_block_C()

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pols = ["never", "always", "entropy_gate_b0", "conformal_exch", "KTC"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for j, (blk, ttl) in enumerate(((b0, "λ=0"), (b1, "λ=0.08"))):
        reg = [blk[p]["regret"] for p in pols]
        fa = [blk[p]["FA"] for p in pols]
        x = np.arange(len(pols))
        ax[j].bar(x - 0.2, reg, 0.4, label="regret")
        ax[j].bar(x + 0.2, fa, 0.4, label="false-spend")
        ax[j].axhline(ALPHA, ls="--", lw=0.8, color="k")
        ax[j].set_xticks(x); ax[j].set_xticklabels(pols, rotation=30, ha="right", fontsize=7)
        ax[j].set_title(f"mixed stream, drift ({ttl})", fontsize=9)
        ax[j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_ktc_regret.png"), dpi=150)

    # lambda frontier for one helpful condition
    t = next(t for t in Td if t["regime"] == "helpful" and t["muD"] > 0.05)
    lams = np.linspace(0, 0.4, 100)
    M_lam = t["M"] - lams * t["cost"] / t["muD"]
    fig2, ax2 = plt.subplots(figsize=(4.6, 3))
    ax2.plot(lams, M_lam, label="M_λ (observable)")
    ax2.axhline(0.06, ls="--", c="g", lw=0.8, label="+β")
    ax2.axhline(-0.06, ls="--", c="r", lw=0.8, label="−β")
    ax2.set_xlabel("price λ"); ax2.set_ylabel("cost-adjusted margin")
    ax2.set_title("certified willingness-to-pay λ*", fontsize=9)
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, "fig_ktc_lambda_frontier.png"), dpi=150)

    print(json.dumps(results, indent=1))
    with open(os.path.join(HERE, "ktc_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
