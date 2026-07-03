#!/usr/bin/env python3
"""Decision baselines for K-Bound: ATC / entropy / EATA-filter / drift-gate as
adapt-vs-freeze DECISION RULES, evaluated head-to-head with KGA on logged
per-condition evidence.

Closes the reviewer gap: Props 22--24 argue these decision styles are degenerate
(epsilon=0) certificates; this script RUNS them on the same logged (Z, a0, aa)
rows the KGA results came from, with identical metrics (policy_metrics from
cifar_tent_mps_v2.py).

Baselines (committal adapt/freeze unless noted), from the logged 11-dim Z
[pre_entropy, pre_conf, pre_pbal, post_entropy, post_conf, post_pbal,
 pbal_drop, entropy_drop, frac_highconf, marginal_KL, update_norm]:

  atc_conf        ADAPT iff post_conf - pre_conf > 0          (ATC-style plug-in sign(M);
                                                               source-calibrated confidence
                                                               comparison, Prop 5 / Prop 22)
  atc_conf_loo    same statistic, threshold tuned leave-one-out (best-effort ATC)
  ent_progress    ADAPT iff entropy_drop > 0                  (entropy-minimization's own
                                                               progress signal)
  ent_progress_loo same, LOO-tuned threshold
  eata_filter_loo ADAPT iff frac_highconf > tau_LOO           (EATA reliable-sample
                                                               criterion as a gate, Prop 23)
  drift_gate_loo  FREEZE iff marginal_KL > tau_LOO else ADAPT (drift-monitor-as-gate,
                                                               'protected adaptation' style)
  gbm_committal   ADAPT iff Bhat_LOO(Z) > 0                   (KGA's own benefit model with
                                                               epsilon=0: isolates the value
                                                               of the abstain band)
  best_single_hindsight  upper envelope: best feature + best threshold + best sign chosen
                         IN HINDSIGHT on the test conditions (not realizable; if KGA beats
                         this, no single-statistic committal rule can win)

NOT computable from logged stats (documented, not faked): true AETTA needs dropout
forward passes; true Agreement-on-the-Line needs a model family across distributions.
Both are represented here only by their decision-style surrogates above.

Usage:
  python run_decision_baselines.py --checkpoint experiments/kbound/results/imagenetc_noise/checkpoint.json \
      --kga-results experiments/kbound/results/imagenetc_noise/decisive_tta_results.json \
      --out-dir experiments/kbound/results/decision_baselines
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ALPHA = 0.10
SEED = 0
EVIDENCE_NAMES = ["pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf",
                  "post_pbal", "pbal_drop", "entropy_drop", "frac_highconf",
                  "marginal_KL", "update_norm"]


# --------------------------------------------------------------------------
# Metrics: identical semantics to cifar_tent_mps_v2.policy_metrics
# --------------------------------------------------------------------------
def policy_metrics(dec, a0, aa, B):
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float); B = np.asarray(B, float)
    dec = np.asarray(dec)
    adapt = dec == "ADAPT"
    real = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    harmful = B < 0
    return {
        "n": int(len(a0)),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "adapt_precision_B>0": float(np.mean(B[adapt] > 0)) if adapt.any() else None,
        "false_adapt_rate_B<0": float(np.mean(B[adapt] < 0)) if adapt.any() else None,
        "harmful_cells_adapted_frac": float(np.mean(adapt[harmful])) if harmful.any() else None,
        "mean_acc": float(real.mean()),
        "regret_vs_oracle": float((oracle - real).mean()),
        "worst_case_acc": float(real.min()),
    }


# --------------------------------------------------------------------------
# KGA reproduction (identical machinery to cifar_tent_mps_v2.decide_kga)
# --------------------------------------------------------------------------
def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=SEED):
    from sklearn.ensemble import GradientBoostingRegressor
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=lr, subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr])
        Bhat[i] = m.predict(Z[i:i + 1])[0]
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, dec


# --------------------------------------------------------------------------
# Threshold rules
# --------------------------------------------------------------------------
def committal(stat, tau=0.0, sign=+1):
    """ADAPT iff sign*(stat - tau) > 0, else FREEZE (no abstain)."""
    s = sign * (np.asarray(stat, float) - tau)
    return np.where(s > 0, "ADAPT", "FREEZE")


def regret_of(dec, a0, aa):
    real = np.where(np.asarray(dec) == "ADAPT", aa, a0)
    return float((np.maximum(a0, aa) - real).mean())


def loo_threshold(stat, a0, aa, sign=+1, grid=201):
    """Leave-one-out tuned threshold: for each held-out i, pick tau minimizing
    training regret on the rest, decide i with it. Same LOO discipline as KGA's
    benefit model -- the baseline gets the SAME calibration information."""
    stat = np.asarray(stat, float)
    n = len(stat)
    lo, hi = float(stat.min()), float(stat.max())
    taus = np.linspace(lo - 1e-9, hi + 1e-9, grid)
    dec = np.empty(n, dtype=object)
    for i in range(n):
        tr = np.arange(n) != i
        best_tau, best_r = taus[0], np.inf
        for t in taus:
            r = regret_of(committal(stat[tr], t, sign), a0[tr], aa[tr])
            if r < best_r - 1e-12:
                best_r, best_tau = r, t
        dec[i] = committal(stat[i:i + 1], best_tau, sign)[0]
    return dec


def best_single_hindsight(Z, a0, aa, grid=201):
    """Upper envelope over ALL single-feature committal rules, feature+threshold+sign
    chosen in hindsight on the evaluation conditions themselves (not realizable)."""
    best = None
    for j in range(Z.shape[1]):
        stat = Z[:, j]
        taus = np.linspace(stat.min() - 1e-9, stat.max() + 1e-9, grid)
        for sign in (+1, -1):
            for t in taus:
                dec = committal(stat, t, sign)
                r = regret_of(dec, a0, aa)
                if best is None or r < best[0] - 1e-12:
                    best = (r, j, float(t), sign, dec)
    r, j, t, sign, dec = best
    return dec, {"feature": EVIDENCE_NAMES[j], "tau": t, "sign": sign}


# --------------------------------------------------------------------------
def run_method(rows, alpha=ALPHA):
    Z = np.array([r["Z"] for r in rows], float)
    a0 = np.array([r["a0"] for r in rows], float)
    aa = np.array([r["aa"] for r in rows], float)
    B = aa - a0
    i = {n: k for k, n in enumerate(EVIDENCE_NAMES)}
    conf_gain = Z[:, i["post_conf"]] - Z[:, i["pre_conf"]]
    ent_drop = Z[:, i["entropy_drop"]]
    frac_hi = Z[:, i["frac_highconf"]]
    kl = Z[:, i["marginal_KL"]]

    out = {}
    # trivial policies + oracle
    out["always_adapt"] = policy_metrics(np.array(["ADAPT"] * len(B)), a0, aa, B)
    out["always_freeze"] = policy_metrics(np.array(["FREEZE"] * len(B)), a0, aa, B)
    out["_oracle_acc"] = float(np.maximum(a0, aa).mean())
    out["_harmful_base_rate"] = float(np.mean(B < 0))

    # decision baselines
    out["atc_conf"] = policy_metrics(committal(conf_gain, 0.0, +1), a0, aa, B)
    out["atc_conf_loo"] = policy_metrics(loo_threshold(conf_gain, a0, aa, +1), a0, aa, B)
    out["ent_progress"] = policy_metrics(committal(ent_drop, 0.0, +1), a0, aa, B)
    out["ent_progress_loo"] = policy_metrics(loo_threshold(ent_drop, a0, aa, +1), a0, aa, B)
    out["eata_filter_loo"] = policy_metrics(loo_threshold(frac_hi, a0, aa, +1), a0, aa, B)
    out["drift_gate_loo"] = policy_metrics(loo_threshold(kl, a0, aa, -1), a0, aa, B)

    dec_h, spec = best_single_hindsight(Z, a0, aa)
    pm = policy_metrics(dec_h, a0, aa, B); pm["hindsight_spec"] = spec
    out["best_single_hindsight"] = pm

    # KGA: full certificate, and its committal (eps=0) ablation
    Bhat, eps, dec_kga = decide_kga(Z, B, alpha=alpha)
    pm = policy_metrics(dec_kga, a0, aa, B); pm["eps_conformal"] = eps
    out["KGA"] = pm
    out["gbm_committal"] = policy_metrics(
        np.where(Bhat > 0, "ADAPT", "FREEZE"), a0, aa, B)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kga-results", default=None,
                    help="decisive_tta_results.json to cross-check KGA reproduction")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    ck = json.load(open(args.checkpoint))
    os.makedirs(args.out_dir, exist_ok=True)
    res = {"alpha": args.alpha, "source_checkpoint": args.checkpoint, "methods": {}}
    for m, rows in ck["rows"].items():
        res["methods"][m] = run_method(rows, alpha=args.alpha)
        print(f"[{m}] n={len(rows)} done")

    # cross-check against logged KGA metrics
    if args.kga_results and os.path.exists(args.kga_results):
        logged = json.load(open(args.kga_results))
        bench = list(logged["benchmarks"].values())[0]
        res["kga_crosscheck"] = {}
        for m in res["methods"]:
            lg = bench["methods"].get(m, {}).get("metrics", {})
            res["kga_crosscheck"][m] = {
                "logged_regret_KGA": lg.get("regret_vs_oracle", {}).get("K_Bound"),
                "reproduced_regret_KGA": res["methods"][m]["KGA"]["regret_vs_oracle"],
            }

    with open(os.path.join(args.out_dir, "decision_baselines.json"), "w") as f:
        json.dump(res, f, indent=1)

    # markdown table
    pol_order = ["always_adapt", "always_freeze", "atc_conf", "atc_conf_loo",
                 "ent_progress", "ent_progress_loo", "eata_filter_loo",
                 "drift_gate_loo", "gbm_committal", "best_single_hindsight", "KGA"]
    lines = ["# Decision baselines vs KGA (same logged conditions, same metrics)", ""]
    for m, r in res["methods"].items():
        lines += [f"## {m}  (oracle acc {r['_oracle_acc']:.3f}, harmful base rate "
                  f"{r['_harmful_base_rate']:.0%})", "",
                  "| policy | mean acc | regret | false-adapt | harmful adapted | worst case | coverage |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for p in pol_order:
            pm = r[p]
            fa = pm["false_adapt_rate_B<0"]
            ha = pm["harmful_cells_adapted_frac"]
            lines.append(
                f"| {p} | {pm['mean_acc']:.3f} | {pm['regret_vs_oracle']:.4f} | "
                f"{'—' if fa is None else f'{fa:.2f}'} | "
                f"{'—' if ha is None else f'{ha:.2f}'} | "
                f"{pm['worst_case_acc']:.3f} | {pm['coverage']:.2f} |")
        lines.append("")
    with open(os.path.join(args.out_dir, "decision_baselines_table.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
