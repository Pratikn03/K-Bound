"""
multiseed_paired_ci.py - torch-free multi-seed aggregation + PAIRED bootstrap CIs +
Holm correction over the per-condition files written by per_condition_serialize.py.

This is the SAME analysis the locked CIFAR-10-C stress grid uses
(experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py):
per-condition pooled regret across N seeds, per-condition PAIRED bootstrap (default 1e4),
Holm over the (candidate x trivial) comparisons, plus the p* regime-law check.  It is
generalized here to:
  * read per_condition_<dataset>_<method>_seed<S>.json from any run directory,
  * accept an arbitrary set of methods (>= 1) and seeds (>= 1; CIs need >= 2 to be
    meaningful but the code runs and reports n_seeds),
  * emit MULTISEED_ANALYSIS_RESULTS.json with the LOCKED_ANALYSIS_RESULTS schema.

Pure numpy. No torch, no sklearn.  Used both by the GPU post-processing step and by the
CPU verification harness (verify_runner_pipeline.py).
"""
from __future__ import annotations
import os
import json
import numpy as np


def load_cell(run_dir, dataset, method, seed):
    p = os.path.join(run_dir, f"per_condition_{dataset}_{method}_seed{seed}.json")
    with open(p) as f:
        return json.load(f)


def _build(run_dir, dataset, method, seeds):
    """Per-(seed,condition) regret arrays. Regret = oracle - policy.
    oracle = max(a0, a_adapted); adapt = a_adapted; freeze = a0;
    KGA = a_adapted if decision==ADAPT else a0 (ABSTAIN/FREEZE -> safe freeze).
    Identical regret convention to the locked stress-grid analysis."""
    cond_order = None
    seed_reg = {p: [] for p in ("kga", "adapt", "freeze")}
    meta = {"false_adapt_num": [], "false_adapt_den": [], "harmful_rate": [],
            "eps": [], "cover_num": [], "cover_den": []}
    for s in seeds:
        recs = load_cell(run_dir, dataset, method, s)["records"]
        keys = [r["condition"] for r in recs]
        if cond_order is None:
            cond_order = keys
        if keys != cond_order:
            raise AssertionError(
                f"condition order mismatch for {method} seed{s}: paired CIs require the "
                f"same condition order across seeds")
        a0 = np.array([r["a0"] for r in recs])
        aad = np.array([r["a_adapted"] for r in recs])
        dec = [r["kga_decision"] for r in recs]
        B = np.array([r["B"] for r in recs])
        orc = np.maximum(a0, aad)
        is_adapt = np.array([d == "ADAPT" for d in dec])
        kb = np.where(is_adapt, aad, a0)
        seed_reg["kga"].append(orc - kb)
        seed_reg["adapt"].append(orc - aad)
        seed_reg["freeze"].append(orc - a0)
        meta["false_adapt_num"].append(int(np.sum(is_adapt & (B <= 0))))
        meta["false_adapt_den"].append(len(recs))
        meta["harmful_rate"].append(float(np.mean(B < 0)))
        meta["eps"].append(float(recs[0]["eps_conformal"]))
        oracle_act = [r["oracle_action"] for r in recs]
        decisive = np.array([d in ("ADAPT", "FREEZE") for d in dec])
        kga_act = np.array(["ADAPT" if d == "ADAPT" else ("FREEZE" if d == "FREEZE" else "ABSTAIN")
                            for d in dec])
        correct = np.array([ka == oa for ka, oa in zip(kga_act, oracle_act)]) & decisive
        meta["cover_num"].append(int(np.sum(correct)))
        meta["cover_den"].append(int(np.sum(decisive)))
    pooled = {p: np.mean(np.vstack(seed_reg[p]), axis=0) for p in seed_reg}
    betvar = ({p: np.var(np.vstack(seed_reg[p]), axis=0, ddof=1) for p in seed_reg}
              if len(seeds) > 1 else {p: np.zeros_like(pooled[p]) for p in seed_reg})
    return cond_order, pooled, betvar, meta, seed_reg


def paired_boot(diff, nboot, rng):
    n = diff.shape[0]
    obs = float(np.mean(diff))
    idx = rng.integers(0, n, size=(nboot, n))
    bs = diff[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    centered = bs - bs.mean()
    p = (np.sum(np.abs(centered) >= abs(obs)) + 1) / (nboot + 1)
    return obs, float(lo), float(hi), float(p)


def holm(pvals, labels):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        a = (m - rank) * pvals[i]
        running = max(running, a)
        adj[i] = min(running, 1.0)
    return {labels[i]: adj[i] for i in range(m)}


def analyze(run_dir, dataset, methods, seeds, nboot=10000, seed_rng=20260611,
            alpha_falseadapt=0.10, holm_alpha=0.05):
    """Returns the multi-seed results dict (LOCKED schema)."""
    rng = np.random.default_rng(seed_rng)
    n_conditions = None
    results = {
        "protocol": "MULTISEED_PAIRED_CI_v1 (generalized stress_grid Protocol-A)",
        "dataset": dataset, "nboot": nboot, "n_seeds": len(seeds), "seeds": list(seeds),
        "methods": list(methods), "alpha_falseadapt": alpha_falseadapt,
        "regret_def": "oracle=max(a0,a_adapted); KGA ABSTAIN/FREEZE->freeze",
        "holm_alpha": holm_alpha,
        "candidates": {}, "comparisons": [], "pstar_law": {},
    }
    pvals = []; labels = []; comp_rows = []
    per_seed_cand = []
    for cand in methods:
        cond, pooled, betvar, meta, seed_reg = _build(run_dir, dataset, cand, seeds)
        if n_conditions is None:
            n_conditions = len(cond)
        cand_rec = {}
        for pol in ("kga", "adapt", "freeze"):
            cand_rec[pol + "_mean_regret"] = float(np.mean(pooled[pol]))
        cand_rec["between_seed_var_mean"] = {p: float(np.mean(betvar[p])) for p in betvar}
        cand_rec["between_seed_var_max"] = {p: float(np.max(betvar[p])) for p in betvar}
        cand_rec["between_seed_std_of_seedmeanregret"] = {
            p: (float(np.std([np.mean(x) for x in seed_reg[p]], ddof=1)) if len(seeds) > 1 else 0.0)
            for p in seed_reg}
        for trivial, tname in (("adapt", "always-adapt"), ("freeze", "always-freeze")):
            diff = pooled["kga"] - pooled[trivial]
            obs, lo, hi, p = paired_boot(diff, nboot, rng)
            lab = f"{cand} vs {tname}"
            pvals.append(p); labels.append(lab)
            comp_rows.append({
                "candidate": cand, "trivial": tname, "label": lab,
                "kga_mean_regret": cand_rec["kga_mean_regret"],
                "trivial_mean_regret": float(np.mean(pooled[trivial])),
                "mean_diff_kga_minus_trivial": obs, "ci95_lo": lo, "ci95_hi": hi, "p_raw": p})
        fa_num = sum(meta["false_adapt_num"]); fa_den = sum(meta["false_adapt_den"])
        cand_rec["false_adapt_rate_pooled"] = fa_num / fa_den if fa_den else None
        cand_rec["false_adapt_num"] = fa_num; cand_rec["false_adapt_den"] = fa_den
        cand_rec["harmful_base_rate_per_seed"] = [round(x, 5) for x in meta["harmful_rate"]]
        cand_rec["harmful_base_rate_range"] = [min(meta["harmful_rate"]), max(meta["harmful_rate"])]
        cn = sum(meta["cover_num"]); cdn = sum(meta["cover_den"])
        cand_rec["coverage_action_correct_among_decisive_pooled"] = (cn / cdn) if cdn else None
        cand_rec["eps_conformal_per_seed"] = [round(x, 6) for x in meta["eps"]]
        eps = np.array(meta["eps"])
        cand_rec["eps_range"] = [float(eps.min()), float(eps.max())]
        cand_rec["eps_cv"] = float(eps.std(ddof=1) / eps.mean()) if len(seeds) > 1 and eps.mean() else None
        results["candidates"][cand] = cand_rec
        # p* per (cand, seed)
        for s in seeds:
            recs = load_cell(run_dir, dataset, cand, s)["records"]
            a0 = np.array([r["a0"] for r in recs]); aad = np.array([r["a_adapted"] for r in recs])
            dec = [r["kga_decision"] for r in recs]; B = np.array([r["B"] for r in recs])
            orc = np.maximum(a0, aad)
            kb = np.where(np.array([d == "ADAPT" for d in dec]), aad, a0)
            r_k = float(np.mean(orc - kb)); r_a = float(np.mean(orc - aad)); r_f = float(np.mean(orc - a0))
            per_seed_cand.append({
                "candidate": cand, "seed": s, "harmful_frac": round(float(np.mean(B < 0)), 4),
                "regret_kga": r_k, "regret_adapt": r_a, "regret_freeze": r_f,
                "beats_both": bool(r_k < r_a and r_k < r_f)})
    results["n_conditions"] = n_conditions
    holm_adj = holm(pvals, labels) if pvals else {}
    for row in comp_rows:
        row["p_holm"] = holm_adj.get(row["label"], 1.0)
        row["kga_lower"] = row["mean_diff_kga_minus_trivial"] < 0
        row["survives_holm"] = bool(row["p_holm"] < holm_alpha and row["kga_lower"])
    results["comparisons"] = comp_rows

    def beats_both(cand):
        rs = [r for r in comp_rows if r["candidate"] == cand]
        return bool(rs) and all(r["survives_holm"] for r in rs)

    bb = {c: beats_both(c) for c in methods}
    results["beats_both_by_candidate"] = bb

    # p* regime-law separability (same rule as locked analysis)
    bb_true_hf = [r["harmful_frac"] for r in per_seed_cand if r["beats_both"]]
    bb_false_hf = [r["harmful_frac"] for r in per_seed_cand if not r["beats_both"]]
    pstar = {"per_seed_cand": per_seed_cand,
             "min_harmful_frac_when_beats_both": (min(bb_true_hf) if bb_true_hf else None),
             "max_harmful_frac_when_NOT_beats_both": (max(bb_false_hf) if bb_false_hf else None)}
    sep = (pstar["min_harmful_frac_when_beats_both"] is not None and
           pstar["max_harmful_frac_when_NOT_beats_both"] is not None and
           pstar["max_harmful_frac_when_NOT_beats_both"] < pstar["min_harmful_frac_when_beats_both"])
    pstar["monotone_separable_by_single_threshold"] = bool(sep)
    results["pstar_law"] = pstar
    return results


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Multi-seed paired-CI analysis over per-condition files")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset", required=True, help="e.g. camelyon17 or imagenet-r")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--nboot", type=int, default=10000)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    res = analyze(a.run_dir, a.dataset, a.methods, a.seeds, nboot=a.nboot)
    out = a.out or os.path.join(a.run_dir, "MULTISEED_ANALYSIS_RESULTS.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", out)
    print(f"n_conditions={res['n_conditions']} n_seeds={res['n_seeds']} methods={res['methods']}")
    for r in res["comparisons"]:
        print(f"  {r['label']:28s} diff={r['mean_diff_kga_minus_trivial']:+.6f} "
              f"CI[{r['ci95_lo']:+.6f},{r['ci95_hi']:+.6f}] p_holm={r['p_holm']:.2e} "
              f"survive={r['survives_holm']}")
    return out


if __name__ == "__main__":
    main()
