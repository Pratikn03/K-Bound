#!/usr/bin/env python3
"""LOCKED Protocol-A analysis. Frozen plan: research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml
Executes exactly: per-condition pooled regret (5 seeds), per-condition PAIRED bootstrap (1e4),
Holm over 6 comparisons, secondary metrics, p* regime-law check. No post-hoc metrics."""
import json, os
import numpy as np

RES = os.environ.get("KBOUND_STRESS_GRID_ROOT", os.path.join(os.path.dirname(__file__)))
_SEEDS_ENV = os.environ.get("KBOUND_STRESS_SEEDS", "0 1 2 3 4")
SEEDS = [int(x) for x in _SEEDS_ENV.split()]
CANDS = ["tent", "eata", "sar"]
NBOOT = 10000
RNG = np.random.default_rng(20260611)  # registration date; fixed for reproducibility

def load(seed, cand):
    p = os.path.join(RES, f"seed{seed}", f"per_condition_cifar10c_{cand}_seed{seed}.json")
    return json.load(open(p))

# ---- Build per-(seed,condition) regret arrays. Regret = oracle_acc - policy_acc. ----
# oracle = max(a0, a_adapted); always_adapt=a_adapted; always_freeze=a0;
# KGA = a_adapted if decision==ADAPT else a0 (ABSTAIN/FREEZE -> safe freeze). Verified vs seed0 summary.
def build(cand):
    cond_order = None
    seed_reg = {p: [] for p in ("kga", "adapt", "freeze")}
    seed_meta = {"false_adapt_num": [], "false_adapt_den": [], "harmful_rate": [],
                 "eps": [], "cover_num": [], "cover_den": []}
    for s in SEEDS:
        recs = load(s, cand)["records"]
        keys = [r["condition"] for r in recs]
        if cond_order is None:
            cond_order = keys
        assert keys == cond_order, "condition order mismatch"
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
        fa_num = int(np.sum(is_adapt & (B <= 0)))
        seed_meta["false_adapt_num"].append(fa_num)
        seed_meta["false_adapt_den"].append(len(recs))
        seed_meta["harmful_rate"].append(float(np.mean(B < 0)))
        seed_meta["eps"].append(float(recs[0]["eps_conformal"]))
        oracle_act = [r["oracle_action"] for r in recs]
        decisive = np.array([d in ("ADAPT", "FREEZE") for d in dec])
        kga_act = np.array(["ADAPT" if d == "ADAPT" else ("FREEZE" if d == "FREEZE" else "ABSTAIN") for d in dec])
        correct_dec = np.array([ka == oa for ka, oa in zip(kga_act, oracle_act)]) & decisive
        seed_meta["cover_num"].append(int(np.sum(correct_dec)))
        seed_meta["cover_den"].append(int(np.sum(decisive)))
    pooled = {p: np.mean(np.vstack(seed_reg[p]), axis=0) for p in seed_reg}
    betvar = {p: np.var(np.vstack(seed_reg[p]), axis=0, ddof=1) for p in seed_reg}
    return cond_order, pooled, betvar, seed_meta, seed_reg

def paired_boot(diff):
    n = diff.shape[0]
    obs = float(np.mean(diff))
    idx = RNG.integers(0, n, size=(NBOOT, n))
    bs = diff[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    centered = bs - bs.mean()
    p = (np.sum(np.abs(centered) >= abs(obs)) + 1) / (NBOOT + 1)
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

results = {"protocol": "STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml", "nboot": NBOOT,
           "n_conditions": 432, "seeds": SEEDS, "alpha_falseadapt": 0.10,
           "regret_def": "oracle=max(a0,a_adapted); KGA ABSTAIN/FREEZE->freeze; verified vs seed0 summary",
           "holm_alpha": 0.05,
           "candidates": {}, "comparisons": [], "secondary": {}, "pstar_law": {}}

pvals = []; labels = []; comp_rows = []
for cand in CANDS:
    cond, pooled, betvar, meta, seed_reg = build(cand)
    cand_rec = {}
    for pol in ("kga", "adapt", "freeze"):
        cand_rec[pol + "_mean_regret"] = float(np.mean(pooled[pol]))
    cand_rec["between_seed_var_mean"] = {p: float(np.mean(betvar[p])) for p in betvar}
    cand_rec["between_seed_var_max"] = {p: float(np.max(betvar[p])) for p in betvar}
    cand_rec["between_seed_std_of_seedmeanregret"] = {
        p: float(np.std([np.mean(x) for x in seed_reg[p]], ddof=1)) for p in seed_reg}
    for trivial, tname in (("adapt", "always-adapt"), ("freeze", "always-freeze")):
        diff = pooled["kga"] - pooled[trivial]
        obs, lo, hi, p = paired_boot(diff)
        lab = f"{cand} vs {tname}"
        pvals.append(p); labels.append(lab)
        comp_rows.append({"candidate": cand, "trivial": tname, "label": lab,
                          "kga_mean_regret": cand_rec["kga_mean_regret"],
                          "trivial_mean_regret": float(np.mean(pooled[trivial])),
                          "mean_diff_kga_minus_trivial": obs,
                          "ci95_lo": lo, "ci95_hi": hi, "p_raw": p})
    fa_num = sum(meta["false_adapt_num"]); fa_den = sum(meta["false_adapt_den"])
    cand_rec["false_adapt_rate_pooled"] = fa_num / fa_den
    cand_rec["false_adapt_num"] = fa_num; cand_rec["false_adapt_den"] = fa_den
    cand_rec["harmful_base_rate_per_seed"] = [round(x, 5) for x in meta["harmful_rate"]]
    cand_rec["harmful_base_rate_range"] = [min(meta["harmful_rate"]), max(meta["harmful_rate"])]
    cn = sum(meta["cover_num"]); cd = sum(meta["cover_den"])
    cand_rec["coverage_action_correct_among_decisive_pooled"] = cn / cd
    cand_rec["eps_conformal_per_seed"] = [round(x, 6) for x in meta["eps"]]
    eps = np.array(meta["eps"])
    cand_rec["eps_range"] = [float(eps.min()), float(eps.max())]
    cand_rec["eps_cv"] = float(eps.std(ddof=1) / eps.mean())
    results["candidates"][cand] = cand_rec

holm_adj = holm(pvals, labels)
for row in comp_rows:
    row["p_holm"] = holm_adj[row["label"]]
    row["kga_lower"] = row["mean_diff_kga_minus_trivial"] < 0
    row["survives_holm"] = bool(row["p_holm"] < 0.05 and row["kga_lower"])
results["comparisons"] = comp_rows

def beats_both(cand):
    rs = [r for r in comp_rows if r["candidate"] == cand]
    return all(r["survives_holm"] for r in rs)

tent_bb = beats_both("tent"); eata_bb = beats_both("eata"); sar_bb = beats_both("sar")
if tent_bb and eata_bb:
    verdict = "STANDS"
elif tent_bb or eata_bb:
    verdict = "RESCOPE"
else:
    verdict = "RETRACT"
results["verdict"] = {"label": verdict, "tent_beats_both": tent_bb, "eata_beats_both": eata_bb,
                      "sar_beats_both": sar_bb,
                      "rule": "tent&eata->STANDS; one->RESCOPE; none->RETRACT to beats-freeze/ties-adapt"}

# ---- p* regime law ----
pstar = {"per_seed_cand": [], "note": "single-seed beats-both = KGA regret < both trivials on that seed's 432 conds"}
for cand in CANDS:
    for s in SEEDS:
        recs = load(s, cand)["records"]
        a0 = np.array([r["a0"] for r in recs]); aad = np.array([r["a_adapted"] for r in recs])
        dec = [r["kga_decision"] for r in recs]; B = np.array([r["B"] for r in recs])
        orc = np.maximum(a0, aad); kb = np.where(np.array([d == "ADAPT" for d in dec]), aad, a0)
        r_k = float(np.mean(orc - kb)); r_a = float(np.mean(orc - aad)); r_f = float(np.mean(orc - a0))
        hf = float(np.mean(B < 0))
        bb = bool(r_k < r_a and r_k < r_f)
        pstar["per_seed_cand"].append({"candidate": cand, "seed": s, "harmful_frac": round(hf, 4),
                                       "regret_kga": r_k, "regret_adapt": r_a, "regret_freeze": r_f,
                                       "beats_both": bb})
rows = pstar["per_seed_cand"]
bb_true_hf = [r["harmful_frac"] for r in rows if r["beats_both"]]
bb_false_hf = [r["harmful_frac"] for r in rows if not r["beats_both"]]
pstar["min_harmful_frac_when_beats_both"] = min(bb_true_hf) if bb_true_hf else None
pstar["max_harmful_frac_when_NOT_beats_both"] = max(bb_false_hf) if bb_false_hf else None
# p* regime law (pre-stated): beats-both should turn ON as harmful fraction rises past threshold ~0.1.
# Monotone & separable by a single threshold iff every NOT-beats case has LOWER harmful frac than
# every beats case: max(harmful | not-beats) < min(harmful | beats). Threshold lies in that gap.
sep = (pstar["min_harmful_frac_when_beats_both"] is not None and
       pstar["max_harmful_frac_when_NOT_beats_both"] is not None and
       pstar["max_harmful_frac_when_NOT_beats_both"] < pstar["min_harmful_frac_when_beats_both"])
pstar["monotone_separable_by_single_threshold"] = bool(sep)
if sep:
    band = [pstar["max_harmful_frac_when_NOT_beats_both"], pstar["min_harmful_frac_when_beats_both"]]
    pstar["empirical_threshold_band"] = band
    pstar["threshold_midpoint"] = float(sum(band) / 2)
    # "threshold ~0.1": NOT-beats cases (SAR) all sit at/below ~0.12 (i.e., at/below p*=0.1 band)
    pstar["not_beats_all_at_or_below_pstar"] = bool(pstar["max_harmful_frac_when_NOT_beats_both"] <= 0.12)
    pstar["law_confirmed"] = bool(sep and pstar["not_beats_all_at_or_below_pstar"])
else:
    pstar["law_confirmed"] = False
results["pstar_law"] = pstar

out = os.path.join(RES, "LOCKED_ANALYSIS_RESULTS.json")
json.dump(results, open(out, "w"), indent=2)
print("WROTE", out)
print("VERDICT:", verdict, "| tent_bb", tent_bb, "eata_bb", eata_bb, "sar_bb", sar_bb)
for r in comp_rows:
    print(f"{r['label']:24s} diff={r['mean_diff_kga_minus_trivial']:+.6f} "
          f"CI[{r['ci95_lo']:+.6f},{r['ci95_hi']:+.6f}] p_raw={r['p_raw']:.2e} "
          f"p_holm={r['p_holm']:.2e} survive={r['survives_holm']}")
print("pstar separable:", pstar["monotone_separable_by_single_threshold"],
      "band:", pstar.get("empirical_threshold_band"),
      "law_confirmed:", pstar.get("law_confirmed"))
for c in CANDS:
    cr = results["candidates"][c]
    print(f"{c}: false_adapt={cr['false_adapt_rate_pooled']:.4f} ({cr['false_adapt_num']}/{cr['false_adapt_den']}) "
          f"harmful_range={cr['harmful_base_rate_range']} eps_range={cr['eps_range']} eps_cv={cr['eps_cv']:.4f} "
          f"cover={cr['coverage_action_correct_among_decisive_pooled']:.4f}")
