#!/usr/bin/env python3
"""
run_mixed_headtohead.py - the mixed harmful+helpful head-to-head harness.

Pre-registration: docs/research/kbound/MIXED_BENCHMARK_PROTOCOL.md.

Runs SIX policies on the SAME logged per-condition signals:
  always-adapt, always-freeze, AETTA, POEM, KGA, oracle
over the mixed benchmark (default MIXED-PRIMARY = TENT records, 5 seeds), serializes
per-condition arrays per policy, and runs the paired-bootstrap + Holm head-to-head
analysis with the pre-registered WIN/TIE/LOSE verdict (protocol sec 4).

INTEGRITY: this harness only TRANSFORMS cached/fresh records into decisions, metrics,
CIs, and the verdict via the committed rules. It fabricates nothing. KGA's decision is
taken from the cached `kga_decision` field by default (the canonical decision already
serialized by the KGA pipeline) OR recomputed with analysis.decide_kga when
--recompute-kga is set (identical machinery). POEM/AETTA decisions come from
poem_aetta.baselines (faithful ports; see that module's docstrings + protocol sec 2).

Device-selectable (--device): the cached arm is pure-CPU numpy; --device only matters
if --recompute-kga triggers the (still numpy/sklearn) KGA estimator. No torch needed.

Reuses experiments/kbound/wilds/multiseed_paired_ci.py primitives (paired_boot, holm).

USAGE (see RUN_ON_MAC_POEM_AETTA.md for the exact Mac invocation):
  python run_mixed_headtohead.py \
      --records-dir experiments/kbound/results/stress_grid_multiseed_v1 \
      --dataset cifar10c --adapter tent --seeds 0 1 2 3 4 \
      --out-dir experiments/kbound/results/mixed_headtohead_v1
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WILDS = os.path.join(os.path.dirname(_HERE), "wilds")
for _p in (_HERE, _WILDS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baselines as BL                      # poem_decision, aetta_decision, etc.
import multiseed_paired_ci as MPC           # paired_boot, holm (reused)

ALPHA_FALSE_ADAPT = 0.10
REGIME_TAU = 0.02
HOLM_ALPHA = 0.05
NBOOT = 10000
BOOT_SEED = 20260619                        # registration date; FROZEN (protocol sec 6)

POLICIES = ["always_adapt", "always_freeze", "aetta", "poem", "kga", "oracle"]
# Policies actually competed in the head-to-head verdict (oracle = ceiling, not a
# competitor; trivials are the floors KGA must also beat for the legacy "beats both").
COMPETITORS_VS_KGA = ["poem", "aetta", "always_adapt", "always_freeze"]


# ----------------------------------------------------------------------------- #
#  Load cached per-condition records for one adapter+seed                         #
# ----------------------------------------------------------------------------- #
def load_records(records_dir, dataset, adapter, seed):
    """Returns the list of per-condition record dicts (the KGA per_condition file)."""
    p = os.path.join(records_dir, f"seed{seed}",
                     f"per_condition_{dataset}_{adapter}_seed{seed}.json")
    if not os.path.exists(p):
        # also try a flat layout (no seed subdir)
        alt = os.path.join(records_dir, f"per_condition_{dataset}_{adapter}_seed{seed}.json")
        if os.path.exists(alt):
            p = alt
        else:
            raise FileNotFoundError(f"records not found: {p} (nor {alt})")
    with open(p) as f:
        return json.load(f)["records"]


# ----------------------------------------------------------------------------- #
#  Compute each policy's decision array on a record list                          #
# ----------------------------------------------------------------------------- #
def policy_decisions(records, recompute_kga=False, kga_alpha=ALPHA_FALSE_ADAPT,
                     num_classes=BL.DEFAULT_NUM_CLASSES):
    """Returns dict policy -> np.array of decisions in {ADAPT,FREEZE,ABSTAIN}.
    KGA may ABSTAIN (scored as FREEZE for accuracy/regret); the others do not."""
    out = {}
    out["always_adapt"] = BL.always_adapt_decision(records)
    out["always_freeze"] = BL.always_freeze_decision(records)
    out["oracle"] = BL.oracle_decision(records)
    out["poem"] = BL.poem_decision(records, num_classes=num_classes)
    out["aetta"] = BL.aetta_decision(records, num_classes=num_classes)

    if recompute_kga:
        # Identical machinery to the paper: LOO gradient-boosted B_hat(Z) + conformal eps.
        import analysis as KGA  # experiments/kbound/wilds/analysis.py
        Z = np.array([r["Z"] for r in records], float)
        B = np.array([r["B"] for r in records], float)
        _, _, dec = KGA.decide_kga(Z, B, alpha=kga_alpha)
        out["kga"] = np.asarray(dec, dtype=object)
    else:
        # use the canonical decision already serialized by the KGA pipeline
        out["kga"] = np.array([r["kga_decision"] for r in records], dtype=object)
    return out


# ----------------------------------------------------------------------------- #
#  Per-condition accuracy / regret arrays for a policy                            #
# ----------------------------------------------------------------------------- #
def policy_regret(records, decisions):
    """Returns (regret_array, accuracy_array). ABSTAIN/FREEZE -> source acc a0 (safe).
    Identical regret convention to multiseed_paired_ci / the locked stress grid."""
    a0 = np.array([r["a0"] for r in records], float)
    aa = np.array([r["a_adapted"] for r in records], float)
    oracle = np.maximum(a0, aa)
    adapt = decisions == "ADAPT"
    pol_acc = np.where(adapt, aa, a0)
    return oracle - pol_acc, pol_acc


# ----------------------------------------------------------------------------- #
#  Serialize per-condition records for a policy in the multiseed_paired_ci schema #
# ----------------------------------------------------------------------------- #
def serialize_policy(out_dir, dataset_tag, policy, seed, records, decisions):
    """Writes per_condition_<dataset_tag>_<policy>_seed<S>.json in the schema
    multiseed_paired_ci.py expects: records[*].kga_decision holds THIS policy's
    decision (the field name is kept for analysis-script compatibility; see protocol
    sec 6). All per-condition arrays are written (not aggregates)."""
    recs_out = []
    for r, d in zip(records, decisions):
        recs_out.append({
            "condition": r["condition"],
            "B": float(r["B"]),
            "a0": float(r["a0"]),
            "a_adapted": float(r["a_adapted"]),
            "oracle_action": r.get("oracle_action",
                                   "ADAPT" if float(r["B"]) > 0 else "FREEZE"),
            "kga_decision": str(d),        # THIS policy's decision (schema compat)
            "policy_decision": str(d),     # explicit alias, unambiguous
            "eps_conformal": float(r.get("eps_conformal", 0.0)),
            "b_hat": float(r.get("b_hat", 0.0)),
            "Z": list(r["Z"]),
            "Z_names": list(r["Z_names"]),
        })
    payload = {
        "seed": seed, "benchmark": dataset_tag, "method": policy,
        "alpha": ALPHA_FALSE_ADAPT, "n_conditions": len(recs_out),
        "policy": policy, "decision_field": "kga_decision holds this policy's decision",
        "records": recs_out,
    }
    # FLAT layout matching multiseed_paired_ci.load_cell (run_dir/per_condition_*.json),
    # so the same analysis script can re-load these files directly (protocol: reuse
    # multiseed_paired_ci.py). No seed<S>/ subdir.
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"per_condition_{dataset_tag}_{policy}_seed{seed}.json")
    with open(p, "w") as f:
        json.dump(payload, f)
    return p


# ----------------------------------------------------------------------------- #
#  Pooled per-condition regret across seeds (paired), for one policy              #
# ----------------------------------------------------------------------------- #
def pooled_regret_across_seeds(per_seed_records, per_seed_decisions):
    """per_seed_* : lists (one entry per seed) of (records, decisions). Returns the
    per-condition regret averaged over seeds (paired; requires identical condition
    order across seeds, which is enforced)."""
    cond_order = None
    seed_regrets = []
    fa_num = fa_den = 0
    harmful_rates = []
    cover_num = cover_den = 0
    for records, decisions in zip(per_seed_records, per_seed_decisions):
        keys = [r["condition"] for r in records]
        if cond_order is None:
            cond_order = keys
        elif keys != cond_order:
            raise AssertionError("condition order mismatch across seeds (paired CI "
                                 "requires identical order) — protocol sec 5")
        reg, _ = policy_regret(records, decisions)
        seed_regrets.append(reg)
        B = np.array([r["B"] for r in records], float)
        adapt = decisions == "ADAPT"
        fa_num += int(np.sum(adapt & (B < 0)))
        fa_den += len(records)
        harmful_rates.append(float(np.mean(B < -REGIME_TAU)))
        # coverage = decisive-and-correct vs oracle action, among decisive
        oracle_act = np.array([("ADAPT" if float(r["B"]) > 0 else "FREEZE")
                               for r in records], dtype=object)
        decisive = np.isin(decisions, ["ADAPT", "FREEZE"])
        correct = (decisions == oracle_act) & decisive
        cover_num += int(np.sum(correct))
        cover_den += int(np.sum(decisive))
    pooled = np.mean(np.vstack(seed_regrets), axis=0)
    meta = {
        "mean_regret": float(np.mean(pooled)),
        "false_adapt_rate": (fa_num / fa_den) if fa_den else None,
        "false_adapt_num": fa_num, "false_adapt_den": fa_den,
        "coverage_decisive_correct": (cover_num / cover_den) if cover_den else None,
        "decisive_rate": (cover_den / fa_den) if fa_den else None,
        "harmful_base_rate_range": [min(harmful_rates), max(harmful_rates)],
    }
    return pooled, meta


# ----------------------------------------------------------------------------- #
#  Head-to-head paired-bootstrap + Holm + pre-registered verdict                  #
# ----------------------------------------------------------------------------- #
def headtohead_analysis(pooled_by_policy, meta_by_policy, set_name):
    """Implements protocol sec 4. diff(KGA, X) = pooled_regret_KGA - pooled_regret_X,
    per-condition paired; 95% bootstrap CI; Holm over the COMPETITORS_VS_KGA family;
    KGA beats X iff CI entirely below 0 AND survives Holm. Returns the verdict dict."""
    rng = np.random.default_rng(BOOT_SEED)
    kga = pooled_by_policy["kga"]
    comp_rows = []
    pvals = []
    labels = []
    for X in COMPETITORS_VS_KGA:
        diff = kga - pooled_by_policy[X]          # negative => KGA lower regret (better)
        obs, lo, hi, p = MPC.paired_boot(diff, NBOOT, rng)
        lab = f"KGA vs {X}"
        pvals.append(p)
        labels.append(lab)
        comp_rows.append({
            "competitor": X, "label": lab,
            "kga_mean_regret": float(np.mean(kga)),
            "competitor_mean_regret": float(np.mean(pooled_by_policy[X])),
            "mean_diff_kga_minus_competitor": obs,
            "ci95_lo": lo, "ci95_hi": hi, "p_raw": p,
        })
    holm_adj = MPC.holm(pvals, labels)
    for row in comp_rows:
        row["p_holm"] = holm_adj[row["label"]]
        row["ci_entirely_below_0"] = bool(row["ci95_hi"] < 0.0)
        row["ci_entirely_above_0"] = bool(row["ci95_lo"] > 0.0)
        row["kga_beats"] = bool(row["ci_entirely_below_0"] and row["p_holm"] < HOLM_ALPHA)
        row["kga_loses"] = bool(row["ci_entirely_above_0"] and row["p_holm"] < HOLM_ALPHA)

    def row(X):
        return next(r for r in comp_rows if r["competitor"] == X)

    poem_r, aetta_r = row("poem"), row("aetta")
    kga_fa = meta_by_policy["kga"]["false_adapt_rate"]
    fa_ok = (kga_fa is not None) and (kga_fa <= ALPHA_FALSE_ADAPT)

    # ----- pre-registered three-way verdict on the no-harm SOTA (POEM, AETTA) -----
    beats_poem = poem_r["kga_beats"]
    beats_aetta = aetta_r["kga_beats"]
    loses_poem = poem_r["kga_loses"]
    loses_aetta = aetta_r["kga_loses"]

    if loses_poem or loses_aetta:
        verdict = "LOSE"
    elif beats_poem and beats_aetta and fa_ok:
        verdict = "WIN"
    else:
        verdict = "TIE"  # at least one CI includes 0, and no loss

    # legacy "beats both trivials" (the existing headline) for cross-reference
    beats_adapt = row("always_adapt")["kga_beats"]
    beats_freeze = row("always_freeze")["kga_beats"]

    return {
        "set_name": set_name,
        "metric": "mean regret-to-oracle (primary); false-adapt rate (secondary)",
        "win_criterion": ("KGA beats X iff 95% paired-bootstrap CI of "
                          "(regret_KGA - regret_X) entirely below 0 AND survives Holm "
                          "(family-wise alpha=0.05); WIN needs beats POEM and AETTA AND "
                          "KGA false-adapt <= 0.10; LOSE if any CI entirely above 0."),
        "nboot": NBOOT, "boot_seed": BOOT_SEED, "holm_alpha": HOLM_ALPHA,
        "alpha_false_adapt": ALPHA_FALSE_ADAPT,
        "kga_false_adapt_rate": kga_fa, "kga_false_adapt_le_alpha": fa_ok,
        "comparisons": comp_rows,
        "beats_poem": beats_poem, "beats_aetta": beats_aetta,
        "loses_to_poem": loses_poem, "loses_to_aetta": loses_aetta,
        "beats_always_adapt": beats_adapt, "beats_always_freeze": beats_freeze,
        "legacy_beats_both_trivials": bool(beats_adapt and beats_freeze),
        "VERDICT": verdict,
        "verdict_legend": {
            "WIN": "KGA dominates the no-harm SOTA on mixed regret (both, Holm, FA<=alpha)",
            "TIE": "KGA matches the no-harm SOTA on regret; differentiator is the "
                   "anytime false-adapt certificate they lack",
            "LOSE": "KGA is beaten by POEM and/or AETTA on mixed regret; rescope to the "
                    "certificate / regimes KGA does win",
        },
    }


# ----------------------------------------------------------------------------- #
#  git hash for the manifest                                                      #
# ----------------------------------------------------------------------------- #
def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- #
#  Main                                                                           #
# ----------------------------------------------------------------------------- #
def run(records_dir, dataset, adapter, seeds, out_dir, recompute_kga=False,
        set_name=None, pool_adapters=None, num_classes=BL.DEFAULT_NUM_CLASSES,
        device="cpu"):
    """pool_adapters: optional list of adapters to POOL per seed (MIXED-POOLED). If
    given, records from each adapter are concatenated per seed (condition keys are
    namespaced by adapter to keep the order well-defined)."""
    t0 = time.time()
    adapters = pool_adapters if pool_adapters else [adapter]
    dataset_tag = set_name or (dataset if len(adapters) == 1 else f"{dataset}_pooled")

    per_seed_records = []
    per_seed_decisions = {p: [] for p in POLICIES}
    serialized = {p: [] for p in POLICIES}

    for s in seeds:
        # build the (pooled) record list for this seed
        recs = []
        for ad in adapters:
            rs = load_records(records_dir, dataset, ad, s)
            if len(adapters) > 1:
                for r in rs:
                    r = dict(r)
                    r["condition"] = f"{ad}::{r['condition']}"
                    recs.append(r)
            else:
                recs.extend(rs)
        per_seed_records.append(recs)
        dec = policy_decisions(recs, recompute_kga=recompute_kga, num_classes=num_classes)
        for p in POLICIES:
            per_seed_decisions[p].append(dec[p])
            path = serialize_policy(out_dir, dataset_tag, p, s, recs, dec[p])
            serialized[p].append(path)

    # pooled regret + meta per policy
    pooled_by_policy = {}
    meta_by_policy = {}
    for p in POLICIES:
        pooled, meta = pooled_regret_across_seeds(per_seed_records, per_seed_decisions[p])
        pooled_by_policy[p] = pooled
        meta_by_policy[p] = meta

    h2h = headtohead_analysis(pooled_by_policy, meta_by_policy, dataset_tag)

    results = {
        "protocol": "MIXED_BENCHMARK_PROTOCOL.md (pre-registered 2026-06-19)",
        "dataset": dataset, "adapters": adapters, "set_name": dataset_tag,
        "seeds": list(seeds), "n_conditions_per_seed": len(per_seed_records[0]),
        "device": device, "recompute_kga": recompute_kga,
        "policy_mean_regret": {p: meta_by_policy[p]["mean_regret"] for p in POLICIES},
        "policy_false_adapt_rate": {p: meta_by_policy[p]["false_adapt_rate"] for p in POLICIES},
        "policy_coverage_decisive_correct": {
            p: meta_by_policy[p]["coverage_decisive_correct"] for p in POLICIES},
        "policy_decisive_rate": {p: meta_by_policy[p]["decisive_rate"] for p in POLICIES},
        "harmful_base_rate_range": meta_by_policy["kga"]["harmful_base_rate_range"],
        "headtohead": h2h,
        "serialized_per_condition_files": serialized,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"HEADTOHEAD_RESULTS_{dataset_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    manifest = {
        "git_hash": _git_hash(), "seeds": list(seeds), "adapters": adapters,
        "records_dir": os.path.abspath(records_dir), "out_dir": os.path.abspath(out_dir),
        "wall_time_sec": round(time.time() - t0, 2), "device": device,
        "boot_seed": BOOT_SEED, "nboot": NBOOT, "alpha_false_adapt": ALPHA_FALSE_ADAPT,
        "python": sys.version.split()[0],
    }
    with open(os.path.join(out_dir, f"result_manifest_{dataset_tag}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return out_path, results


def _print_summary(results):
    h = results["headtohead"]
    print("=" * 78)
    print(f"MIXED HEAD-TO-HEAD  set={results['set_name']}  "
          f"adapters={results['adapters']}  seeds={results['seeds']}  "
          f"n_cond/seed={results['n_conditions_per_seed']}")
    print(f"harmful base rate range: {results['harmful_base_rate_range']}")
    print("-" * 78)
    print("policy           mean_regret   false_adapt   decisive   cover(dec)")
    for p in POLICIES:
        mr = results["policy_mean_regret"][p]
        fa = results["policy_false_adapt_rate"][p]
        dr = results["policy_decisive_rate"][p]
        cv = results["policy_coverage_decisive_correct"][p]
        fa_s = "  n/a " if fa is None else f"{fa:.4f}"
        dr_s = "  n/a " if dr is None else f"{dr:.3f}"
        cv_s = "  n/a " if cv is None else f"{cv:.3f}"
        print(f"  {p:14s}  {mr:10.5f}   {fa_s:>9s}   {dr_s:>7s}   {cv_s:>7s}")
    print("-" * 78)
    for r in h["comparisons"]:
        print(f"  {r['label']:22s} diff={r['mean_diff_kga_minus_competitor']:+.5f} "
              f"CI[{r['ci95_lo']:+.5f},{r['ci95_hi']:+.5f}] p_holm={r['p_holm']:.2e} "
              f"beats={r['kga_beats']} loses={r['kga_loses']}")
    print("-" * 78)
    print(f"  KGA false-adapt <= alpha(0.10): {h['kga_false_adapt_le_alpha']} "
          f"(rate={h['kga_false_adapt_rate']})")
    print(f"  beats POEM={h['beats_poem']}  beats AETTA={h['beats_aetta']}  "
          f"loses_POEM={h['loses_to_poem']}  loses_AETTA={h['loses_to_aetta']}")
    print(f"  legacy beats-both-trivials: {h['legacy_beats_both_trivials']}")
    print(f"  >>> PRE-REGISTERED VERDICT: {h['VERDICT']}  "
          f"({h['verdict_legend'][h['VERDICT']]})")
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records-dir", required=True,
                    help="dir with seed<S>/per_condition_<dataset>_<adapter>_seed<S>.json")
    ap.add_argument("--dataset", default="cifar10c")
    ap.add_argument("--adapter", default="tent", help="MIXED-PRIMARY adapter (default tent)")
    ap.add_argument("--pool-adapters", nargs="*", default=None,
                    help="if set, POOL these adapters per seed (MIXED-POOLED), e.g. tent eata")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--set-name", default=None, help="override the dataset_tag in outputs")
    ap.add_argument("--num-classes", type=int, default=BL.DEFAULT_NUM_CLASSES)
    ap.add_argument("--recompute-kga", action="store_true",
                    help="recompute KGA via analysis.decide_kga instead of cached field")
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda (only affects "
                    "--recompute-kga; cached arm is numpy/CPU)")
    a = ap.parse_args(argv)
    out_path, results = run(
        a.records_dir, a.dataset, a.adapter, a.seeds, a.out_dir,
        recompute_kga=a.recompute_kga, set_name=a.set_name,
        pool_adapters=a.pool_adapters, num_classes=a.num_classes, device=a.device)
    _print_summary(results)
    print("WROTE", out_path)
    return out_path


if __name__ == "__main__":
    main()
