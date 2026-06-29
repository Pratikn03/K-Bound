#!/usr/bin/env python3
"""KGA over the REAL ELARA engine candidates (A-POWERED-2 held-out category).

PRE-REGISTERED DESIGN (fixed before viewing test results; report whatever it gives):
  Cache : experiments/phase2/predictions/A-POWERED-2__MVTec_3D-AD__PatchCore_held-out_category
          12 real engine methods x {validation,test} x seeds 42..71 (per-sample raw_score+label).
  Freeze/default candidate : confidence_weighted_mean  (the simple reliability fusion = what the
                             earlier KGA x ELARA proxy panel deployed).
  Adapt candidate pool     : the other 11 engine methods (craf_attention, static_attention,
                             rga_meta_router, rga_boosted_fusion, late_fusion_ensemble,
                             early_fusion_mlp, random_forest, eata/sar/tent/ttt score adapters).
  Selection (label-free)   : per seed, pick the adapt candidate with best VALIDATION AUROC.
  Certificate              : empirical-Bernstein on placement benefits (candidate - baseline) on
                             TEST, alpha=0.10 -> ADAPT / FREEZE / ABSTAIN.
  Metric                   : test AUROC. Aggregate over 30 seeds.
  Reported                 : mean AUROC {always-baseline, always-val-best-candidate, KGA-routed,
                             oracle}, beats_both, false-adapt rate, commit rate. NO cherry-pick.
"""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd
import sys
REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
sys.path.insert(0, REPO)
from kga import KGA
from kga.policy import Decision
from src.uais.kbound.multimodal_guard import placement_benefits, auroc

BASE = os.path.join(REPO, "experiments/phase2/predictions/A-POWERED-2__MVTec_3D-AD__PatchCore_held-out_category")
ALPHA = 0.10
BASELINE = "confidence_weighted_mean"

methods = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)) and not d.startswith("._"))
cand_methods = [m for m in methods if m != BASELINE]
seeds = sorted({int(os.path.basename(f).split("_")[1].split(".")[0])
                for f in glob.glob(os.path.join(BASE, BASELINE, "test", "seed_*.parquet"))
                if not os.path.basename(f).startswith("._")})

def load(method, split, seed):
    f = os.path.join(BASE, method, split, f"seed_{seed}.parquet")
    df = pd.read_parquet(f)[["sample_id", "label", "raw_score"]].dropna(subset=["raw_score"])
    return df.set_index("sample_id")

rows = []
for seed in seeds:
    # baseline val/test
    b_val, b_te = load(BASELINE, "validation", seed), load(BASELINE, "test", seed)
    yv, yt = b_val["label"].astype(int).values, b_te["label"].astype(int).values
    base_te = b_te["raw_score"].values
    # candidate val AUROCs (label-free selection) + test scores aligned to baseline index
    val_auc, te_scores = {}, {}
    for m in cand_methods:
        mv, mt = load(m, "validation", seed), load(m, "test", seed)
        mv = mv.reindex(b_val.index); mt = mt.reindex(b_te.index)
        if mv["raw_score"].isna().any() or mt["raw_score"].isna().any():
            continue
        val_auc[m] = auroc(yv, mv["raw_score"].values)
        te_scores[m] = mt["raw_score"].values
    if not val_auc:
        continue
    best = max(val_auc, key=val_auc.get)               # val-best candidate (label-free)
    cand_te = te_scores[best]
    benefits = placement_benefits(yt, base_te, cand_te)  # mean = AUROC(cand)-AUROC(base)
    br = float(min(2.0, max(np.max(benefits) - np.min(benefits) + 0.05, 0.1))) if benefits.size >= 2 else 2.0
    kga = KGA(alpha=ALPHA)
    cert = kga.certify(scores=benefits, benefit_range=br); dec = kga.decide(cert)
    auc_base = auroc(yt, base_te)
    auc_cand = auroc(yt, cand_te)
    auc_oracle = max([auc_base] + [auroc(yt, te_scores[m]) for m in te_scores])
    auc_kga = auc_cand if dec == Decision.ADAPT else auc_base   # freeze/abstain -> baseline
    rows.append(dict(seed=seed, best_cand=best, decision=dec.value,
                     auc_base=auc_base, auc_cand=auc_cand, auc_oracle=auc_oracle, auc_kga=auc_kga,
                     true_benefit=auc_cand - auc_base,
                     false_adapt=bool(dec == Decision.ADAPT and (auc_cand - auc_base) < 0),
                     delta_hat=cert.delta_hat, eps=cert.epsilon, lo=cert.lower, hi=cert.upper))

n = len(rows)
def mean(k): return float(np.mean([r[k] for r in rows]))
always_base = mean("auc_base"); always_cand = mean("auc_cand"); kga_routed = mean("auc_kga"); oracle = mean("auc_oracle")
adapts = [r for r in rows if r["decision"] == "ADAPT"]
commit = sum(r["decision"] != "ABSTAIN" for r in rows)
beats_both = bool(kga_routed > always_base + 1e-9 and kga_routed > always_cand + 1e-9)
fa_rate = (sum(r["false_adapt"] for r in adapts) / len(adapts)) if adapts else None

from collections import Counter
out = dict(schema="kga_elara_engine_apowered2", cache=os.path.basename(BASE), alpha=ALPHA,
           n_seeds=n, baseline=BASELINE, n_candidates=len(cand_methods),
           mean_auroc=dict(always_baseline=always_base, always_val_best_candidate=always_cand,
                           kga_routed=kga_routed, oracle=oracle),
           beats_both=beats_both, false_adapt_rate=fa_rate,
           commit_rate=commit / n, n_adapt=len(adapts),
           decisions=dict(Counter(r["decision"] for r in rows)),
           best_cand_counts=dict(Counter(r["best_cand"] for r in rows)),
           rows=rows)
od = os.path.join(REPO, "experiments/kbound/results/kga_elara_engine_apowered2")
os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=2, default=float)

print(f"=== KGA over REAL ELARA engine (A-POWERED-2 held-out, {n} seeds, alpha={ALPHA}) ===")
print(f"baseline(freeze)={BASELINE} | candidates={len(cand_methods)}")
print(f"val-best candidate picks: {dict(Counter(r['best_cand'] for r in rows))}")
print(f"decisions: {dict(Counter(r['decision'] for r in rows))}")
print(f"mean test AUROC:  always-baseline={always_base:.4f}  always-val-best={always_cand:.4f}  "
      f"KGA-routed={kga_routed:.4f}  oracle={oracle:.4f}")
print(f"beats_both={beats_both}  false_adapt_rate={fa_rate}  commit_rate={commit/n:.3f}")
print(f"wrote {od}/results.json")
