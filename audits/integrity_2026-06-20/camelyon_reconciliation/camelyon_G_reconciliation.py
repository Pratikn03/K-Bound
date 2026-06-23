#!/usr/bin/env python3
"""
camelyon_G_reconciliation.py - Phase 1 gating analysis (2026-06-20)

Resolves the Protocol G contradiction: is the Camelyon17 'beats-both' headline a
legitimate held-out OOD win, or a domain-pooling artifact?

Method: reproduce the EXACT analyze_F.run_split that produced the reported Protocol G
regret (eata_online, gbr, global eps, dev seeds {0,1} / test seeds {2,3,4}), then
re-run it on domain-filtered slices. Uses analyze_F's own functions (no reimplementation).
"""
import json, importlib.util
import numpy as np

ROOT = "/Volumes/T9/uav/AutoML_Flagship_V8"
REC  = f"{ROOT}/experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json"
AFP  = f"{ROOT}/docs/research/kbound/scripts/analyze_F.py"

spec = importlib.util.spec_from_file_location("aF", AFP)
aF = importlib.util.module_from_spec(spec); spec.loader.exec_module(aF)

d = json.load(open(REC))
recs = d["records"]
eata = [r for r in recs if r["candidate"] == "eata_online"]

def to_recs(rows):
    return [aF._one_record(r, candidate="eata_online") for r in rows]

out = {"source": REC, "candidate": "eata_online", "estimator": "gbr",
       "conformal": "global", "dev_seeds": [0, 1], "test_seeds": [2, 3, 4],
       "domain_counts_eata": {}, "harm_profile_by_domain": {}, "run_split_by_domain": {}}

# 0) record bookkeeping
from collections import Counter
out["domain_counts_eata"] = dict(Counter(r["domain"] for r in eata))

# 1) harm profile by domain (B<0 == adapting reduces true accuracy)
for dom in ["test", "val", "id_val"]:
    rows = [r for r in eata if r["domain"] == dom]
    B = np.array([r["B"] for r in rows])
    out["harm_profile_by_domain"][dom] = {
        "n": len(rows), "mean_B": float(B.mean()),
        "frac_harm_Blt0": float(np.mean(B < 0)),
        "min_B": float(B.min()), "max_B": float(B.max())}

# 2) re-run the EXACT locked decision (gbr+global) on domain slices
slices = {"POOLED_test_val_idval": ["test", "val", "id_val"],
          "OOD_test_only": ["test"], "OOD_val_only": ["val"],
          "OOD_test_plus_val": ["test", "val"], "idval_only": ["id_val"]}
for label, doms in slices.items():
    rows = [r for r in eata if r["domain"] in doms]
    m = aF.run_split(to_recs(rows), [0, 1], [2, 3, 4], estimator="gbr", conformal="global")
    if m is None:
        out["run_split_by_domain"][label] = None; continue
    bb = bool(m["regret_kga"] < m["regret_adapt"] and m["regret_kga"] < m["regret_freeze"])
    fa_ok = bool(m["false_adapt"] <= 0.10)
    m["beats_both"] = bb; m["fa_le_alpha"] = fa_ok
    m["preregistered_win"] = bool(bb and fa_ok)
    out["run_split_by_domain"][label] = m

print(json.dumps(out, indent=2))
OUTP = f"{ROOT}/audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json"
json.dump(out, open(OUTP, "w"), indent=2)
print("\nSAVED:", OUTP)
