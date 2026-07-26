#!/usr/bin/env python3
"""Probe: lock down field semantics + reproduce harm-AUC 0.912 from the file."""
# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). This file previously hard-coded a
# --- Cowork *session sandbox* mount, which is worse than a
# --- home directory: it is valid only inside one ephemeral container.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

import json, itertools
import numpy as np

P = KB_REPO_ROOT + "/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"
d = json.load(open(P))
recs = d["records"]
conds = d["conditions"]
det = d["detectability"]
print("n_records", len(recs), "n_conditions", len(conds))
print("reported certificate_harm_AUC_negBhat", det["certificate_harm_AUC_negBhat"])
print("reported certificate_eps", det["certificate_eps"])
print("reported n_harmful", det["n_harmful"], "base_rate", det["base_rate_harmful"], "mean_B", det["mean_B"])

# candidate order in conditions.cand_names (index 0 = freeze_f0)
cand_names = conds[0]["cand_names"]
print("cand_names", cand_names)

# Build per-(condition,candidate) cells: align records B to conditions b_hat.
# key a condition by (seed,domain,comp,regime,aggr)
def ckey(x):
    return (x["seed"], x["domain"], x["comp"], x["regime"], x["aggr"])

cond_by_key = {ckey(c): c for c in conds}
# records carry candidate name; B = aa - a0 (true per-candidate benefit vs f0)
# verify B == aa - a0
bad = 0
for r in recs[:50]:
    if abs(r["B"] - (r["aa"] - r["a0"])) > 1e-9:
        bad += 1
print("records: B==aa-a0 mismatches in first 50:", bad)

# For each record, find its condition's b_hat for that candidate.
# cand index in b_hat: position of candidate in cand_names
cand_idx = {n: i for i, n in enumerate(cand_names)}
scores = []   # -b_hat (certificate predicted harm score)
labels = []   # 1 if truly harmful (B<0)
Bvals = []
Bhat_adv = []
n_join_fail = 0
for r in recs:
    c = cond_by_key.get(ckey(r))
    if c is None:
        n_join_fail += 1
        continue
    cand = r["candidate"]
    if cand not in cand_idx:
        n_join_fail += 1
        continue
    bh = c["route"]["b_hat"][cand_idx[cand]]
    scores.append(-bh)            # -Bhat
    labels.append(1 if r["B"] < 0 else 0)
    Bvals.append(r["B"])
    Bhat_adv.append(bh)
print("join failures:", n_join_fail, "cells used:", len(scores))

scores = np.array(scores); labels = np.array(labels); Bvals = np.array(Bvals); Bhat_adv = np.array(Bhat_adv)
print("n_harmful (B<0) from join:", int(labels.sum()), "base rate:", labels.mean())

def auc(score, label):
    # rank-based AUC
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    # average ties
    # simple Mann-Whitney
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # U statistic
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1
    rp = r[:len(pos)].sum()
    U = rp - len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))

print("REPRODUCED harm-AUC(-Bhat):", auc(scores, labels))

# relationship between Bhat_adv (advantage scale) and B (accuracy-diff scale)
print("corr(Bhat_adv, B):", np.corrcoef(Bhat_adv, Bvals)[0,1])
print("Bhat_adv range:", Bhat_adv.min(), Bhat_adv.max())
print("B range:", Bvals.min(), Bvals.max())
# slope B ~ Bhat_adv
A = np.vstack([Bhat_adv, np.ones_like(Bhat_adv)]).T
sl, ic = np.linalg.lstsq(A, Bvals, rcond=None)[0]
print("B ~= %.4f * Bhat_adv + %.4f" % (sl, ic))
