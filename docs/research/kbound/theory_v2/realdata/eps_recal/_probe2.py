#!/usr/bin/env python3
"""Probe 2: reproduce reported eps_conformal (per candidate) + harm-AUC 0.912
using the canonical LOO GBR recipe from run_wilds_camelyon17.decide_kga."""
import json, numpy as np
from sklearn.ensemble import GradientBoostingRegressor

P = "/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"
d = json.load(open(P))
recs = d["records"]
det = d["detectability"]
ra = d["routing_a_single_candidate"]
ALPHA = 0.10

def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=0):
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=lr, subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr])
        Bhat[i] = m.predict(Z[i:i+1])[0]
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT",
                   np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, dec

# group records by candidate
from collections import defaultdict
by_cand = defaultdict(list)
for r in recs:
    by_cand[r["candidate"]].append(r)

# Reproduce per-candidate eps and assemble pooled Bhat for AUC over all 432 cells
all_Bhat = []; all_B = []
print("candidate           reported_eps   reproduced_eps   beats_both(rep)")
for cand in ["tent_online","tent_episodic","eata_online","eata_episodic","sar_online","sar_episodic"]:
    rs = by_cand[cand]
    Z = np.array([r["Z"] for r in rs], float)
    B = np.array([r["B"] for r in rs], float)
    Bhat, eps, dec = decide_kga(Z, B)
    rep = ra[cand]["kga"]["eps_conformal"]
    bb = ra[cand]["kga"]["beats_both"]
    print("%-18s  %.5f       %.5f          %s" % (cand, rep, eps, bb))
    all_Bhat.append(Bhat); all_B.append(B)

all_Bhat = np.concatenate(all_Bhat); all_B = np.concatenate(all_B)
# harm-AUC of -Bhat predicting (B<0)
def auc(score, label):
    pos = score[label==1]; neg = score[label==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    allv=np.concatenate([pos,neg]); r=np.argsort(np.argsort(allv))+1
    U=r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))
lab = (all_B < 0).astype(int)
print("\nPOOLED over 432 cells:")
print("  reported certificate_harm_AUC_negBhat:", det["certificate_harm_AUC_negBhat"])
print("  reproduced harm-AUC(-Bhat)           :", auc(-all_Bhat, lab))
print("  reported certificate_eps:", det["certificate_eps"])
print("  reproduced pooled eps   :", float(np.quantile(np.abs(all_Bhat-all_B), 1-ALPHA)))
print("  n_harmful B<0:", int(lab.sum()), "of", len(lab))
