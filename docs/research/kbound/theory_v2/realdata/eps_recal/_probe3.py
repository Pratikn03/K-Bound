#!/usr/bin/env python3
"""Probe 3: reproduce detectability.certificate_harm_AUC_negBhat (0.912) and
certificate_eps (0.0598). Hypothesis: single GBR fit over ALL 432 cells (LOO),
features = Z (and maybe candidate id). Try a few variants."""
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

import json, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

P = KB_REPO_ROOT + "/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"
d = json.load(open(P))
recs = d["records"]
det = d["detectability"]
ALPHA = 0.10
Z = np.array([r["Z"] for r in recs], float)        # 432 x 10
B = np.array([r["B"] for r in recs], float)
lab = (B < 0).astype(int)

def auc(score, label):
    pos=score[label==1]; neg=score[label==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    allv=np.concatenate([pos,neg]); r=np.argsort(np.argsort(allv))+1
    U=r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))

print("target AUC 0.91218, target eps 0.059781")

def loo_pooled(Z,B,ne,md,lr,seed=0,sub=0.8):
    N=len(B); Bhat=np.zeros(N)
    for i in range(N):
        tr=np.arange(N)!=i
        m=GradientBoostingRegressor(n_estimators=ne,max_depth=md,learning_rate=lr,subsample=sub,random_state=seed)
        m.fit(Z[tr],B[tr]); Bhat[i]=m.predict(Z[i:i+1])[0]
    return Bhat

# Variant A: pooled LOO, same hyperparams as decide_kga (250,2,0.05)
Bhat=loo_pooled(Z,B,250,2,0.05)
print("A pooled-LOO(250,2,.05): AUC=%.4f eps=%.4f"%(auc(-Bhat,lab), np.quantile(np.abs(Bhat-B),1-ALPHA)))

# Variant B: in-sample (fit all, predict all) — overfit, high AUC small eps
m=GradientBoostingRegressor(n_estimators=250,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)
m.fit(Z,B); Bin=m.predict(Z)
print("B in-sample(250,2,.05):  AUC=%.4f eps=%.4f"%(auc(-Bin,lab), np.quantile(np.abs(Bin-B),1-ALPHA)))

# Variant C: 5-fold CV pooled
kf=KFold(n_splits=5,shuffle=True,random_state=0); Bcv=np.zeros(len(B))
for tr,te in kf.split(Z):
    m=GradientBoostingRegressor(n_estimators=250,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)
    m.fit(Z[tr],B[tr]); Bcv[te]=m.predict(Z[te])
print("C 5fold(250,2,.05):      AUC=%.4f eps=%.4f"%(auc(-Bcv,lab), np.quantile(np.abs(Bcv-B),1-ALPHA)))

# Variant D: include candidate one-hot in features, pooled LOO
cand=[r["candidate"] for r in recs]; names=sorted(set(cand))
onehot=np.array([[1.0 if c==n else 0.0 for n in names] for c in cand])
Zc=np.hstack([Z,onehot])
Bhat2=loo_pooled(Zc,B,250,2,0.05)
print("D pooled-LOO+candOH:     AUC=%.4f eps=%.4f"%(auc(-Bhat2,lab), np.quantile(np.abs(Bhat2-B),1-ALPHA)))
