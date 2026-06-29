#!/usr/bin/env python3
"""D33: controlled multimodal corruption test of KGA (faithful to the sealed protocol
research_lock/CONTROLLED_MULTIMODAL_PROTOCOL_D33_v1.yaml). Report whatever it gives.
Honest scope: a win shows the mechanism works WHEN the named condition holds + is detectable;
it does NOT claim a natural benchmark exhibits the condition."""
from __future__ import annotations
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from torchvision import datasets

SEED = 20260615; ALPHA = 0.10
SIGMAS = [0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
K_BATCH = 10; N_BATCH = 500
rng = np.random.default_rng(SEED)

# --- data: MNIST left/right halves as two modalities ---
tr = datasets.MNIST(root="/tmp/mnist_test", train=True, download=True)
te = datasets.MNIST(root="/tmp/mnist_test", train=False, download=True)
Xtr = (tr.data.numpy().reshape(-1,28,28)/255.0); ytr = tr.targets.numpy()
Xte = (te.data.numpy().reshape(-1,28,28)/255.0); yte = te.targets.numpy()
idx = rng.permutation(len(Xtr))[:12000]; Xtr, ytr = Xtr[idx], ytr[idx]
def halves(X): return X[:,:,:14].reshape(len(X),-1), X[:,:,14:].reshape(len(X),-1)
Atr,Btr = halves(Xtr); Ate,Bte = halves(Xte)
mA = LogisticRegression(max_iter=200, n_jobs=-1).fit(Atr, ytr)
mB = LogisticRegression(max_iter=200, n_jobs=-1).fit(Btr, ytr)

def ent(P):
    P=np.clip(P,1e-9,1); return -(P*np.log(P)).sum(1)
def acc(P,y): return float((P.argmax(1)==y).mean())

# --- build per-condition records (z, true benefit, accs) ---
rows=[]
for s in SIGMAS:
    for k in range(K_BATCH):
        bi = rng.integers(0, len(Xte), N_BATCH)
        A, B, y = Ate[bi], Bte[bi].copy(), yte[bi]
        B = np.clip(B + rng.normal(0, s, B.shape), 0, 1)   # corrupt modality B
        pA = mA.predict_proba(A); pB = mB.predict_proba(B)
        pF = 0.5*(pA+pB)                                    # late fusion
        accA, accF = acc(pA,y), acc(pF,y)
        z = [float(pB.max(1).mean()), float(ent(pB).mean()),
             float((pA.argmax(1)!=pF.argmax(1)).mean())]    # label-free evidence
        rows.append(dict(sigma=s, z=z, accA=accA, accF=accF, benefit=accF-accA))

n=len(rows); Z=np.array([r["z"] for r in rows]); ben=np.array([r["benefit"] for r in rows])
accA=np.array([r["accA"] for r in rows]); accF=np.array([r["accF"] for r in rows])

# --- LOCO conformal certificate ---
kga=np.empty(n); dec=[]; n_adapt=0; fa=0
for i in range(n):
    tr_idx=[j for j in range(n) if j!=i]
    gb=GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)
    gb.fit(Z[tr_idx], ben[tr_idx])
    bhat=float(gb.predict(Z[i:i+1])[0])
    resid=np.abs(ben[tr_idx]-gb.predict(Z[tr_idx])); eps=float(np.quantile(resid,1-ALPHA))
    if bhat-eps>0: d_,a="ADAPT",accF[i]; n_adapt+=1; fa+=int(ben[i]<0)
    elif bhat+eps<0: d_,a="FREEZE",accA[i]
    else: d_,a="ABSTAIN",accA[i]
    kga[i]=a; dec.append(d_)

mk,mf,ma=float(kga.mean()),float(accF.mean()),float(accA.mean())
orc=float(np.mean(np.maximum(accF,accA)))
# paired bootstrap over conditions
bs=np.random.default_rng(SEED+1); B=10000; dkf=[];dka=[]
for _ in range(B):
    ii=bs.integers(0,n,n); dkf.append(kga[ii].mean()-accF[ii].mean()); dka.append(kga[ii].mean()-accA[ii].mean())
dkf=np.array(dkf);dka=np.array(dka)
from collections import Counter
P_gt_fuse=float((dkf>0).mean()); P_gt_A=float((dka>0).mean())
verdict = "STRONG" if (P_gt_fuse>=0.95 and P_gt_A>=0.95 and (fa/max(n_adapt,1))<=ALPHA) else \
          ("STANDS" if (P_gt_A>=0.95) else "HONEST_NEGATIVE")
out=dict(n_conditions=n, alpha=ALPHA, decisions=dict(Counter(dec)),
         mean_acc=dict(always_fuse=mf, always_A=ma, kga=mk, oracle=orc),
         kga_minus_fuse=mk-mf, kga_minus_A=mk-ma,
         P_kga_gt_fuse=P_gt_fuse, P_kga_gt_A=P_gt_A,
         ci_kga_minus_fuse=[float(np.quantile(dkf,.025)),float(np.quantile(dkf,.975))],
         ci_kga_minus_A=[float(np.quantile(dka,.025)),float(np.quantile(dka,.975))],
         n_adapt=n_adapt, false_adapt=fa, false_adapt_rate=fa/max(n_adapt,1), verdict=verdict)
od="experiments/kbound/results/controlled_multimodal_d33"; os.makedirs(od,exist_ok=True)
json.dump(out, open(od+"/results.json","w"), indent=2, default=float)
print(f"=== D33 controlled multimodal ({n} conditions, alpha={ALPHA}) ===")
print(f"decisions: {dict(Counter(dec))}")
print(f"mean acc: always-fuse={mf:.4f}  always-A(single)={ma:.4f}  KGA={mk:.4f}  oracle={orc:.4f}")
print(f"KGA-fuse={mk-mf:+.4f} (P={P_gt_fuse:.3f})   KGA-A={mk-ma:+.4f} (P={P_gt_A:.3f})")
print(f"false_adapt={fa}/{n_adapt} (rate={fa/max(n_adapt,1):.3f})")
print(f"VERDICT: {verdict}")
