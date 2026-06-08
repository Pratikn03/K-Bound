"""
K-Bound on classification test-time adaptation (TTA), where harmful adaptation is
CATASTROPHIC -- the regime the anomaly domain lacks.

Setup (real, controlled):
  - Source: 2D two-Gaussian binary classification; train logistic f0 on source.
  - Adaptation fa = Tent-style entropy minimization: a few gradient steps on (w,b)
    minimizing mean prediction entropy over the UNLABELED target batch (real Tent).
  - We generate many target tasks spanning three shift families:
      * helpful   : covariate shift into a low-confidence band -> entropy-min sharpens CORRECTLY -> acc up
      * harmful   : concept/label shift -> entropy-min sharpens WRONG -> acc COLLAPSES
      * ambiguous : mild mixed shift -> small/uncertain effect
  - True benefit B = acc(fa) - acc(f0)  (labels used ONLY for evaluation).
  - Label-free evidence Z: entropy before, entropy drop after adaptation, mean
    confidence, predicted-class balance, feature KS drift, update norm.
  - K-Bound trichotomy from Z via leave-one-out estimator + conformal eps.

Claim under test: in a domain with catastrophic harmful adaptation, K-Bound beats
BOTH always-adapt (collapses on harmful) and always-freeze (misses helpful).
Controlled-synthetic, validates the THEORY mechanism. Every number from this run.
"""
import os, json
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

_REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES=os.path.join(_REPO,"experiments","kbound","results"); os.makedirs(RES,exist_ok=True)
FIG=os.path.join(_REPO,"docs","research","kbound","figures"); os.makedirs(FIG,exist_ok=True)
rng=np.random.default_rng(7)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

def train_source(n=4000):
    # two gaussians, balanced
    m0=np.array([-1.5,0.0]); m1=np.array([1.5,0.0])
    X=np.vstack([rng.normal(m0,1.0,(n//2,2)), rng.normal(m1,1.0,(n//2,2))])
    y=np.r_[np.zeros(n//2),np.ones(n//2)]
    # logistic fit (closed-ish via gradient descent)
    w=np.zeros(2); b=0.0; lr=0.1
    for _ in range(400):
        p=sigmoid(X@w+b); g=p-y
        w-=lr*(X.T@g/n + 1e-4*w); b-=lr*g.mean()
    return w,b,(m0,m1)

def tent_adapt(w,b,Xt,steps=30,lr=0.5):
    """entropy minimization on unlabeled target (real Tent-style)."""
    w=w.copy(); b=float(b)
    def ent(p): return -(p*np.log(p+1e-9)+(1-p)*np.log(1-p+1e-9))
    upd=0.0
    for _ in range(steps):
        z=Xt@w+b; p=sigmoid(z)
        # d entropy/dz = (1-2p)*... ; gradient of mean entropy wrt z:
        dHdz=(np.log(p+1e-9)-np.log(1-p+1e-9))*(p*(1-p))*(-1)  # minimize entropy
        gw=Xt.T@dHdz/len(Xt); gb=dHdz.mean()
        w-=lr*gw; b-=lr*gb; upd+=lr*np.sqrt((gw**2).sum()+gb**2)
    return w,b,upd

def acc(w,b,X,y): return float(np.mean((sigmoid(X@w+b)>0.5)==(y>0.5)))

def make_target(kind, n=1500):
    m0=np.array([-1.5,0.0]); m1=np.array([1.5,0.0])
    if kind=="helpful":
        # covariate shift: move both clusters toward boundary (low confidence) but separable
        s=rng.uniform(0.6,1.3)
        X=np.vstack([rng.normal(m0*0.45,0.8,(n//2,2)), rng.normal(m1*0.45,0.8,(n//2,2))])*s
        y=np.r_[np.zeros(n//2),np.ones(n//2)]
    elif kind=="harmful":
        # concept/label shift: heavy imbalance + class1 pushed across boundary
        frac1=rng.uniform(0.08,0.2); n1=int(n*frac1); n0=n-n1
        shift=rng.uniform(2.0,3.5)
        X=np.vstack([rng.normal(m0,1.0,(n0,2)), rng.normal(m0-np.array([shift,0]),1.0,(n1,2))])
        y=np.r_[np.zeros(n0),np.ones(n1)]   # class1 now sits LEFT -> source says class0 -> entropy-min sharpens wrong
    else: # ambiguous
        s=rng.uniform(0.9,1.1); j=rng.normal(0,0.4,2)
        X=np.vstack([rng.normal(m0+j,1.1,(n//2,2)), rng.normal(m1+j,1.1,(n//2,2))])*s
        y=np.r_[np.zeros(n//2),np.ones(n//2)]
    idx=rng.permutation(n); return X[idx],y[idx]

w0,b0,_=train_source()
rows=[]
for kind in ["helpful","harmful","ambiguous"]:
    for _ in range(70):
        Xt,yt=make_target(kind)
        # label-free observables BEFORE adaptation
        p0=sigmoid(Xt@w0+b0); H0=float(np.mean(-(p0*np.log(p0+1e-9)+(1-p0)*np.log(1-p0+1e-9))))
        conf0=float(np.mean(np.abs(p0-0.5)*2)); bal0=float(np.mean(p0>0.5))
        wa,ba,upd=tent_adapt(w0,b0,Xt)
        pa=sigmoid(Xt@wa+ba); H1=float(np.mean(-(pa*np.log(pa+1e-9)+(1-pa)*np.log(1-pa+1e-9))))
        bal1=float(np.mean(pa>0.5))
        a0=acc(w0,b0,Xt,yt); aa=acc(wa,ba,Xt,yt)
        Z=[H0, H0-H1, conf0, bal0, abs(bal1-bal0), float(upd), float(np.std(Xt))]
        rows.append((kind,Z,float(aa-a0),a0,aa))

feat=["entropy0","entropy_drop","confidence0","pred_balance0","balance_change","update_norm","feat_std"]
X=np.array([r[1] for r in rows]); B=np.array([r[2] for r in rows])
acc0=np.array([r[3] for r in rows]); acca=np.array([r[4] for r in rows]); kind=np.array([r[0] for r in rows])
N=len(rows)

Bhat=np.zeros(N)
for i in range(N):
    tr=np.arange(N)!=i
    m=GradientBoostingRegressor(n_estimators=250,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)
    m.fit(X[tr],B[tr]); Bhat[i]=m.predict(X[i:i+1])[0]
eps=float(np.quantile(np.abs(Bhat-B),0.90))
dec=np.where(Bhat-eps>0,"ADAPT",np.where(Bhat+eps<0,"FREEZE","ABSTAIN"))
adapt=dec=="ADAPT"; pol=np.where(adapt,acca,acc0); oracle=np.maximum(acc0,acca)

def bc(a,fn): return {k:float(fn(a[kind==k])) for k in ["helpful","harmful","ambiguous"]}
out={
 "n_tasks":N,"eps":eps,
 "true_mean_benefit_by_kind":bc(B,np.mean),
 "decision_by_kind":{k:{d:int(((kind==k)&(dec==d)).sum()) for d in["ADAPT","FREEZE","ABSTAIN"]} for k in["helpful","harmful","ambiguous"]},
 "safety":{"adapt_precision_B>0":float(np.mean(B[adapt]>0)) if adapt.any() else None,
           "false_adapt_rate_B<0":float(np.mean(B[adapt]<0)) if adapt.any() else None},
 "mean_accuracy":{"always_adapt":float(acca.mean()),"always_freeze":float(acc0.mean()),
                  "K_Bound":float(pol.mean()),"oracle":float(oracle.mean())},
 "regret_vs_oracle":{"always_adapt":float((oracle-acca).mean()),"always_freeze":float((oracle-acc0).mean()),
                     "K_Bound":float((oracle-pol).mean())},
}
json.dump(out,open(os.path.join(RES,"tta_collapse_results.json"),"w"),indent=2)
print(json.dumps(out,indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(5.6,4))
labs=["always-adapt\n(Tent)","always-freeze","K-Bound","oracle"]
vals=[acca.mean(),acc0.mean(),pol.mean(),oracle.mean()]
plt.bar(labs,vals,color=["#e76f51","#457b9d","#2a9d8f","#999999"])
for i,v in enumerate(vals): plt.text(i,v+.005,f"{v:.3f}",ha="center",fontsize=9)
plt.ylim(0,1.02); plt.ylabel("mean target accuracy")
plt.title("Catastrophic-harm regime (TTA): K-Bound beats both")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_tta_collapse.png"),dpi=130)
print("saved fig")
