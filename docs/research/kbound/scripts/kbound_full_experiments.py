# SUPERSEDED RULE -- EXPLORATORY v1 CODE (defect D10).
# This script computes its certificate radius as ``np.quantile(|Bhat - B|, 1 - alpha)``,
# numpy's *linearly interpolated* quantile.  That is NOT the rule the paper declares.
# The declared rule is the exact split-conformal rank quantile
# ``eps = r_(k)``, ``k = ceil((n + 1)(1 - alpha))``, leave-one-out-of-pool, with ``+inf``
# => ABSTAIN when ``k > n`` -- implemented once in ``kga/certificate.py`` and reached from
# ``docs/research/kbound/scripts/kbound_decide.py``.
#
# This file is retained unconverted on purpose: it is v1/exploratory code, no promoted
# number in the paper comes from it, and its archived JSON outputs were produced under the
# interpolated rule, so converting it in place would silently make those outputs
# irreproducible.  Do not cite any number it prints, and do not copy its radius line.
# It is on the named allowlist in ``tests/test_one_radius_rule.py``; adding a new
# interpolated radius anywhere else fails that test.

"""
K-Bound consolidated experiments: statistical rigor, ablations, and the
regression covariate-shift track (Theorem 4). One file, three parts.

  python kbound_full_experiments.py rigor       # multi-seed + paired t-tests (mixed regime)
  python kbound_full_experiments.py ablation    # evidence-drop, alpha sweep, batch-size sweep
  python kbound_full_experiments.py regression  # Theorem-4 covariate-shift validation

Reads the migrated score archive at kbound_paper/data/score_archive (self-contained).
All numbers produced by the run; nothing fabricated.
"""
import sys, os, glob, json
import numpy as np
from scipy.stats import ks_2samp, rankdata, ttest_rel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error

_REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCH=os.path.join(_REPO,"experiments","elara_u","score_archive")
RES=os.path.join(_REPO,"experiments","kbound","results"); FIG=os.path.join(_REPO,"docs","research","kbound","figures")
os.makedirs(RES,exist_ok=True); os.makedirs(FIG,exist_ok=True)

def rank_norm(S):
    R=np.empty_like(S,dtype=float)
    for j in range(S.shape[1]): R[:,j]=(rankdata(S[:,j])-1)/(len(S)-1)
    return R
def auc(y,s): return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

FEAT=["val_max","val_gap","val_mean","ks_mean","ks_max","disagree","val_std","anomaly_rate","log_ntest"]
def evidence(Sval,St,val_auc,yval):
    vs=np.sort(val_auc)[::-1]
    ks=[ks_2samp(Sval[:,j],St[:,j]).statistic for j in range(Sval.shape[1])]
    Rt=rank_norm(St); C=np.corrcoef(Rt.T); iu=np.triu_indices_from(C,k=1)
    return [float(vs[0]),float(vs[0]-vs[1]),float(np.mean(val_auc)),float(np.mean(ks)),
            float(np.max(ks)),float(1-np.nanmean(C[iu])),float(np.std(val_auc)),
            float(np.mean(yval)),float(np.log(len(St)))], ks
def gated_average(St,ks,c=0.05):
    w=1.0/(np.asarray(ks)+c); w/=w.sum(); return rank_norm(St)@w

def build_mixed(seed, ntest_cap=None):
    """Build the 3-condition mixed regime (clean/detectable/covert) with a given seed."""
    rng=np.random.default_rng(seed)
    rows=[]
    for f in sorted(glob.glob(os.path.join(ARCH,"*.npz"))):
        d=np.load(f,allow_pickle=True)
        Sval,yval,Stest,ytest=d["Sval"],d["yval"],d["Stest"],d["ytest"]; val_auc=d["val_auc"]
        if len(np.unique(ytest))<2: continue
        if ntest_cap and len(ytest)>ntest_cap:
            idx=rng.choice(len(ytest),ntest_cap,replace=False); Stest=Stest[idx]; ytest=ytest[idx]
        j0=int(np.argmax(val_auc))
        for cond in ("clean","detectable","covert"):
            St=Stest.copy()
            if cond=="detectable":
                col=St[:,j0]; St[:,j0]=rng.normal(col.mean()+3*col.std()+1,col.std()*2+1e-6,len(col))
            elif cond=="covert":
                St[:,j0]=rng.permutation(St[:,j0])
            Z,ks=evidence(Sval,St,val_auc,yval)
            a0=auc(ytest,St[:,j0]); aa=auc(ytest,gated_average(St,ks))
            if np.isnan(a0) or np.isnan(aa): continue
            rows.append((cond,Z,float(aa-a0),float(a0),float(aa)))
    cond=np.array([r[0] for r in rows]); X=np.array([r[1] for r in rows])
    B=np.array([r[2] for r in rows]); a0=np.array([r[3] for r in rows]); aa=np.array([r[4] for r in rows])
    return cond,X,B,a0,aa

def kfold_bhat(X,B,seed,k=5,feat_mask=None):
    if feat_mask is not None: X=X[:,feat_mask]
    rng=np.random.default_rng(seed); idx=rng.permutation(len(B)); folds=np.array_split(idx,k)
    Bhat=np.zeros(len(B))
    for i in range(k):
        te=folds[i]; tr=np.setdiff1d(idx,te)
        m=GradientBoostingRegressor(n_estimators=150,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=seed)
        m.fit(X[tr],B[tr]); Bhat[te]=m.predict(X[te])
    return Bhat

def decide(Bhat,B,alpha=0.10):
    eps=float(np.quantile(np.abs(Bhat-B),1-alpha))
    dec=np.where(Bhat-eps>0,"ADAPT",np.where(Bhat+eps<0,"FREEZE","ABSTAIN"))
    return dec,eps

def policy_auc(dec,a0,aa):
    pol=np.where(dec=="ADAPT",aa,a0); return pol

# ---------------- PART 1: RIGOR (multi-seed + t-tests) ----------------
def run_rigor(seeds=range(8)):
    rec={"always_adapt":[],"always_freeze":[],"K_Bound":[],"oracle":[],
         "false_adapt":[],"adapt_precision":[],"coverage":[]}
    per_seed_pol={"K_Bound":[],"always_adapt":[],"always_freeze":[]}
    for s in seeds:
        cond,X,B,a0,aa=build_mixed(s)
        Bhat=kfold_bhat(X,B,s); dec,eps=decide(Bhat,B)
        pol=policy_auc(dec,a0,aa); oracle=np.maximum(a0,aa); adapt=dec=="ADAPT"
        rec["always_adapt"].append(aa.mean()); rec["always_freeze"].append(a0.mean())
        rec["K_Bound"].append(pol.mean()); rec["oracle"].append(oracle.mean())
        rec["false_adapt"].append(float(np.mean(B[adapt]<0)) if adapt.any() else 0.0)
        rec["adapt_precision"].append(float(np.mean(B[adapt]>0)) if adapt.any() else 0.0)
        rec["coverage"].append(float(np.mean(dec!="ABSTAIN")))
        per_seed_pol["K_Bound"].append(pol.mean()); per_seed_pol["always_adapt"].append(aa.mean())
        per_seed_pol["always_freeze"].append(a0.mean())
    def ms(x): return [float(np.mean(x)),float(np.std(x))]
    kb=np.array(per_seed_pol["K_Bound"]); af=np.array(per_seed_pol["always_adapt"]); ff=np.array(per_seed_pol["always_freeze"])
    out={"n_seeds":len(list(seeds)),
         "mean_std":{k:ms(v) for k,v in rec.items()},
         "paired_ttest_KBound_vs_always_freeze":{"t":float(ttest_rel(kb,ff).statistic),"p":float(ttest_rel(kb,ff).pvalue)},
         "paired_ttest_KBound_vs_always_adapt":{"t":float(ttest_rel(kb,af).statistic),"p":float(ttest_rel(kb,af).pvalue)}}
    json.dump(out,open(os.path.join(RES,"rigor_multiseed.json"),"w"),indent=2)
    print(json.dumps(out,indent=2))

# ---------------- PART 2: ABLATIONS ----------------
def run_ablation():
    cond,X,B,a0,aa=build_mixed(0); oracle=np.maximum(a0,aa)
    # (a) evidence-drop
    full=kfold_bhat(X,B,0); dec,_=decide(full,B); base_fa=float(np.mean(B[dec=="ADAPT"]<0)); base_reg=float((oracle-policy_auc(dec,a0,aa)).mean())
    drop={}
    for j,name in enumerate(FEAT):
        mask=[i for i in range(X.shape[1]) if i!=j]
        bh=kfold_bhat(X,B,0,feat_mask=mask); dd,_=decide(bh,B)
        drop[name]={"false_adapt":float(np.mean(B[dd=="ADAPT"]<0)) if (dd=="ADAPT").any() else 0.0,
                    "regret":float((oracle-policy_auc(dd,a0,aa)).mean()),
                    "coverage":float(np.mean(dd!="ABSTAIN"))}
    # (b) alpha sweep
    alpha={}
    for a in [0.01,0.05,0.10,0.20]:
        dd,eps=decide(full,B,alpha=a)
        adapt=dd=="ADAPT"
        alpha[str(a)]={"eps":eps,"coverage":float(np.mean(dd!="ABSTAIN")),
                       "false_adapt":float(np.mean(B[adapt]<0)) if adapt.any() else 0.0}
    # (c) batch-size sweep
    batch={}
    for n in [16,32,64,128,256]:
        c2,X2,B2,a02,aa2=build_mixed(0,ntest_cap=n)
        bh=kfold_bhat(X2,B2,0); dd,eps=decide(bh,B2)
        batch[str(n)]={"eps":eps,"coverage":float(np.mean(dd!="ABSTAIN"))}
    out={"baseline":{"false_adapt":base_fa,"regret":base_reg,"features":FEAT},
         "evidence_drop":drop,"alpha_sweep":alpha,"batchsize_sweep":batch}
    json.dump(out,open(os.path.join(RES,"ablations.json"),"w"),indent=2)
    print(json.dumps(out,indent=2))
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5,3.6))
    names=FEAT; reg=[drop[n]["regret"] for n in names]
    plt.bar(names,reg,color="#457b9d"); plt.axhline(base_reg,color="#e76f51",ls="--",label=f"all features ({base_reg:.3f})")
    plt.ylabel("regret vs oracle (drop feature)"); plt.xticks(rotation=45,ha="right",fontsize=8)
    plt.title("Evidence-component ablation"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"fig_ablation_evidence.png"),dpi=130)
    a_keys=list(alpha.keys()); cov=[alpha[k]["coverage"] for k in a_keys]; fa=[alpha[k]["false_adapt"] for k in a_keys]
    plt.figure(figsize=(5,3.8)); plt.plot(cov,fa,"o-",color="#2a9d8f")
    for k,x,y in zip(a_keys,cov,fa): plt.annotate(f"α={k}",(x,y),fontsize=8)
    plt.xlabel("coverage (non-abstain)"); plt.ylabel("false-adapt rate"); plt.title("Coverage vs safety (α sweep)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_alpha_coverage.png"),dpi=130)
    print("saved ablation figs")

# ---------------- PART 3: REGRESSION COVARIATE SHIFT (Theorem 4) ----------------
def run_regression(seeds=range(6)):
    rng0=np.random.default_rng(0); d=5; beta=rng0.normal(size=d)
    rows=[]  # (Z, B, mse0, mse_a, kind)
    for s in seeds:
        rng=np.random.default_rng(100+s)
        # source
        Xs=rng.normal(size=(2000,d)); ys=Xs@beta+rng.normal(0,0.5,2000)
        f0=LinearRegression().fit(Xs,ys)
        for mag in np.linspace(0.0,4.0,12):
            shift=np.zeros(d); shift[0]=mag
            Xt=rng.normal(size=(1500,d))+shift; yt=Xt@beta+rng.normal(0,0.5,1500)  # P(Y|X) invariant
            # importance weights via Gaussian ratio (known shift family); clip for stability
            logr=(Xt@shift)-0.5*(shift@shift)  # N(shift,I)/N(0,I)
            w=np.exp(np.clip(logr,-10,10)); w/=w.mean()
            fa=LinearRegression().fit(Xs,ys,sample_weight=None)  # base; IW used in fit below
            faw=LinearRegression().fit(Xs,ys,sample_weight=np.exp(np.clip((Xs@shift)-0.5*(shift@shift),-10,10)))
            mse0=mean_squared_error(yt,f0.predict(Xt)); msea=mean_squared_error(yt,faw.predict(Xt))
            B=mse0-msea  # benefit>0 means adapt (IW) helps
            # label-free Z: covariate KS drift (dim0), weight variance, support-overlap proxy
            ks0=ks_2samp(Xs[:,0],Xt[:,0]).statistic
            wvar=float(np.var(w)); overlap=float(np.mean((Xt[:,0]>Xs[:,0].min())&(Xt[:,0]<Xs[:,0].max())))
            Z=[ks0,wvar,overlap,float(mag)]; rows.append((Z,float(B),float(mse0),float(msea)))
    X=np.array([r[0] for r in rows]); B=np.array([r[1] for r in rows])
    m0=np.array([r[2] for r in rows]); ma=np.array([r[3] for r in rows])
    Bhat=kfold_bhat(X,B,0); dec,eps=decide(Bhat,B,alpha=0.1); adapt=dec=="ADAPT"
    pol=np.where(adapt,ma,m0)  # MSE: lower better; ADAPT->fa else frozen
    oracle=np.minimum(m0,ma)
    out={"n_instances":len(B),"eps":eps,
         "mean_B(mse0-msea)":float(B.mean()),"harmful_rate_B<0":float(np.mean(B<0)),
         "decision_counts":{d2:int((dec==d2).sum()) for d2 in ["ADAPT","FREEZE","ABSTAIN"]},
         "adapt_precision_B>0":float(np.mean(B[adapt]>0)) if adapt.any() else None,
         "false_adapt_rate_B<0":float(np.mean(B[adapt]<0)) if adapt.any() else None,
         "mean_MSE":{"always_adapt(IW)":float(ma.mean()),"always_freeze":float(m0.mean()),
                     "K_Bound":float(pol.mean()),"oracle":float(oracle.mean())}}
    json.dump(out,open(os.path.join(RES,"regression_covariate.json"),"w"),indent=2)
    print(json.dumps(out,indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5.6,3.8))
    labs=["always-adapt\n(IW reg)","always-freeze","K-Bound","oracle"]
    vals=[ma.mean(),m0.mean(),pol.mean(),oracle.mean()]
    plt.bar(labs,vals,color=["#e76f51","#457b9d","#2a9d8f","#999999"])
    for i,v in enumerate(vals): plt.text(i,v+.005,f"{v:.3f}",ha="center",fontsize=9)
    plt.ylabel("mean target MSE (lower better)"); plt.title("Regression covariate shift (Theorem 4)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_regression.png"),dpi=130); print("saved regression fig")

# ---------------- PART 4: CLEAN NON-IDENTIFIABILITY WITNESS (Theorem 1) ----------------
def run_witness(M=300, n=400):
    """Two worlds with PROVABLY identical Z-law (Z depends only on X) but opposite
    benefit sign. Show Z is statistically indistinguishable across worlds and KGA abstains."""
    from scipy.stats import skew
    rng=np.random.default_rng(0)
    rows=[]
    for _ in range(M):
        world=int(rng.integers(0,2))            # hidden; NOT in Z
        X=rng.normal(0,1,n)
        f0=(X>0).astype(int); fa=(X<0).astype(int)
        Y=(X>0).astype(int) if world==0 else (X<0).astype(int)
        a0=float(np.mean(f0==Y)); aa=float(np.mean(fa==Y)); B=aa-a0
        # Z = functions of X and the fixed maps ONLY (no labels) -> identical law across worlds
        Z=[float(np.mean(X>0)), float(np.std(X)),
           float(ks_2samp(X,rng.normal(0,1,n)).statistic),
           float(np.mean(f0)), float(skew(X))]
        rows.append((world,Z,B,a0,aa))
    world=np.array([r[0] for r in rows]); X=np.array([r[1] for r in rows])
    B=np.array([r[2] for r in rows]); a0=np.array([r[3] for r in rows]); aa=np.array([r[4] for r in rows])
    # (a) verify Z-law indistinguishable across worlds: per-feature KS p-values
    names=["frac_pos","std","ks_vs_ref","mean_f0","skew"]
    zks={names[j]: float(ks_2samp(X[world==0,j], X[world==1,j]).pvalue) for j in range(X.shape[1])}
    # (b) KGA on (Z,B): estimator cannot separate -> abstain
    Bhat=kfold_bhat(X,B,0); dec,eps=decide(Bhat,B,alpha=0.1)
    abst=dec=="ABSTAIN"
    # forced-commit regret: if forced to pick adapt/freeze by sign of Bhat
    forced=np.where(Bhat>0,aa,a0); oracle=np.maximum(a0,aa)
    out={"M":M,"mean_abs_true_benefit":float(np.mean(np.abs(B))),
         "world0_mean_B":float(B[world==0].mean()),"world1_mean_B":float(B[world==1].mean()),
         "Z_indistinguishable_KS_pvalues":zks,
         "all_Z_features_p>0.05":bool(all(p>0.05 for p in zks.values())),
         "abstain_rate":float(np.mean(abst)),
         "eps":eps,"mean_|Bhat|":float(np.mean(np.abs(Bhat))),
         "forced_commit_regret_vs_oracle":float((oracle-forced).mean()),
         "abstain_regret_vs_oracle(default_freeze)":float((oracle-np.where(dec=="ADAPT",aa,a0)).mean())}
    json.dump(out,open(os.path.join(RES,"witness_clean.json"),"w"),indent=2)
    print(json.dumps(out,indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.scatter(Bhat[world==0],B[world==0],s=14,c="#457b9d",label="world 1 (adapt hurts)")
    plt.scatter(Bhat[world==1],B[world==1],s=14,c="#e76f51",label="world 2 (adapt helps)")
    plt.axvspan(-eps,eps,color="#cccccc",alpha=.4,label=f"abstain band ±{eps:.2f}")
    plt.axhline(0,color="k",lw=.6); plt.axvline(0,color="k",lw=.6)
    plt.xlabel("estimated benefit B̂(Z)  (label-free)"); plt.ylabel("true benefit B")
    plt.title("Clean non-identifiability: identical Z-law, opposite truth → abstain")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"fig_witness_clean.png"),dpi=130); print("saved witness fig")

if __name__=="__main__":
    part=sys.argv[1] if len(sys.argv)>1 else "rigor"
    {"rigor":run_rigor,"ablation":run_ablation,"regression":run_regression,"witness":run_witness}[part]()
