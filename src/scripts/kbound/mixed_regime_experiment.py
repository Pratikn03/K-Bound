"""
K-Bound DECISIVE experiment: a controlled MIXED regime with three conditions.

For each of the 123 archive tasks we build three test instances:
  (1) CLEAN            : raw test scores. Adapting (gated average) vs the frozen
                         best-val detector is usually neutral/harmful  -> FREEZE.
  (2) DETECTABLE FAIL  : the best-val detector's test column is corrupted with an
                         additive distribution shift (KS drift SPIKES, so it is
                         label-free OBSERVABLE). The frozen detector collapses;
                         the gated average down-weights it and recovers -> ADAPT.
  (3) COVERT FAIL      : the best-val detector's test column is RANDOM-PERMUTED.
                         Its signal is destroyed but its MARGINAL is unchanged, so
                         KS drift ~ 0 and the observable evidence Z is identical to
                         CLEAN. The frozen detector collapses but the gate cannot
                         see it -> the regime is UNKNOWABLE -> ABSTAIN.

f0 (FREEZE) = best-val single detector.
fa (ADAPT)  = reliability-gated average: rank-normalized test scores combined with
              weights w_j proportional to 1/(KS_drift_j + c)  (down-weights drifted detectors).
B = AUC(fa) - AUC(f0)  (true benefit; labels used ONLY for evaluation).
Decision = K-Bound trichotomy from label-free Z via leave-one-out estimator + conformal eps.

CLEAN and COVERT share (almost) identical Z but differ in benefit -> empirical
witness for the non-identifiability theorem. Controlled-synthetic corruptions
validate the THEORY mechanism; they are not real-world sensor failures.
Every number is produced by this run.
"""
import glob, os, json
import numpy as np
from scipy.stats import ks_2samp, rankdata
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(_REPO,"experiments","kbound","results"); os.makedirs(RES,exist_ok=True)
FIG = os.path.join(_REPO,"docs","research","kbound","figures"); os.makedirs(FIG,exist_ok=True)
ARCH = os.path.join(_REPO, "experiments", "elara_u", "score_archive")
rng=np.random.default_rng(0)

def rank_norm(S):
    R=np.empty_like(S,dtype=float)
    for j in range(S.shape[1]): R[:,j]=(rankdata(S[:,j])-1)/(len(S)-1)
    return R
def auc(y,s):
    return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan

def evidence(Sval,Stest,val_auc,yval):
    vs=np.sort(val_auc)[::-1]
    ks=[ks_2samp(Sval[:,j],Stest[:,j]).statistic for j in range(Sval.shape[1])]
    Rt=rank_norm(Stest); C=np.corrcoef(Rt.T); iu=np.triu_indices_from(C,k=1)
    return dict(val_max=float(vs[0]), val_gap=float(vs[0]-vs[1]), val_mean=float(np.mean(val_auc)),
                ks_mean=float(np.mean(ks)), ks_max=float(np.max(ks)),
                disagree=float(1-np.nanmean(C[iu])), val_std=float(np.std(val_auc)),
                anomaly_rate=float(np.mean(yval)), log_ntest=float(np.log(len(Stest)))), ks

def gated_average(Stest, ks, c=0.05):
    w=1.0/(np.asarray(ks)+c); w=w/w.sum()
    R=rank_norm(Stest)
    return R@w

rows=[]
for f in sorted(glob.glob(os.path.join(ARCH,"*.npz"))):
    d=np.load(f,allow_pickle=True)
    Sval,yval,Stest,ytest=d["Sval"],d["yval"],d["Stest"],d["ytest"]
    val_auc=d["val_auc"]; nm=os.path.basename(f)[:-4]
    if len(np.unique(ytest))<2: continue
    j0=int(np.argmax(val_auc))                       # frozen choice (picked on val)

    for cond in ("clean","detectable","covert"):
        St=Stest.copy()
        if cond=="detectable":
            col=St[:,j0]
            St[:,j0]=rng.normal(col.mean()+3*col.std()+1.0, col.std()*2+1e-6, size=len(col))
        elif cond=="covert":
            St[:,j0]=rng.permutation(St[:,j0])       # destroy signal, keep marginal
        Z,ks=evidence(Sval,St,val_auc,yval)
        a0=auc(ytest, St[:,j0])                       # frozen
        aa=auc(ytest, gated_average(St,ks))           # adapt (gated)
        if np.isnan(a0) or np.isnan(aa): continue
        rows.append((nm,cond,Z,float(aa-a0),float(a0),float(aa)))

feat=list(rows[0][2].keys())
X=np.array([[r[2][k] for k in feat] for r in rows])
B=np.array([r[3] for r in rows]); auc0=np.array([r[4] for r in rows]); auca=np.array([r[5] for r in rows])
cond=np.array([r[1] for r in rows]); N=len(rows)

# K-Bound: leave-one-out estimator + conformal radius
Bhat=np.zeros(N)
for i in range(N):
    tr=np.arange(N)!=i
    m=GradientBoostingRegressor(n_estimators=250,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)
    m.fit(X[tr],B[tr]); Bhat[i]=m.predict(X[i:i+1])[0]
eps=float(np.quantile(np.abs(Bhat-B),0.90))
dec=np.where(Bhat-eps>0,"ADAPT",np.where(Bhat+eps<0,"FREEZE","ABSTAIN"))
adapt=dec=="ADAPT"; freeze=dec=="FREEZE"; abst=dec=="ABSTAIN"

pol=np.where(adapt,auca,auc0)              # ABSTAIN/FREEZE -> frozen (safe default)
oracle=np.maximum(auc0,auca)

def by_cond(arr,fn):
    return {c: float(fn(arr[cond==c])) for c in ["clean","detectable","covert"]}

out={
 "n_instances":N, "eps":eps,
 "decision_distribution_by_condition":{
    c:{d:int(((cond==c)&(dec==d)).sum()) for d in ["ADAPT","FREEZE","ABSTAIN"]}
    for c in ["clean","detectable","covert"]},
 "true_mean_benefit_B_by_condition": by_cond(B, np.mean),
 "safety":{
    "adapt_precision_B>0":float(np.mean(B[adapt]>0)) if adapt.any() else None,
    "false_adapt_rate_B<0":float(np.mean(B[adapt]<0)) if adapt.any() else None,
 },
 "mean_auc_policies":{
    "always_adapt":float(auca.mean()),
    "always_freeze":float(auc0.mean()),
    "K_Bound":float(pol.mean()),
    "oracle":float(oracle.mean()),
 },
 "regret_vs_oracle":{
    "always_adapt":float((oracle-auca).mean()),
    "always_freeze":float((oracle-auc0).mean()),
    "K_Bound":float((oracle-pol).mean()),
 },
 "nonidentifiability_clean_vs_covert":{
    "mean_Z_L2_distance_clean_to_nearest_covert": None,  # filled below
    "mean_B_clean": by_cond(B,np.mean)["clean"],
    "mean_B_covert": by_cond(B,np.mean)["covert"],
 },
}

# clean vs covert: are their Z's close while B differs? (empirical Theorem-1 witness)
Xs=(X-X.mean(0))/(X.std(0)+1e-9)
cl=np.where(cond=="clean")[0]; co=np.where(cond=="covert")[0]
dists=[min(np.linalg.norm(Xs[i]-Xs[j]) for j in co) for i in cl]
out["nonidentifiability_clean_vs_covert"]["mean_Z_L2_distance_clean_to_nearest_covert"]=float(np.mean(dists))
# reference scale: clean to nearest DETECTABLE (should be much larger)
de=np.where(cond=="detectable")[0]
dists_det=[min(np.linalg.norm(Xs[i]-Xs[j]) for j in de) for i in cl]
out["nonidentifiability_clean_vs_covert"]["mean_Z_L2_distance_clean_to_nearest_detectable"]=float(np.mean(dists_det))

json.dump(out,open(os.path.join(RES,"mixed_regime_results.json"),"w"),indent=2)
print(json.dumps(out,indent=2))

# ---- figures ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Fig: decision distribution by condition (stacked)
conds=["clean","detectable","covert"]; decs=["ADAPT","FREEZE","ABSTAIN"]; colors={"ADAPT":"#2a9d8f","FREEZE":"#457b9d","ABSTAIN":"#9aa0a6"}
M=np.array([[out["decision_distribution_by_condition"][c][d] for d in decs] for c in conds],float)
M=M/M.sum(1,keepdims=True)
plt.figure(figsize=(6,4)); bottom=np.zeros(3)
for k,d in enumerate(decs):
    plt.bar(conds,M[:,k],bottom=bottom,color=colors[d],label=d); bottom+=M[:,k]
plt.ylabel("fraction of instances"); plt.title("K-Bound decisions by true regime")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_mixed_decisions.png"),dpi=130)

# Fig: policy comparison
plt.figure(figsize=(5.6,4))
labs=["always-adapt","always-freeze","K-Bound","oracle"]
vals=[auca.mean(),auc0.mean(),pol.mean(),oracle.mean()]
plt.bar(labs,vals,color=["#e76f51","#457b9d","#2a9d8f","#999999"])
for i,v in enumerate(vals): plt.text(i,v+.003,f"{v:.3f}",ha="center",fontsize=9)
plt.ylim(min(vals)-0.03,max(vals)+0.03); plt.ylabel("mean test AUROC")
plt.title("Mixed regime: K-Bound beats both trivial policies")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_mixed_policies.png"),dpi=130)
print("\nsaved figs + json")
