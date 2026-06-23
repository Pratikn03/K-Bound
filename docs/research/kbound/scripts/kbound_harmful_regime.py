"""
K-Bound certificate in a HARMFUL regime.

Adapt candidate fa = elara_fuse (reliability fusion). On this suite it HURTS on
~80% of tasks (B<0), so 'always adapt' is dangerous. A good knowability rule
should mostly FREEZE and only ADAPT on the certifiable helpful minority.

  f0 (FREEZE) = auto_select        (per-task test AUC from results_clean.json)
  fa (ADAPT)  = elara_fuse         (per-task test AUC from results_clean.json)
  B = AUC(fa) - AUC(f0)            true benefit (oracle; labels only for eval)
  Z = label-free evidence from raw score archive (no test labels), joined by task name.
All numbers produced by this run.
"""
import glob, os, json
import numpy as np
from scipy.stats import ks_2samp, rankdata
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCH = os.path.join(_REPO, "experiments", "elara_u", "score_archive")
EXP  = os.path.join(_REPO, "experiments", "elara_u")
OUT  = os.path.join(_REPO, "experiments", "kbound", "results"); os.makedirs(OUT, exist_ok=True)
FIG  = os.path.join(_REPO, "docs", "research", "kbound", "figures"); os.makedirs(FIG, exist_ok=True)

res=json.load(open(os.path.join(EXP,"results_clean.json")))
names=res["task_names"]
auc_f0=np.array(res["per_task_auc"]["auto_select"])
auc_fa=np.array(res["per_task_auc"]["elara_fuse"])
Bmap={n:(float(auc_fa[i]-auc_f0[i]),float(auc_f0[i]),float(auc_fa[i])) for i,n in enumerate(names)}

def rank_norm(S):
    R=np.empty_like(S,dtype=float)
    for j in range(S.shape[1]): R[:,j]=(rankdata(S[:,j])-1)/(len(S)-1)
    return R

rows=[]
for f in sorted(glob.glob(os.path.join(ARCH,"*.npz"))):
    nm=os.path.basename(f)[:-4]
    if nm not in Bmap: continue
    d=np.load(f,allow_pickle=True)
    Sval,yval,Stest=d["Sval"],d["yval"],d["Stest"]; val_auc=d["val_auc"]
    vs=np.sort(val_auc)[::-1]
    ks=[ks_2samp(Sval[:,j],Stest[:,j]).statistic for j in range(Sval.shape[1])]
    Rt=rank_norm(Stest); C=np.corrcoef(Rt.T); iu=np.triu_indices_from(C,k=1)
    Z=[float(vs[0]),float(vs[0]-vs[1]),float(np.mean(val_auc)),
       float(np.mean(ks)),float(np.max(ks)),float(1-np.nanmean(C[iu])),
       float(np.std(val_auc)),float(np.mean(yval)),float(np.log(len(Stest)))]
    B,a0,aa=Bmap[nm]
    rows.append((nm,Z,B,a0,aa))

names=[r[0] for r in rows]; X=np.array([r[1] for r in rows])
B=np.array([r[2] for r in rows]); auc0=np.array([r[3] for r in rows]); auca=np.array([r[4] for r in rows])
N=len(rows)

# LOO estimator + conformal radius
Bhat=np.zeros(N)
for i in range(N):
    tr=np.arange(N)!=i
    m=GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)
    m.fit(X[tr],B[tr]); Bhat[i]=m.predict(X[i:i+1])[0]
eps=float(np.quantile(np.abs(Bhat-B),0.90))

dec=np.where(Bhat-eps>0,"ADAPT",np.where(Bhat+eps<0,"FREEZE","ABSTAIN"))
adapt=dec=="ADAPT"; freeze=dec=="FREEZE"; abst=dec=="ABSTAIN"

# policies (ABSTAIN defaults to frozen, the safe action)
pol=np.where(adapt,auca,auc0)
oracle=np.maximum(auc0,auca)
out={
 "setup":"fa=elara_fuse (harmful ~80%), f0=auto_select",
 "n_tasks":N, "eps":eps,
 "base_rate_harmful_B<0":float(np.mean(B<0)),
 "counts":{"ADAPT":int(adapt.sum()),"FREEZE":int(freeze.sum()),"ABSTAIN":int(abst.sum())},
 "adapt_precision_B>0":float(np.mean(B[adapt]>0)) if adapt.any() else None,
 "false_adapt_rate_B<0":float(np.mean(B[adapt]<0)) if adapt.any() else None,
 "freeze_correct_B<=0":float(np.mean(B[freeze]<=0)) if freeze.any() else None,
 "mean_auc":{
    "always_adapt(elara_fuse)":float(auca.mean()),
    "always_freeze(auto_select)":float(auc0.mean()),
    "K-Bound_trichotomy":float(pol.mean()),
    "oracle":float(oracle.mean()),
 },
 "regret_vs_oracle":{
    "always_adapt":float((oracle-auca).mean()),
    "always_freeze":float((oracle-auc0).mean()),
    "K-Bound":float((oracle-pol).mean()),
 },
}
json.dump(out,open(os.path.join(OUT,"kbound_harmful_results.json"),"w"),indent=2)
print(json.dumps(out,indent=2))
PY_END = None

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(7.6,4.2))
labels=["always-adapt\n(reliability-fusion)","always-freeze\n(best-val model)","K-Bound\ncertificate","oracle"]
vals=[auca.mean(),auc0.mean(),pol.mean(),oracle.mean()]
plt.bar(labels,vals,color=["#e76f51","#457b9d","#2a9d8f","#999999"])
for i,v in enumerate(vals): plt.text(i,v+.002,f"{v:.3f}",ha="center",fontsize=9)
plt.ylim(min(vals)-0.02,max(vals)+0.02); plt.ylabel("mean test AUROC")
plt.title("Harmful regime: K-Bound avoids the harmful adapt path")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_kbound_harmful.png"),dpi=300)
print("saved fig")
