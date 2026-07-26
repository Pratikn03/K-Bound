import sys, numpy as np
sys.path.insert(0,"docs/research/kbound/scripts")
import analyze_F as A
# fix-queue item 15 / defect D10: the certificate radius is the shipped one.
import kbound_decide as _kb  # noqa: E402
ALPHA=0.10; ADAPTER="sar_online_aggressive"
calf="experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json"
testf="experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json"
cr,_=A.load_records(calf); tr,_=A.load_records(testf)
def pick(recs): 
    return [r for r in recs if r.get("candidate")==ADAPTER and r.get("seed") in (0,1)]
cr=pick(cr); tr=pick(tr)
print("cal recs", len(cr), "test recs", len(tr))
Zc=np.array([r["Z"] for r in cr],float); Bc=np.array([r["B"] for r in cr],float)
Zt=np.array([r["Z"] for r in tr],float); Bt=np.array([r["B"] for r in tr],float)
a0t=np.array([r["a0"] for r in tr],float); aat=np.array([r["aa"] for r in tr],float)
m=A.fit_point(Zc,Bc); Bhat_c=m.predict(Zc); Bhat_t=m.predict(Zt)
# D10: exact split-conformal rank radius via the shipped library, not np.quantile.
eps=float(_kb.conformal_radius(np.abs(Bhat_c-Bc),ALPHA))
dec=A.decide_global(Bhat_t,eps)
adapt=dec=="ADAPT"; kga=np.where(adapt,aat,a0t); oracle=np.maximum(a0t,aat)
rk=oracle-kga; ra=oracle-aat; rf=oracle-a0t
print("reproduced: kga %.5f adapt %.5f freeze %.5f (locked 0.0022/0.0468/0.0158) | FA %.3f adapt_rate %.3f n %d"%(
    rk.mean(),ra.mean(),rf.mean(), float((Bt[adapt]<0).mean()) if adapt.any() else 0.0, adapt.mean(), len(rk)))
n=len(rk); rng=np.random.default_rng(20260619); B=10000; dka=[];dkf=[]
for _ in range(B):
    ii=rng.integers(0,n,n); dka.append(ra[ii].mean()-rk[ii].mean()); dkf.append(rf[ii].mean()-rk[ii].mean())
dka=np.array(dka); dkf=np.array(dkf)
def ci(a): return [round(float(np.quantile(a,.025)),5),round(float(np.quantile(a,.975)),5)]
print("KGA<adapt  +%.5f CI %s P=%.3f"%(ra.mean()-rk.mean(),ci(dka),(dka>0).mean()))
print("KGA<freeze +%.5f CI %s P=%.3f"%(rf.mean()-rk.mean(),ci(dkf),(dkf>0).mean()))
print("BEATS_BOTH_SIGNIFICANT:", bool((dka>0).mean()>=0.95 and (dkf>0).mean()>=0.95))
