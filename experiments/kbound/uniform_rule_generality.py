# (c) Generality probe: does ONE adapter + ONE uniform method win across all real shifts,
# or does each dataset need its own hand-picked adapter? Same gbr+global-conformal, alpha=0.10,
# same decision rule everywhere. Re-scores existing records; no GPU. Honest, report as-is.
import sys, numpy as np
sys.path.insert(0,"docs/research/kbound/scripts")
import analyze_F as A
ALPHA=0.10
DS = {
 "Camelyon17": dict(cal="experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json",
                    test="experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json",
                    cs=[0,1], ts=[2,3,4]),
 "Office-Home": dict(cal="experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json",
                     test="experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json",
                     cs=[0,1], ts=[0,1]),
 "iWildCam":  dict(cal="experiments/kbound/results/iwildcam_full_test/result_e40faf29.json",
                   test="experiments/kbound/results/iwildcam_full_test/result_e40faf29.json",
                   cs=[0], ts=[1]),
}
def evalk(cal,test,cs,ts,cand):
    cr,_=A.load_records(cal); tr,_=A.load_records(test)
    cr=[r for r in cr if r.get("candidate")==cand and r.get("seed") in cs]
    tr=[r for r in tr if r.get("candidate")==cand and r.get("seed") in ts]
    if len(cr)<3 or len(tr)<3: return None
    Zc=np.array([r["Z"] for r in cr],float); Bc=np.array([r["B"] for r in cr],float)
    Zt=np.array([r["Z"] for r in tr],float); Bt=np.array([r["B"] for r in tr],float)
    a0=np.array([r["a0"] for r in tr],float); aa=np.array([r["aa"] for r in tr],float)
    m=A.fit_point(Zc,Bc); eps=float(np.quantile(np.abs(m.predict(Zc)-Bc),1-ALPHA))
    dec=A.decide_global(m.predict(Zt),eps); adapt=dec=="ADAPT"
    kga=np.where(adapt,aa,a0); orc=np.maximum(a0,aa)
    rk=(orc-kga).mean(); ra=(orc-aa).mean(); rf=(orc-a0).mean()
    fa=float((Bt[adapt]<0).mean()) if adapt.any() else 0.0
    return dict(rk=rk,ra=ra,rf=rf,fa=fa,adapt=float(adapt.mean()),n=len(tr),
                beats=bool(rk<ra-1e-12 and rk<rf-1e-12 and fa<=ALPHA))
# candidate universe = union across datasets
allc=set()
for cfg in DS.values():
    cr,_=A.load_records(cfg["cal"]); allc|={r.get("candidate") for r in cr if r.get("candidate")}
allc=sorted(c for c in allc if c)
print("candidates:",len(allc))
res={}
for cand in allc:
    row={}
    for name,cfg in DS.items():
        try: row[name]=evalk(cfg["cal"],cfg["test"],cfg["cs"],cfg["ts"],cand)
        except Exception as e: row[name]=None
    res[cand]=row
# print: for each candidate, beats-both count across datasets
print("\n=== adapter x dataset : beats-both (FA<=a) ? [regret_kga] ===")
hdr="%-26s %-12s %-12s %-12s  %s"%("adapter","Camelyon17","Office-Home","iWildCam","#win")
print(hdr); print("-"*len(hdr))
for cand in allc:
    cells=[]; nwin=0
    for name in DS:
        r=res[cand][name]
        if r is None: cells.append("  --   "); continue
        tag="WIN" if r["beats"] else ("tie/– " )
        if r["beats"]: nwin+=1
        cells.append("%s %.4f"%("Y" if r["beats"] else "n", r["rk"]))
    print("%-26s %-12s %-12s %-12s  %d/3"%(cand,cells[0],cells[1],cells[2],nwin))
# universal adapter?
print("\n=== generality verdict ===")
best=max(allc,key=lambda c:sum(1 for n in DS if res[c][n] and res[c][n]["beats"]))
nb=sum(1 for n in DS if res[best][n] and res[best][n]["beats"])
print("best single uniform adapter: %s beats-both on %d/3 datasets"%(best,nb))
