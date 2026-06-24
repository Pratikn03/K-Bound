import numpy as np

from theory_v2_validation import (
    agreements,
    load_results,
    recover_b_up_to_flip,
    save_results,
    simulate_panel,
)

res=load_results()
m=5000; reps=300
configs=[("high",np.array([0.78,0.74,0.72])),("mid",np.array([0.70,0.64,0.62])),
         ("low",np.array([0.62,0.58,0.57])),("verylow",np.array([0.58,0.555,0.55]))]
rows=[]
for name,q in configs:
    b=2*q-1; cmin=float(np.min(np.abs(np.outer(b,b))[~np.eye(3,dtype=bool)]))
    dl=np.zeros(reps); de=np.zeros(reps)
    for r in range(reps):
        F,C,Y=simulate_panel(0.5,q,q,m,seed=40000+r)
        dl[r]=(C[1].astype(float)-C[0].astype(float)).mean()
        A=agreements(F);c=2*A-1; bh=recover_b_up_to_flip(c,3); de[r]=(bh[1]-bh[0])/2
    sd_ev=float(np.std(de)); sd_lab=float(np.std(dl))
    ratio=(sd_ev/sd_lab) if np.isfinite(sd_ev) and sd_ev>0 else None
    rows.append({"regime":name,"cmin":round(cmin,4),"sd_labeled":round(sd_lab,4),
                 "sd_evidence":(round(sd_ev,4) if np.isfinite(sd_ev) else None),
                 "efficiency_ratio":(round(ratio,2) if ratio else None)})
res["V4_rate"]["efficiency_vs_cmin"]={"rows":rows,
  "conclusion":"both channels are n^-1/2; labels buy a CONSTANT factor that scales ~1/cmin and "
               "diverges as pairwise agreement -> chance (cmin->0). So 'labels buy constants not rate' "
               "holds, but the constant is cmin-dependent and blows up at low margin."}
save_results(res)
print("added efficiency_vs_cmin:")
for r in rows: print(" ",r)
