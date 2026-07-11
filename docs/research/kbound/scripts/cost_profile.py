#!/usr/bin/env python3
"""
KGA controller overhead: the *added* cost of the decision layer on top of an
adaptation step (which must run regardless, since f_a is evaluated before commit).
Measures evidence assembly + benefit-model evaluation latency, and the memory of the
rollback model copy. No target labels, no T9; uses the in-repo logged evidence and the
resnet18_cifar.pt checkpoint size for the shadow-copy figure.
"""
import json, os, time, pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "..", "experiments", "kbound", "results")
CKPT = os.path.join(HERE, "..", "experiments", "kbound", "cifar", "resnet18_cifar.pt")

def evidence_11(p0, pa):
    """The 11-dim base panel from frozen/adapted softmax batches (representative timing)."""
    e0 = -(p0*np.log(p0+1e-9)).sum(1); ea = -(pa*np.log(pa+1e-9)).sum(1)
    c0 = p0.max(1); ca = pa.max(1); C = p0.shape[1]
    mb0 = p0.mean(0); mba = pa.mean(0)
    pb0 = -(mb0*np.log(mb0+1e-9)).sum()/np.log(C); pba = -(mba*np.log(mba+1e-9)).sum()/np.log(C)
    return np.array([e0.mean(), c0.mean(), pb0, ea.mean(), ca.mean(), pba,
                     pb0-pba, e0.mean()-ea.mean(), float((ca>0.9).mean()),
                     float((mba*np.log((mba+1e-9)/(mb0+1e-9))).sum()), 0.037])

# fit the benefit model on logged tent evidence
recs = json.load(open(os.path.join(R,"per_condition_cifar10c_tent_seed0.json")))["records"]
Z = np.array([r["Z"] for r in recs], float); B = np.array([r["B"] for r in recs], float)
gbr = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                subsample=0.8, random_state=0).fit(Z, B)

# 1) evidence assembly latency (batch m=200, C=10 — representative CIFAR adaptation batch)
rng = np.random.default_rng(0)
p0 = rng.dirichlet(np.ones(10), 200); pa = rng.dirichlet(np.ones(10), 200)
N=2000; t=time.perf_counter()
for _ in range(N): evidence_11(p0, pa)
ev_ms = (time.perf_counter()-t)/N*1e3

# 2) benefit-model eval latency (per decision)
z1 = Z[:1]; t=time.perf_counter()
for _ in range(N): gbr.predict(z1)
gbr_ms = (time.perf_counter()-t)/N*1e3

# 3) memory: rollback model copy + benefit-model size
ckpt_mb = os.path.getsize(CKPT)/1e6 if os.path.exists(CKPT) else None
gbr_kb = len(pickle.dumps(gbr))/1e3

out = {
  "evidence_assembly_ms_per_batch": round(ev_ms,4),
  "benefit_model_ms_per_decision": round(gbr_ms,4),
  "controller_added_ms_per_decision": round(ev_ms+gbr_ms,4),
  "rollback_copy_MB (resnet18_cifar.pt)": round(ckpt_mb,2) if ckpt_mb else "n/a",
  "benefit_model_size_KB": round(gbr_kb,1),
  "note": "Model forward + adaptation step is unchanged by KGA (f_a computed regardless); "
          "the numbers above are the controller's ADDED cost. Latency on sandbox CPU."
}
json.dump(out, open(os.path.join(R,"cost_profile.json"),"w"), indent=2)
print(json.dumps(out, indent=2))
