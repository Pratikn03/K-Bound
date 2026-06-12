import json, numpy as np
J="/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/decisive_tta_results.json"
j=json.load(open(J))
t=j["benchmarks"]["cifar10c"]["methods"]["tent"]
met=t["metrics"]
print("=== metrics scalar/struct summary ===")
for kk,v in met.items():
    if isinstance(v,list):
        print(f"{kk}: list len={len(v)} sample={v[:3]}")
    elif isinstance(v,dict):
        print(f"{kk}: dict keys={list(v.keys())}")
    else:
        print(f"{kk}: {v}")
print("\n=== pareto detail ===")
par=met.get("pareto")
print(type(par), (list(par.keys()) if isinstance(par,dict) else (len(par) if isinstance(par,list) else par)))
if isinstance(par,dict):
    for pk,pv in par.items():
        print(" ",pk, type(pv).__name__, (pv[:5] if isinstance(pv,list) else pv))
