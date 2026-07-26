# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). KB_REPO_ROOT is discovered from this
# --- file's own location; override with $KBOUND_REPO_ROOT.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

import json, numpy as np
J=KB_REPO_ROOT + "/experiments/kbound/results/decisive_tta_results.json"
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
