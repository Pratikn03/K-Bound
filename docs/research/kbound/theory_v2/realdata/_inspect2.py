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
def show(path, tag):
    print(f"\n##### {tag}: {path}")
    try:
        j=json.load(open(path))
    except Exception as e:
        print("  ERR", e); return
    bm=j.get("benchmarks",{})
    for bench,bd in bm.items():
        for meth,md in bd.get("methods",{}).items():
            met=md["metrics"]
            ma=met["mean_acc"]; rv=met["regret_vs_oracle"]; wc=met["worst_case_acc"]
            # check if any value is a list (per-condition array)
            listy={k:(len(v) if isinstance(v,list) else "scalar") for k,v in ma.items()}
            print(f"  {bench}/{meth}: n={met['n']} mean_acc(types={listy}) regret={ {k:round(v,5) if isinstance(v,(int,float)) else type(v).__name__ for k,v in rv.items()} }")
            # look for any per-condition raw array key anywhere in md
            for kk,vv in md.items():
                if isinstance(vv,list) and len(vv)>50 and all(isinstance(x,(int,float)) for x in vv[:5]):
                    print(f"    RAW-ARRAY found in md[{kk}] len={len(vv)}")
            for kk,vv in met.items():
                if isinstance(vv,list) and len(vv)>50 and all(isinstance(x,(int,float)) for x in vv[:5]):
                    print(f"    RAW-ARRAY found in metrics[{kk}] len={len(vv)}")

show(KB_REPO_ROOT + "/experiments/kbound/results/decisive_tta_results.json","MAIN cifar10c")
show(KB_REPO_ROOT + "/experiments/kbound/results/imagenetc_1pct/decisive_tta_results.json","imagenetc_1pct")
show(KB_REPO_ROOT + "/experiments/kbound/results/imagenetc_noise/decisive_tta_results.json","imagenetc_noise")
show(KB_REPO_ROOT + "/experiments/kbound/results/cifar101/decisive_tta_results.json","cifar101")
