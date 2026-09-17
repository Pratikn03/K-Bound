"""Rebuild results/result_manifest.json by scanning the organized results/ groups."""
import json, os, glob
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # docs/research/kbound
idx = {"groups": {}}
for grp in ["main", "ablations", "tta", "regression", "witness", "multimodal"]:
    idx["groups"][grp] = sorted(os.path.basename(p) for p in glob.glob(f"{HERE}/results/{grp}/*.json"))
json.dump(idx, open(f"{HERE}/results/result_manifest.json", "w"), indent=2)
print("wrote results/result_manifest.json:", {k: len(v) for k, v in idx["groups"].items()})
