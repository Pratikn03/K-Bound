#!/usr/bin/env python3
"""Fix-queue item 19 (F4-6, F4-14): quantify the seed-0 environment heterogeneity.

Reads every `result_manifest.json` under experiments/kbound/results/ and tabulates
git hash, interpreter, torch, numpy, sklearn (if recorded), finish time, quick flag
and argv, so the panel footnote can name exactly what differs.

Also checks: does ANY manifest pin scikit-learn?  (b_hat comes from
GradientBoostingRegressor(subsample=0.8); without a pinned sklearn every eps and
every decision is version-dependent.)

Run: python3 07_env_heterogeneity.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import REPO

FIELDS = ["git_hash", "python", "torch", "numpy", "sklearn", "scikit_learn",
          "quick", "finished", "wall_time_sec"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, "experiments/kbound/**/result_manifest*.json"),
                              recursive=True)):
        rel = os.path.relpath(p, REPO)
        raw = open(p, "rb").read()
        if len(raw) == 0 or b"\x00" in raw:
            rows.append({"path": rel, "status": "PLACEHOLDER"})
            continue
        try:
            d = json.loads(raw)
        except Exception as e:
            rows.append({"path": rel, "status": f"UNPARSEABLE {e}"})
            continue
        row = {"path": rel, "status": "OK", "seed": d.get("seed")}
        for f in FIELDS:
            if f in d:
                row[f] = d[f]
        row["argv"] = d.get("argv")
        rows.append(row)

    groups = {}
    for r in rows:
        if r["status"] != "OK":
            continue
        top = r["path"].split("/")[3] if len(r["path"].split("/")) > 3 else r["path"]
        groups.setdefault(top, []).append(r)

    any_sklearn = [r["path"] for r in rows
                   if r.get("sklearn") or r.get("scikit_learn")]

    out = {"manifests": rows,
           "manifests_pinning_sklearn": any_sklearn,
           "n_manifests": len(rows),
           "n_placeholder": sum(1 for r in rows if r["status"] != "OK")}
    json.dump(out, open(os.path.join(here, "out_env.json"), "w"), indent=1)

    for top in sorted(groups):
        rs = groups[top]
        stacks = {(r.get("git_hash"), r.get("python"), r.get("torch"), r.get("numpy"))
                  for r in rs}
        if len(rs) < 2:
            continue
        flag = "  <== HETEROGENEOUS" if len(stacks) > 1 else ""
        print(f"\n{top}   ({len(rs)} manifests, {len(stacks)} distinct stacks){flag}")
        for r in sorted(rs, key=lambda x: str(x.get("seed"))):
            print(f"   seed={str(r.get('seed')):>5s}  git={str(r.get('git_hash'))[:12]}"
                  f"  py={r.get('python')}  torch={r.get('torch')}  numpy={r.get('numpy')}"
                  f"  quick={r.get('quick')}  finished={r.get('finished')}")
    print(f"\nmanifests pinning scikit-learn: {len(any_sklearn)} of {len(rows)}")
    print("(b_hat comes from GradientBoostingRegressor(subsample=0.8); an unpinned "
          "sklearn makes every eps and every decision version-dependent.)")

    # argv diffs inside a group
    print("\n--- argv differences within a seed group ---")
    for top in sorted(groups):
        rs = [r for r in groups[top] if r.get("argv")]
        if len(rs) < 2:
            continue
        def strip_seed(a):
            out, skip = [], False
            for i, x in enumerate(a):
                if skip:
                    skip = False; continue
                if x in ("--seed", "--out-results"):
                    skip = True; continue
                out.append(x)
            return tuple(out)
        sig = {strip_seed(r["argv"]) for r in rs}
        if len(sig) > 1:
            print(f"  {top}: {len(sig)} distinct argv signatures")
            for r in sorted(rs, key=lambda x: str(x.get("seed"))):
                print(f"     seed={r.get('seed')}  {' '.join(map(str, r['argv']))}")


if __name__ == "__main__":
    main()
