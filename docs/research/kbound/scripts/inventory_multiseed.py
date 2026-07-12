#!/usr/bin/env python3
"""Exhaustive multi-seed inventory. Walks the ENTIRE results tree, opens every
per_condition_*_seed*.json, and reports per (dataset, candidate): which seeds exist, whether the
deployed KGA decision (a_kbound/kga_decision) and regret ingredients (a0/a_adapted/a_oracle/B) are
stored -> i.e. whether multi-seed no-harm is computable from disk. Emits exact run commands.
Pure stdlib (no numpy). Usage: inventory_multiseed.py <repo_root>
"""
import json, os, re, sys
from collections import defaultdict
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RES = os.path.join(ROOT, "experiments/kbound/results")
KNOWN = ["cifar10c","cifar101","imagenetc","imagenet-c","imagenet-r","imagenetr",
         "camelyon17","iwildcam","officehome","office_home","rxrx1","pacs"]
inv = defaultdict(lambda: {"seeds": set(), "dirs": set(), "kga": False, "ingr": False, "files": 0})
nfiles = nok = 0
for dp, dns, fns in os.walk(RES):
    for fn in fns:
        if not (fn.startswith("per_condition_") and fn.endswith(".json") and "seed" in fn):
            continue
        nfiles += 1
        base = fn[len("per_condition_"):-5]
        m = re.search(r"_seed(\d+)$", base)
        if not m:
            continue
        seed = int(m.group(1)); body = base[:m.start()]
        ds = next((k for k in KNOWN if body.startswith(k + "_")), body.split("_")[0])
        cand = body[len(ds) + 1:] if body.startswith(ds + "_") else body
        p = os.path.join(dp, fn)
        try:
            d = json.load(open(p)); nok += 1
        except Exception:
            continue
        recs = d.get("records") if isinstance(d, dict) else None
        if not recs:
            continue
        r0 = recs[0]; e = inv[(ds, cand)]
        e["seeds"].add(seed); e["dirs"].add(os.path.relpath(dp, RES)); e["files"] += 1
        if ("a_kbound" in r0) or ("kga_decision" in r0): e["kga"] = True
        if all(k in r0 for k in ("a0", "a_adapted", "a_oracle", "B")): e["ingr"] = True

print(f"opened {nok}/{nfiles} per_condition seed files under {RES}\n")
byds = defaultdict(list)
for (ds, cand), e in inv.items(): byds[ds].append((cand, e))
print(f"{'dataset':13s} {'candidate':24s} {'seeds':20s} {'KGA':4s} {'ingr':5s} ready")
cmds = []
for ds in sorted(byds):
    for cand, e in sorted(byds[ds]):
        ready = len(e["seeds"]) >= 3 and e["kga"] and e["ingr"]
        print(f"{ds:13s} {cand:24s} {str(sorted(e['seeds'])):20s} {str(e['kga']):4s} {str(e['ingr']):5s} {'YES' if ready else '-'}")
        if ready:
            cmds.append((ds, cand, sorted(e["dirs"])[0]))
print(f"\n=== {len(cmds)} (dataset,candidate) multi-seed-ready (>=3 seeds + deployed decision) ===")
for ds, cand, d in cmds:
    print(f"python3 docs/research/kbound/scripts/multiseed_natural.py --dataset {ds} --candidate {cand} --dir {ROOT}/experiments/kbound/results/{d}")
open("/tmp/multiseed_cmds.txt", "w").write("\n".join(
    f"{ROOT}/experiments/kbound/results/{d}||{ds}||{cand}" for ds, cand, d in cmds) + "\n")
print("\nwrote /tmp/multiseed_cmds.txt")
