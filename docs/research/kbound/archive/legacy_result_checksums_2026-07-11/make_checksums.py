#!/usr/bin/env python3
"""Write SHA-256 content checksums for the headline result artifacts.

Run from the monorepo root:  python3 docs/research/kbound/scripts/make_checksums.py
Verify:                      sha256sum -c docs/research/kbound/results/CHECKSUMS.sha256
Rationale: run manifests carry git SHAs but no content hashes; this file pins the
exact bytes behind every number promoted to the paper (kbound_short.tex).
"""
import hashlib, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ARTIFACTS = [
    "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json",
    "experiments/kbound/results/decisive_tta_results.json",
    "experiments/kbound/results/decisive_tta_cis.json",
    "experiments/kbound/results/win_hunt_v5/imagenetc_aggr/decisive_tta_results.json",
    "experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json",
    "docs/research/kbound/experiments/kbound/results/decisive_tta_results.json",
    "docs/research/kbound/results_source.json",
    "docs/research/kbound/paper/generated/kbound_numbers.tex",
    "docs/research/kbound/percondition_bootstrap.json",
    "research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json",
    "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml",
    "research_lock/WIN_HUNT_v2_PROTOCOL.yaml",
    "research_lock/WIN_HUNT_v3_PROTOCOL.yaml",
    "audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json",
]
out = []
missing = []
for rel in ARTIFACTS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        missing.append(rel)
        continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    out.append("%s  %s" % (h, rel))
dst = os.path.join(ROOT, "docs/research/kbound/results/CHECKSUMS.sha256")
open(dst, "w").write("\n".join(out) + "\n")
print("wrote %d checksums -> %s" % (len(out), dst))
if missing:
    print("MISSING (not hashed):")
    for m in missing:
        print("  ", m)
    sys.exit(1)
