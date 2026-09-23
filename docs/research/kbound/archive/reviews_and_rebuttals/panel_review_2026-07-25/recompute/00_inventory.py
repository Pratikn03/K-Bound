#!/usr/bin/env python3
"""Inventory every JSON artifact in the K-Bound tree that carries per-cell
(B, b_hat, eps_conformal, kga_decision) records -- i.e. everything from which a
conformal radius / decision can be recomputed.

Also flags NUL-filled iCloud placeholders and unreadable files.

Usage:  python3 00_inventory.py            (writes inventory.json next to this file)
"""
import json, os, sys

REPO = "/home/claude/kb"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory.json")

placeholders = []
unreadable = []
scored = []

for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
    for fn in files:
        if not fn.endswith(".json"):
            continue
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, REPO)
        try:
            raw = open(p, "rb").read()
        except OSError as e:
            unreadable.append({"path": rel, "error": str(e)})
            continue
        if len(raw) == 0 or b"\x00" in raw:
            placeholders.append({"path": rel, "bytes": len(raw), "has_nul": b"\x00" in raw})
            continue
        if raw.strip() == b"":
            placeholders.append({"path": rel, "bytes": len(raw), "has_nul": False,
                                 "note": "whitespace-only"})
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        recs = None
        if isinstance(d, dict) and isinstance(d.get("records"), list):
            recs = d["records"]
        elif isinstance(d, list) and d and isinstance(d[0], dict):
            recs = d
        if not recs or not isinstance(recs[0], dict):
            continue
        k = set(recs[0].keys())
        if "B" in k and ("b_hat" in k or "bhat" in k):
            scored.append({
                "path": rel, "n": len(recs),
                "has_eps": "eps_conformal" in k,
                "has_dec": "kga_decision" in k,
                "has_a0": "a0" in k,
                "has_a_adapted": "a_adapted" in k,
                "keys": sorted(k),
            })

scored.sort(key=lambda r: r["path"])
res = {"n_scored_files": len(scored), "n_placeholder_json": len(placeholders),
       "n_unreadable_json": len(unreadable),
       "scored": scored, "placeholders": placeholders, "unreadable": unreadable}
json.dump(res, open(OUT, "w"), indent=1)
print("scored per-cell files:", len(scored))
print("NUL/empty placeholder JSONs:", len(placeholders))
print("unreadable JSONs:", len(unreadable))
for s in scored:
    print(f"  n={s['n']:5d}  eps={int(s['has_eps'])} dec={int(s['has_dec'])} a0={int(s['has_a0'])}  {s['path']}")
