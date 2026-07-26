#!/usr/bin/env python3
"""Full-tree scan for iCloud placeholders (fix-queue item 9's release guard).

The naive "is it whitespace-only?" test does NOT catch these; the reliable test is
a NUL byte anywhere in the file (or zero length) for a file whose extension says
it should be text.

Run: python3 13_placeholder_scan.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import REPO

TEXT_EXT = {".json", ".md", ".py", ".tex", ".csv", ".yaml", ".yml", ".txt", ".sh",
            ".cfg", ".toml", ".lean", ".bib", ".cff", ".ipynb"}

placeholders, unreadable, whitespace_only, total = [], [], [], 0
for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
    for fn in files:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in TEXT_EXT:
            continue
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, REPO)
        total += 1
        try:
            raw = open(p, "rb").read()
        except OSError as e:
            unreadable.append({"path": rel, "error": str(e)})
            continue
        if len(raw) == 0 or b"\x00" in raw:
            placeholders.append({"path": rel, "bytes": len(raw),
                                 "has_nul": b"\x00" in raw})
        elif raw.strip() == b"":
            whitespace_only.append({"path": rel, "bytes": len(raw)})

by_ext = {}
for r in placeholders:
    by_ext[os.path.splitext(r["path"])[1]] = by_ext.get(
        os.path.splitext(r["path"])[1], 0) + 1

here = os.path.dirname(os.path.abspath(__file__))
json.dump({"n_text_files_scanned": total,
           "n_placeholders": len(placeholders),
           "n_whitespace_only": len(whitespace_only),
           "n_unreadable": len(unreadable),
           "by_extension": by_ext,
           "placeholders": placeholders,
           "whitespace_only": whitespace_only,
           "unreadable": unreadable},
          open(os.path.join(here, "out_placeholders.json"), "w"), indent=1)

print(f"text files scanned          : {total}")
print(f"NUL-filled / zero-byte      : {len(placeholders)}")
print(f"whitespace-only (naive test): {len(whitespace_only)}")
print(f"unreadable (OSError)        : {len(unreadable)}")
print("by extension:", by_ext)
print("\nfirst 40 placeholders:")
for r in placeholders[:40]:
    print(f"   {r['bytes']:7d} B  nul={r['has_nul']}  {r['path']}")
