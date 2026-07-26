#!/usr/bin/env python3
"""
export_pacs_hf.py -- pull PACS from HuggingFace (flwrlabs/pacs) and write it in the DomainBed
ImageFolder layout that pacs_vlcs_runner.py expects:  <out>/PACS/<domain>/<class>/<i>.jpg
Avoids the quota-blocked DomainBed Google Drive link.

Usage:
  pip install datasets
  python scripts/export_pacs_hf.py <repo root>/experiments/kbound/domainbed
Then:
  python scripts/pacs_vlcs_runner.py --dataset PACS --root <out> --device cpu --smoke
"""
import os, sys
from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
REPO = sys.argv[2] if len(sys.argv) > 2 else "flwrlabs/pacs"

# normalize domain strings to the runner's expected folder names
NORM = {"photo": "photo", "art_painting": "art_painting", "art painting": "art_painting",
        "art": "art_painting", "cartoon": "cartoon", "sketch": "sketch"}
def norm_dom(s):
    k = str(s).strip().lower().replace("-", " ")
    return NORM.get(k, k.replace(" ", "_"))

print(f"loading {REPO} ...", flush=True)
ds = load_dataset(REPO)
print("dataset:", ds, flush=True)

# concatenate all splits/configs into one iterable of examples
splits = list(ds.keys())
first = ds[splits[0]]
cols = first.column_names
print("columns:", cols, "| features:", first.features, flush=True)

img_col = next((c for c in cols if "image" in c.lower() or c.lower() == "img"), None)
dom_col = next((c for c in cols if "domain" in c.lower()), None)
lab_col = next((c for c in cols if c.lower() in ("label", "class", "category", "y", "labels")), None)
if img_col is None or lab_col is None:
    sys.exit(f"Could not auto-detect image/label columns from {cols}. "
             f"Paste this line to me and I'll adjust the exporter.")

def names(feat, col):
    f = feat[col]
    return getattr(f, "names", None)

n = 0
for sp in splits:
    d = ds[sp]
    labs = names(d.features, lab_col)
    doms = names(d.features, dom_col) if dom_col else None
    # if the split name encodes the domain (some PACS HF repos split by domain), use it
    sp_is_domain = norm_dom(sp) in NORM.values()
    for ex in d:
        domain = norm_dom(sp) if sp_is_domain else (
            norm_dom(doms[ex[dom_col]] if doms else ex[dom_col]) if dom_col else "unknown")
        cls = labs[ex[lab_col]] if labs else str(ex[lab_col])
        folder = os.path.join(OUT, "PACS", domain, str(cls))
        os.makedirs(folder, exist_ok=True)
        ex[img_col].convert("RGB").save(os.path.join(folder, f"{n}.jpg"))
        n += 1
        if n % 1000 == 0:
            print(f"  wrote {n} images ...", flush=True)

root = os.path.join(OUT, "PACS")
print(f"done: {n} images -> {root}")
print("domains:", sorted(os.listdir(root)))
