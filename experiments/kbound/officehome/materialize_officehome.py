"""
materialize_officehome.py - turn the HF `flwrlabs/office-home` parquet mirror into a
standard Office-Home ImageFolder tree:  <root>/<Domain>/<Class>/<idx>.jpg

Domains (4): Art, Clipart, Product, Real_World   (65 classes, ~15.5k images).
INTEGRITY: this only decodes the public dataset to disk; no labels are altered.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
from collections import Counter, defaultdict

DOMAIN_FOLDER = {
    "Art": "Art", "Clipart": "Clipart", "Product": "Product",
    "Real World": "Real_World", "Real_World": "Real_World", "RealWorld": "Real_World",
}


def norm_domain(d: str) -> str:
    return DOMAIN_FOLDER.get(d, d.replace(" ", "_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output ImageFolder root")
    ap.add_argument("--repo", default="flwrlabs/office-home")
    args = ap.parse_args()

    from datasets import load_dataset
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.repo} (cached to default HF cache)", flush=True)
    ds = load_dataset(args.repo, split="train")
    names = ds.features["label"].names
    print(f"[schema] {len(names)} classes; columns={ds.column_names}", flush=True)

    counts = Counter()
    dc = defaultdict(Counter)
    t0 = time.time()
    n = len(ds)
    for i in range(n):
        r = ds[i]
        dom = norm_domain(r["domain"])
        cls = names[r["label"]]
        d = out / dom / cls
        d.mkdir(parents=True, exist_ok=True)
        img = r["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(d / f"{i:06d}.jpg", quality=95)
        counts[dom] += 1
        dc[dom][cls] += 1
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{n}] {time.time()-t0:.0f}s", flush=True)

    print("\n[done] per-domain image counts:", flush=True)
    for dom in sorted(counts):
        cls_counts = dc[dom]
        print(f"  {dom:12s} n={counts[dom]:5d}  classes={len(cls_counts):2d} "
              f"min/cls={min(cls_counts.values())} max/cls={max(cls_counts.values())}", flush=True)
    print(f"[total] {sum(counts.values())} images, {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
