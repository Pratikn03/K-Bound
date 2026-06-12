"""Build a D23-format cache from the REAL Real-IAD-D3 natural-degradation per-modality
scores (D29). No GPU/reprocessing: the deep PatchCore per-modality scores already exist
as CSVs in experiments/fusion/realiad_d3_headroom_parts/ (derived from the 259 GB raw
Real-IAD-D3). We pivot each category to (sample x modality) score matrices with the
native validation/test split and labels, so the ELARA-U reliability router can be run
on REAL natural degradation (not injected failure).

Honest scope: the source CSVs are OPENED_DEVELOPMENT data (not the sealed D18 heldout),
so any result is development evidence on real natural degradation.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PARTS = ROOT / "experiments/fusion/realiad_d3_headroom_parts"
OUT = ROOT / "experiments/fusion/realiad_natdeg_score_cache"
MODS = ["rgb", "ps", "xyz"]


def build_category(csv: Path):
    df = pd.read_csv(csv)
    df = df[df["domain"].isin(MODS)]
    # one score per (sample, modality); pivot to columns rgb/ps/xyz
    piv = df.pivot_table(index="sample_id", columns="domain", values="score", aggfunc="mean")
    meta = df.groupby("sample_id").agg(split=("split", "first"), label=("label", "first"))
    piv = piv.join(meta).dropna(subset=MODS)
    if piv.empty:
        return None
    val = piv[piv["split"] == "validation"]
    test = piv[piv["split"] == "test"]
    if len(val) < 8 or len(test) < 8:
        return None
    Sval = val[MODS].to_numpy(float).copy(); yval = val["label"].to_numpy(int)
    Stest = test[MODS].to_numpy(float).copy(); ytest = test["label"].to_numpy(int)
    if len(np.unique(yval)) < 2 or len(np.unique(ytest)) < 2:
        return None
    # orient + normalize PER MODALITY on VALIDATION ONLY (no test labels):
    #   1) flip sign if val-AUROC < 0.5 (orientation chosen on validation)
    #   2) z-sigmoid using validation mean/std -> [0,1], matching the D23 cache format
    for j in range(3):
        if roc_auc_score(yval, Sval[:, j]) < 0.5:
            Sval[:, j] = -Sval[:, j]; Stest[:, j] = -Stest[:, j]
        mu, sd = float(Sval[:, j].mean()), float(Sval[:, j].std() + 1e-6)
        Sval[:, j] = 1.0 / (1.0 + np.exp(-(Sval[:, j] - mu) / sd))
        Stest[:, j] = 1.0 / (1.0 + np.exp(-(Stest[:, j] - mu) / sd))
    vauc = np.array([roc_auc_score(yval, Sval[:, j]) for j in range(3)], float)
    return Sval, yval, Stest, ytest, vauc


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    csvs = [f for f in sorted(glob.glob(str(PARTS / "*.csv"))) if not os.path.basename(f).startswith("._")]
    n = 0
    for csv in csvs:
        cat = os.path.basename(csv)[:-4]
        try:
            res = build_category(Path(csv))
        except Exception as e:
            print(f"[{cat}] FAILED: {type(e).__name__}: {e}"); continue
        if res is None:
            print(f"[{cat}] skipped (insufficient labelled split)"); continue
        Sval, yval, Stest, ytest, vauc = res
        np.savez(OUT / f"{cat}.npz", Sval=Sval, yval=yval, Stest=Stest, ytest=ytest, valauc=vauc)
        n += 1
        print(f"[{cat}] val={len(yval)} test={len(ytest)} valauc(rgb,ps,xyz)={np.round(vauc,3)}", flush=True)
    print(f"\nwrote {n} Real-IAD-D3 natural-degradation caches to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
