"""Build a fully-independent external suite (D27 phase 1) -- reliable, offline-first.

OpenML's API was unavailable; we instead use sources that are reliable AND verifiably
absent from the 135-task development archive:
  - sklearn `digits` (8x8, 10 classes)  -> 10 one-vs-rest tasks  (NOT optdigits/pendigits/mnist)
  - sklearn `wine` (3 classes)          -> 3 tasks
  - sklearn `wdbc` / breast_cancer      -> 1 task (malignant = anomaly)  (NOT breastw)
  - HAR smartphones (561 feats, 6 acts) -> 6 one-vs-rest tasks  (Kaggle uciml mirror)
Construction (frozen): for class c, anomaly := class c, normal := all other classes.
We cache raw (X, y_binary) per task; anomaly downsampling + scoring run once, later.
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/indep_external"
HAR = ROOT / "data/raw/har"


def _one_vs_rest(X, y, src, tag_fn):
    """Yield (name, X, y_binary) with each class in turn as the anomaly."""
    X = StandardScaler().fit_transform(np.nan_to_num(X.astype(float)))
    for c in sorted(set(y)):
        yb = (y == c).astype(int)
        if yb.sum() < 15 or (len(yb) - yb.sum()) < 30:
            continue
        yield f"{src}__{tag_fn(c)}", X, yb


def _load_har():
    fs = [f for f in glob.glob(str(HAR / "*.csv")) if not os.path.basename(f).startswith("._")]
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    y = df["Activity"].astype(str).to_numpy()
    X = df.drop(columns=[c for c in df.columns if c.lower() in ("activity", "subject")]).to_numpy(dtype=float)
    return X, y


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    sources = []
    dig = load_digits(); sources.append(("digits", dig.data, dig.target.astype(int), lambda c: f"d{c}"))
    win = load_wine(); sources.append(("wine", win.data, win.target.astype(int), lambda c: f"c{c}"))
    wd = load_breast_cancer()  # minority (malignant, target==0) = anomaly
    sources.append(("wdbc", wd.data, (wd.target == 0).astype(int), lambda c: "malignant" if c == 1 else "benign"))
    har = _load_har()
    if har is not None:
        sources.append(("har", har[0], har[1], lambda c: str(c).replace(" ", "")))

    for src, X, y, tagf in sources:
        for name, Xt, yb in _one_vs_rest(X, y, src, tagf):
            if src == "wdbc" and not name.endswith("malignant"):
                continue   # binary: keep only the malignant-as-anomaly task
            np.savez_compressed(OUT / f"{name}.npz", X=Xt, y=yb)
            n += 1
            print(f"OK  {name:22} X={Xt.shape} anom_rate={yb.mean():.4f}", flush=True)
    print(f"\ncached {n} independent tasks -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
