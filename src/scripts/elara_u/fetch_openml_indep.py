"""Download + cache a fully-independent tabular suite from OpenML (D27 phase 1).

These datasets are verifiably ABSENT from the 135-task ELARA-U development archive
(checked by name against score_archive/). Minority class := anomaly. We cache raw
(X, y_binary) npz now; the frozen scoring (anomaly downsample + zoo + stacking) runs
separately and ONCE. Download success is independent of any result, so caching first
cleanly separates "what data exists" from "the measured effect".
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/openml_indep"

# (name, version) -- chosen by domain/name BEFORE scoring; disjoint from the archive.
SUITE = [
    ("phoneme", 1), ("credit-g", 1), ("eeg-eye-state", 1), ("nomao", 1),
    ("jm1", 1), ("kc1", 1), ("pc1", 1), ("churn", 1), ("Bioresponse", 1),
    ("mozilla4", 1), ("climate-model-simulation-crashes", 1),
    ("ozone-level-8hr", 1), ("wdbc", 1), ("qsar-biodeg", 1),
]


def _encode(df: pd.DataFrame) -> np.ndarray:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.factorize(df[c].astype(str))[0]
    return np.nan_to_num(df.to_numpy(dtype=float))


def fetch_one(name, ver, tries=5):
    for k in range(tries):
        try:
            d = fetch_openml(name=name, version=ver, as_frame=True, parser="auto")
            X = _encode(d.data.copy())
            yv = d.target.astype(str)
            minority = yv.value_counts().idxmin()
            y = (yv == minority).astype(int).to_numpy()
            return X, y, minority
        except Exception as e:
            if k == tries - 1:
                print(f"  ERR {name}: {type(e).__name__}: {str(e)[:90]}")
                return None
            time.sleep(3 * (k + 1))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = []
    for name, ver in SUITE:
        r = fetch_one(name, ver)
        if r is None:
            continue
        X, y, minority = r
        if X.shape[0] < 100 or int(y.sum()) < 15 or len(np.unique(y)) < 2:
            print(f"  skip {name}: too few rows/anomalies (n={X.shape[0]}, anom={int(y.sum())})")
            continue
        np.savez_compressed(OUT / f"{name}.npz", X=X, y=y)
        ok.append(name)
        print(f"OK  {name}: X={X.shape} anom_rate={y.mean():.4f} (minority='{minority}')", flush=True)
    print(f"\ncached {len(ok)}/{len(SUITE)} -> {OUT}")
    print("cached:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
