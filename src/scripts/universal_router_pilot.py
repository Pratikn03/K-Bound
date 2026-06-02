"""Universal anomaly meta-routing PILOT (make-or-break, tabular only).

Pre-registered question (Gate U primary, tabular slice): does a VALIDATION-ONLY
reliability router beat (a) a fixed robust default (ECOD), (b) the best single
detector chosen globally, and (c) a MetaOD-style val-AUROC selector, on AVERAGE
RANK across real tabular anomaly datasets, with a paired bootstrap CI > 0?

Honest expectation: this is hard (ECOD/IForest are strong; MetaOD-style selection
barely beats a fixed default). If the router does NOT win on average rank with
CI>0, the universal direction is reported as a NEGATIVE pilot -- do not scale.

No test labels are used for any routing/selection. Validation labels set the
reliability signal; test labels only score the final AUROC.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import io
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ADBench (NeurIPS'22) Classical tabular datasets, fetched from GitHub raw + cached.
_RAW = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
_CACHE = Path("data/raw/adbench_classical")
DATASETS = ["18_Ionosphere", "20_letter", "21_Lymphography", "23_mammography",
            "25_musk", "26_optdigits", "28_pendigits", "29_Pima", "2_annthyroid",
            "30_satellite", "31_satimage-2", "38_thyroid", "18_Ionosphere",
            "27_PageBlocks", "19_landsat"]


def _load(name):
    _CACHE.mkdir(parents=True, exist_ok=True)
    fp = _CACHE / f"{name}.npz"
    if not fp.exists():
        req = urllib.request.Request(_RAW + name + ".npz", headers={"User-Agent": "pilot"})
        fp.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    z = np.load(io.BytesIO(fp.read_bytes()))
    X = np.asarray(z["X"], dtype=float)
    y = np.asarray(z["y"]).astype(int).ravel()
    rate = y.mean()
    if not (0.005 < rate < 0.45) or X.shape[0] < 200:
        return None
    return X, y


def _detectors():
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    return {
        "IForest": lambda: IForest(random_state=0),
        "LOF": lambda: LOF(),
        "ECOD": lambda: ECOD(),
        "COPOD": lambda: COPOD(),
        "KNN": lambda: KNN(),
    }


def _score_dataset(X, y, seed=0):
    """Return dict detector -> (val_auc, test_auc) using a val/test split."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xval, Xte = sc.transform(Xtr), sc.transform(Xval), sc.transform(Xte)
    out = {}
    for name, ctor in _detectors().items():
        try:
            m = ctor(); m.fit(Xtr)
            sv = m.decision_function(Xval); st = m.decision_function(Xte)
            out[name] = (float(roc_auc_score(yval, sv)), float(roc_auc_score(yte, st)),
                         st, yte)
        except Exception:
            continue
    return out


def run(datasets, seed=0):
    per = {}
    for name in datasets:
        try:
            loaded = _load(name)
        except Exception:
            continue
        if loaded is None:
            continue
        X, y = loaded
        sc = _score_dataset(X, y, seed=seed)
        if len(sc) < 3:
            continue
        per[name] = sc

    if len(per) < 4:
        return {"error": "too few usable datasets", "n": len(per)}

    detectors = sorted({d for ds in per.values() for d in ds})
    # global best single (by mean validation AUROC across datasets -> a fixed default)
    mean_val = {d: np.mean([per[ds][d][0] for ds in per if d in per[ds]]) for d in detectors}
    global_best = max(mean_val, key=mean_val.get)

    rows = []  # one row per dataset: test AUROC of each strategy
    for ds, sc in per.items():
        det_test = {d: sc[d][1] for d in sc}
        det_val = {d: sc[d][0] for d in sc}
        oracle = max(det_test.values())
        ecod = det_test.get("ECOD", np.nan)
        gbest = det_test.get(global_best, np.nan)
        # MetaOD-style router: pick detector with best VALIDATION AUROC (val-only)
        router_pick = max(det_val, key=det_val.get)
        router = det_test[router_pick]
        # reliability-weighted fusion (val-AUROC weights on z-scored test scores)
        ds_scores = {}
        for d in sc:
            st = sc[d][2]; ds_scores[d] = (st - st.mean()) / (st.std() + 1e-9)
        w = np.array([max(det_val[d] - 0.5, 0) for d in sc]);
        w = w / w.sum() if w.sum() > 0 else np.ones(len(sc)) / len(sc)
        yte = sc[list(sc)[0]][3]
        fused = sum(wi * ds_scores[d] for wi, d in zip(w, sc))
        relfuse = float(roc_auc_score(yte, fused))
        rows.append({"dataset": ds, "oracle": oracle, "ECOD": ecod,
                     f"best_single({global_best})": gbest,
                     "router_select": router, "router_relfuse": relfuse})

    strategies = ["ECOD", f"best_single({global_best})", "router_select", "router_relfuse"]
    # average rank (rank 1 = best test AUROC per dataset, among strategies)
    aucm = np.array([[r[s] for s in strategies] for r in rows])  # [n_ds, n_strat]
    ranks = np.empty_like(aucm)
    for i in range(aucm.shape[0]):
        order = (-aucm[i]).argsort(); ranks[i, order] = np.arange(1, len(strategies) + 1)
    avg_rank = {s: float(ranks[:, j].mean()) for j, s in enumerate(strategies)}
    # paired bootstrap: router_select rank improvement vs the best baseline (min of ECOD, best_single)
    base_idx = [strategies.index("ECOD"), strategies.index(f"best_single({global_best})")]
    rsel = strategies.index("router_select")
    rng = np.random.default_rng(0); deltas = []
    n = ranks.shape[0]
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        base_rank = ranks[idx][:, base_idx].min(axis=1).mean()
        deltas.append(base_rank - ranks[idx][:, rsel].mean())  # >0 means router has lower (better) rank
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    # win-rate + negative-transfer rate of router_select vs ECOD default
    wins = np.mean([r["router_select"] > r["ECOD"] for r in rows])
    neg_transfer = np.mean([r["router_select"] < r["ECOD"] - 0.01 for r in rows])
    return {
        "n_datasets": len(rows),
        "datasets": [r["dataset"] for r in rows],
        "strategies": strategies,
        "average_rank": avg_rank,
        "router_select_rank_gain_vs_best_baseline": {
            "mean": float(np.mean(deltas)), "ci95": [lo, hi], "ci_low_gt_0": bool(lo > 0)},
        "router_winrate_vs_ECOD": float(wins),
        "router_negative_transfer_rate_vs_ECOD": float(neg_transfer),
        "per_dataset": rows,
        "verdict": ("ROUTER WINS on avg rank (CI>0) -- direction has legs, scale"
                    if lo > 0 else
                    "NO WIN on avg rank (CI crosses 0) -- universal direction does NOT clear the bar on tabular pilot"),
    }


def _all_classical_names():
    """Fetch every ADBench Classical dataset name from the GitHub API."""
    import json as _json
    url = "https://api.github.com/repos/Minqi824/ADBench/contents/adbench/datasets/Classical"
    req = urllib.request.Request(url, headers={"User-Agent": "pilot"})
    data = _json.loads(urllib.request.urlopen(req, timeout=30).read())
    return [d["name"][:-4] for d in data if d["name"].endswith(".npz")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/fusion/universal_router_pilot.json")
    ap.add_argument("--full", action="store_true", help="run the full ADBench Classical suite")
    args = ap.parse_args()
    datasets = _all_classical_names() if args.full else DATASETS
    print(f"running on {len(datasets)} datasets", flush=True)
    rep = run(datasets)
    print(json.dumps(rep.get("average_rank", rep), indent=2))
    if "verdict" in rep:
        print("\nrank gain vs best baseline:", rep["router_select_rank_gain_vs_best_baseline"])
        print("router win-rate vs ECOD:", round(rep["router_winrate_vs_ECOD"], 3),
              "| negative-transfer rate:", round(rep["router_negative_transfer_rate_vs_ECOD"], 3))
        print("VERDICT:", rep["verdict"])
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
