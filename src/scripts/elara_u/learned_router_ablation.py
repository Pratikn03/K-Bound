"""DECISIVE ablation: does a LEARNED router extract signal from reliability
features beyond plain validation-AUROC selection?

The hand-designed hybrid policy (evaluate_universal_contract.py) loses to plain
val-AUROC selection. That could be a weak policy, not a dead idea. This script
runs the stronger test: a leave-one-task-out gradient-boosted regressor that
predicts per-detector test AUROC from meta-features, routed to the argmax, with
a clean reliability-features-removed ABLATION:

  val_select     per-task argmax validation AUROC          (strong simple baseline)
  learned_norel  leave-task-out regressor on [detector id, val-AUROC, dataset
                 meta-features]                              (NO reliability feats)
  learned_rel    learned_norel + reliability meta-features   (the ablation pair)
  learned_rel_lfo learned_rel under LEAVE-FAMILY-OUT          (hardest generalization)
  oracle         per-task argmax test AUROC                  (upper bound)

Decisive paired-bootstrap contrasts (>0 means the first beats the second):
  learned_rel - val_select       does learned reliability routing beat the baseline?
  learned_rel - learned_norel    ABLATION: do reliability features add value at all?

Reports whatever the data says. Reliability features either earn their place or
they do not. Reads the score archive only -- no detector re-run, no test labels
used for routing (val->test KS drift uses unlabeled test scores, which is allowed).
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, ks_2samp, skew, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "experiments/elara_u/score_archive"
OUT = ROOT / "experiments/elara_u/learned_router_ablation.json"
RNG = 0
N_BOOT = 10000


def load_archive():
    tasks = []
    for f in sorted(glob.glob(str(ARCHIVE / "*.npz"))):
        if os.path.basename(f).startswith("._"):
            continue
        z = np.load(f, allow_pickle=True)
        t = {"name": os.path.basename(f).replace(".npz", ""),
             "domain": str(z["domain"]),
             "Sval": np.asarray(z["Sval"], float), "yval": np.asarray(z["yval"], int),
             "Stest": np.asarray(z["Stest"], float), "ytest": np.asarray(z["ytest"], int),
             "dets": [str(d) for d in z["det_names"]], "val_auc": np.asarray(z["val_auc"], float)}
        if len(np.unique(t["yval"])) == 2 and len(np.unique(t["ytest"])) == 2:
            tasks.append(t)
    return tasks


def _family(domain):
    return "tabular" if domain.startswith("adb") or domain == "tabular" else domain


def _gini(x):
    x = np.sort(np.abs(x)); n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


REL_KEYS = ["std", "iqr", "skew", "kurt", "topmass", "gini", "disagree", "drift"]
DS_KEYS = ["log_nval", "log_ntest", "anom_rate", "auc_spread", "mean_disagree"]


def meta_features(t):
    Sval, Stest, yval = t["Sval"], t["Stest"], t["yval"]
    nval, ndet = Sval.shape
    corr = np.eye(ndet)
    for i in range(ndet):
        for j in range(i + 1, ndet):
            c = spearmanr(Sval[:, i], Sval[:, j]).correlation
            corr[i, j] = corr[j, i] = 0.0 if np.isnan(c) else c
    rel = []
    for j in range(ndet):
        sv, st = Sval[:, j], Stest[:, j]
        q25, q75 = np.percentile(sv, [25, 75])
        rel.append({
            "val_auc": float(t["val_auc"][j]),
            "std": float(np.std(sv)), "iqr": float(q75 - q25),
            "skew": float(skew(sv)), "kurt": float(kurtosis(sv)),
            "topmass": float(np.mean(np.sort(sv)[-max(1, nval // 10):]) - np.mean(sv)),
            "gini": _gini(sv),
            "disagree": float(1.0 - (corr[j].sum() - 1.0) / max(1, ndet - 1)),
            "drift": float(ks_2samp(sv, st).statistic)})
    ds = {"log_nval": float(np.log10(nval)), "log_ntest": float(np.log10(len(t["ytest"]))),
          "anom_rate": float(np.mean(yval)),
          "auc_spread": float(t["val_auc"].max() - t["val_auc"].min()),
          "mean_disagree": float(np.mean([r["disagree"] for r in rel]))}
    return ds, rel


def feature_row(ds, rel_j, det_idx, ndet, use_rel):
    onehot = [1.0 if k == det_idx else 0.0 for k in range(ndet)]
    row = onehot + [rel_j["val_auc"]] + [ds[k] for k in DS_KEYS]
    return row + ([rel_j[k] for k in REL_KEYS] if use_rel else [])


def test_auc(t, j):
    return float(roc_auc_score(t["ytest"], t["Stest"][:, j]))


def learned_router(tasks, feats, use_rel, lfo=False):
    X, y, tk, dm = [], [], [], []
    for t in tasks:
        ds, rel = feats[t["name"]]; ndet = len(t["dets"])
        for j in range(ndet):
            X.append(feature_row(ds, rel[j], j, ndet, use_rel))
            y.append(test_auc(t, j)); tk.append(t["name"]); dm.append(t["domain"])
    X, y, tk, dm = np.array(X), np.array(y), np.array(tk), np.array(dm)
    out = {}
    for t in tasks:
        ds, rel = feats[t["name"]]; ndet = len(t["dets"])
        mask = (np.array([_family(d) for d in dm]) != _family(t["domain"])) if lfo else (tk != t["name"])
        if mask.sum() < 20:
            out[t["name"]] = test_auc(t, int(np.argmax(t["val_auc"]))); continue
        reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                            max_leaf_nodes=15, min_samples_leaf=10,
                                            random_state=RNG).fit(X[mask], y[mask])
        cand = np.array([feature_row(ds, rel[j], j, ndet, use_rel) for j in range(ndet)])
        out[t["name"]] = test_auc(t, int(np.argmax(reg.predict(cand))))
    return out


def paired_ci(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d)
    rng = np.random.default_rng(RNG)
    boot = [d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ci_low_gt_0": bool(np.percentile(boot, 2.5) > 0),
            "win_rate": float(np.mean(d > 0))}


def main():
    tasks = load_archive()
    names = [t["name"] for t in tasks]
    dets = tasks[0]["dets"]; ndet = len(dets)
    feats = {t["name"]: meta_features(t) for t in tasks}
    print(f"loaded {len(tasks)} tasks ({dict(Counter(_family(t['domain']) for t in tasks))}), "
          f"{ndet} detectors")

    per = {}
    for t in tasks:
        aucs = [test_auc(t, j) for j in range(ndet)]
        per[t["name"]] = {"oracle": float(np.max(aucs)),
                          "val_select": aucs[int(np.argmax(t["val_auc"]))],
                          **{f"fixed/{dets[j]}": aucs[j] for j in range(ndet)}}
    norel = learned_router(tasks, feats, use_rel=False)
    rel = learned_router(tasks, feats, use_rel=True)
    rel_lfo = learned_router(tasks, feats, use_rel=True, lfo=True)
    for n in names:
        per[n]["learned_norel"] = norel[n]; per[n]["learned_rel"] = rel[n]
        per[n]["learned_rel_lfo"] = rel_lfo[n]

    strategies = ["val_select", "learned_norel", "learned_rel", "learned_rel_lfo",
                  *[f"fixed/{d}" for d in dets]]
    mean_auc = {s: float(np.mean([per[n][s] for n in names])) for s in strategies + ["oracle"]}
    A = np.array([[per[n][s] for s in strategies] for n in names])
    ranks = np.empty_like(A)
    for i in range(A.shape[0]):
        ranks[i, (-A[i]).argsort()] = np.arange(1, len(strategies) + 1)
    avg_rank = {s: float(ranks[:, k].mean()) for k, s in enumerate(strategies)}
    best_fixed = min((f"fixed/{d}" for d in dets), key=lambda s: avg_rank[s])

    col = lambda s: [per[n][s] for n in names]
    decisive = {
        "rel_vs_val_select": paired_ci(col("learned_rel"), col("val_select")),
        "rel_vs_norel_ABLATION": paired_ci(col("learned_rel"), col("learned_norel")),
        "norel_vs_val_select": paired_ci(col("learned_norel"), col("val_select")),
        "rel_lfo_vs_val_select": paired_ci(col("learned_rel_lfo"), col("val_select")),
        "rel_vs_best_fixed": paired_ci(col("learned_rel"), col(best_fixed)),
        "val_select_vs_best_fixed": paired_ci(col("val_select"), col(best_fixed)),
    }
    beats = decisive["rel_vs_val_select"]["ci_low_gt_0"]
    ablation = decisive["rel_vs_norel_ABLATION"]["ci_low_gt_0"]
    if beats and ablation:
        verdict = "POSITIVE: learned reliability routing beats val-selection AND ablation confirms reliability features add the gain. Novel claim earned."
    elif decisive["rel_vs_val_select"]["mean"] > 0 and decisive["rel_vs_norel_ABLATION"]["mean"] > 0:
        verdict = "WEAK: directionally positive but neither CI clears 0 -- need more tasks/seeds before claiming novelty."
    else:
        verdict = "NEGATIVE: reliability features do NOT beat plain validation-selection even with a learned router. Novelty not demonstrated; val-selection is the honest headline."

    result = {"protocol": "ELARA_U_LEARNED_ROUTER_ABLATION_v1", "n_tasks": len(tasks),
              "families": dict(Counter(_family(t["domain"]) for t in tasks)),
              "best_fixed": best_fixed,
              "average_rank": dict(sorted(avg_rank.items(), key=lambda kv: kv[1])),
              "mean_test_auroc": mean_auc, "decisive_contrasts": decisive,
              "verdict": verdict, "per_task": per}
    OUT.write_text(json.dumps(result, indent=2))

    print(f"\n{'strategy':18}{'avg_rank':>9}{'mean_AUROC':>11}")
    for s in sorted(strategies, key=lambda s: avg_rank[s]):
        print(f"{s:18}{avg_rank[s]:9.2f}{mean_auc[s]:11.3f}")
    print(f"{'oracle':18}{'-':>9}{mean_auc['oracle']:11.3f}   best_fixed={best_fixed}")
    print("\n--- DECISIVE (mean delta, CI95, sig=CI_low>0, win) ---")
    for k, v in decisive.items():
        print(f"{k:26}{v['mean']:+.4f}  CI{[round(x,4) for x in v['ci95']]}  sig={v['ci_low_gt_0']}  win={v['win_rate']:.2f}")
    print(f"\nVERDICT: {verdict}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
