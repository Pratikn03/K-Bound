"""DECISIVE round 2: does reliability routing beat val-selection under HETEROGENEOUS,
per-task deployment degradation -- the one regime where reliability has structural
signal that detector-identity priors and stale validation-AUROC cannot capture?

The uniform-shift test (shift_stress_ablation.py) showed the gain under drift comes
from detector-IDENTITY priors, not reliability features: uniform corruption makes
"which detector survives" architecture-determined, so identity captures it. That is
the wrong failure mode for reliability.

Here the degradation is HETEROGENEOUS and TASK-SPECIFIC: on each task an independent
random subset of features goes MISSING (imputed to the train mean) in the TEST split
only. Which detector this hurts depends on that task's random feature subset, so it
is NOT predictable from detector identity or from clean validation-AUROC. The only
way to route correctly is to SENSE per-task drift / disagreement at deployment --
exactly what reliability features are for. This is unlabeled deployment monitoring
(val->test KS drift + test-score dispersion/disagreement), NOT pure validation-only
routing, and is labelled as such.

  val_select     pick by CLEAN validation AUROC, evaluate on degraded test (blind)
  learned_norel  leave-task-out regressor on [det id, val-AUROC, dataset feats]
  learned_rel    + reliability features on degraded test (drift/dispersion/disagree)
  oracle         best detector on degraded test (upper bound)

Decisive: learned_rel - val_select (does reliability beat the baseline under task-
specific degradation?) and learned_rel - learned_norel (ABLATION: do reliability
features carry the gain, given identity priors are already available?). No test
labels are used for routing. Reports whatever the data says.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, ks_2samp, skew, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.elara_u.gate_u_seed_eval import RNG, detector_zoo, load_tasks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/elara_u/heterogeneous_degradation_ablation.json"
MISS_FRACS = [0.0, 0.3, 0.5, 0.7]     # fraction of features that go missing on test
N_BOOT = 10000
REL_KEYS = ["std", "iqr", "skew", "kurt", "topmass", "gini", "disagree", "drift"]
DS_KEYS = ["log_nval", "log_ntest", "anom_rate", "auc_spread", "mean_disagree"]


def _family(domain):
    return "tabular" if domain.startswith("adb") or domain == "tabular" else domain


def _gini(x):
    x = np.sort(np.abs(x)); n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def degrade(Xte_scaled, frac, seed):
    """Heterogeneous per-task missingness: a random feature subset -> mean (0 in the
    fitted standardized space). Different subset per task (rng seeded by task hash)."""
    if frac <= 0:
        return Xte_scaled
    rng = np.random.default_rng(seed)
    d = Xte_scaled.shape[1]
    k = max(1, int(round(frac * d)))
    cols = rng.choice(d, k, replace=False)
    Xc = Xte_scaled.copy()
    Xc[:, cols] = 0.0           # standardized mean == imputed missing value
    return Xc


def score_task(name, X, y, frac):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=RNG, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=RNG, stratify=ytmp)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    # per-task missingness seed: deterministic + distinct per task name (stable
    # across runs, unlike Python's hash() which is process-randomized)
    seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)
    Xte_s = degrade(sc.transform(Xte), frac, seed=seed)
    dets, vauc, tauc, sval, stest = [], [], [], [], []
    for dn, ctor in detector_zoo().items():
        try:
            m = ctor().fit(Xtr_s)
            sv = m.decision_function(Xva_s); st = m.decision_function(Xte_s)
            va = roc_auc_score(yva, sv); ta = roc_auc_score(yte, st)
        except Exception:
            sv = np.full(len(yva), 0.0); st = np.full(len(yte), 0.0); va = ta = 0.5
        dets.append(dn); vauc.append(float(va)); tauc.append(float(ta))
        sval.append(sv); stest.append(st)
    return {"dets": dets, "val_auc": np.array(vauc), "test_auc": np.array(tauc),
            "yval": yva, "ytest": yte, "Sval": np.column_stack(sval), "Stest": np.column_stack(stest)}


def meta_features(rec):
    Sval, Stest = rec["Sval"], rec["Stest"]; ndet = Stest.shape[1]
    corr = np.eye(ndet)
    for i in range(ndet):
        for j in range(i + 1, ndet):
            c = spearmanr(Stest[:, i], Stest[:, j]).correlation
            corr[i, j] = corr[j, i] = 0.0 if np.isnan(c) else c
    rel = []
    for j in range(ndet):
        st = Stest[:, j]; q25, q75 = np.percentile(st, [25, 75])
        rel.append({"val_auc": float(rec["val_auc"][j]),
                    "std": float(np.std(st)), "iqr": float(q75 - q25),
                    "skew": float(skew(st)), "kurt": float(kurtosis(st)),
                    "topmass": float(np.mean(np.sort(st)[-max(1, len(st) // 10):]) - np.mean(st)),
                    "gini": _gini(st),
                    "disagree": float(1.0 - (corr[j].sum() - 1.0) / max(1, ndet - 1)),
                    "drift": float(ks_2samp(Sval[:, j], st).statistic)})
    ds = {"log_nval": float(np.log10(len(rec["yval"]))), "log_ntest": float(np.log10(len(rec["ytest"]))),
          "anom_rate": float(np.mean(rec["yval"])),
          "auc_spread": float(rec["val_auc"].max() - rec["val_auc"].min()),
          "mean_disagree": float(np.mean([r["disagree"] for r in rel]))}
    return ds, rel


def feature_row(ds, rel_j, det_idx, ndet, use_rel):
    row = [1.0 if k == det_idx else 0.0 for k in range(ndet)] + [rel_j["val_auc"]] + [ds[k] for k in DS_KEYS]
    return row + ([rel_j[k] for k in REL_KEYS] if use_rel else [])


def learned_router(recs, feats, doms, use_rel):
    names = list(recs); X, y, tk = [], [], []
    for n in names:
        ds, rel = feats[n]; ndet = len(recs[n]["dets"])
        for j in range(ndet):
            X.append(feature_row(ds, rel[j], j, ndet, use_rel)); y.append(recs[n]["test_auc"][j]); tk.append(n)
    X, y, tk = np.array(X), np.array(y), np.array(tk)
    out = {}
    for n in names:
        ds, rel = feats[n]; ndet = len(recs[n]["dets"]); mask = tk != n
        if mask.sum() < 20:
            out[n] = float(recs[n]["test_auc"][int(np.argmax(recs[n]["val_auc"]))]); continue
        reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=10, random_state=RNG).fit(X[mask], y[mask])
        cand = np.array([feature_row(ds, rel[j], j, ndet, use_rel) for j in range(ndet)])
        out[n] = float(recs[n]["test_auc"][int(np.argmax(reg.predict(cand)))])
    return out


def paired_ci(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d)
    rng = np.random.default_rng(RNG)
    boot = [d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ci_low_gt_0": bool(np.percentile(boot, 2.5) > 0), "win_rate": float(np.mean(d > 0))}


def main():
    tasks = load_tasks()
    print(f"loaded {len(tasks)} raw tasks ({dict(Counter(_family(d) for _, d, _, _ in tasks))})")
    results = {}
    for frac in MISS_FRACS:
        recs, doms = {}, {}
        for name, dom, X, y in tasks:
            r = score_task(name, X, y, frac)
            if len(np.unique(r["yval"])) < 2 or len(np.unique(r["ytest"])) < 2:
                continue
            recs[name] = r; doms[name] = dom
        feats = {n: meta_features(recs[n]) for n in recs}; names = list(recs)
        val_sel = {n: float(recs[n]["test_auc"][int(np.argmax(recs[n]["val_auc"]))]) for n in names}
        oracle = {n: float(np.max(recs[n]["test_auc"])) for n in names}
        norel = learned_router(recs, feats, doms, use_rel=False)
        rel = learned_router(recs, feats, doms, use_rel=True)
        col = lambda d: [d[n] for n in names]
        contrasts = {"rel_vs_val_select": paired_ci(col(rel), col(val_sel)),
                     "rel_vs_norel_ABLATION": paired_ci(col(rel), col(norel)),
                     "norel_vs_val_select": paired_ci(col(norel), col(val_sel))}
        results[f"missing_{frac}"] = {
            "n_tasks": len(names),
            "mean_auroc": {"val_select": float(np.mean(col(val_sel))), "learned_norel": float(np.mean(col(norel))),
                           "learned_rel": float(np.mean(col(rel))), "oracle": float(np.mean(col(oracle)))},
            "contrasts": contrasts}
        m = results[f"missing_{frac}"]["mean_auroc"]
        print(f"\n=== missing_frac {frac} ({len(names)} tasks) ===")
        print(f"  val_select={m['val_select']:.3f} norel={m['learned_norel']:.3f} "
              f"rel={m['learned_rel']:.3f} oracle={m['oracle']:.3f}")
        for k, v in contrasts.items():
            print(f"  {k:24}{v['mean']:+.4f} CI[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}] "
                  f"sig={v['ci_low_gt_0']} win={v['win_rate']:.2f}")

    won = [f for f in MISS_FRACS if f > 0
           and results[f"missing_{f}"]["contrasts"]["rel_vs_val_select"]["ci_low_gt_0"]
           and results[f"missing_{f}"]["contrasts"]["rel_vs_norel_ABLATION"]["ci_low_gt_0"]]
    if won:
        verdict = (f"BREAKTHROUGH CORE: at missing_frac {won} reliability routing beats val-selection "
                   "AND the ablation confirms reliability features carry the gain. Scale this regime.")
    else:
        directional = [f for f in MISS_FRACS if f > 0
                       and results[f"missing_{f}"]["contrasts"]["rel_vs_norel_ABLATION"]["mean"] > 0]
        verdict = ("RELIABILITY STILL DEAD: reliability features do not significantly beat the no-"
                   "reliability router under heterogeneous missingness"
                   + (f" (ablation directionally positive at {directional}, CI crosses 0)" if directional else "")
                   + ". This was the strongest honest regime; reliability routing is robustly not supported.")
    results["verdict"] = verdict
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nVERDICT: {verdict}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
