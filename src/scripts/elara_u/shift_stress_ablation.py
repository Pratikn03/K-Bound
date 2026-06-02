"""NOVEL-OR-DEAD test: does reliability routing beat val-selection UNDER SHIFT?

On i.i.d. val/test splits, validation-AUROC is the sufficient statistic and
reliability features are redundant by construction -- the learned-router ablation
confirmed they add nothing (learned_router_ablation.py). The reliability premise
(drift gates) only has a chance to matter when the TEST distribution differs from
VALIDATION, so that stale val-AUROC mis-ranks detectors and reliability-drift
features carry independent information.

This script induces a controlled, detector-agnostic distribution shift on the TEST
features (validation stays clean = design-time distribution; test = deployment
drift), re-scores the detector zoo, and runs the IDENTICAL ablation at each
severity:
  val_select     pick detector by CLEAN validation AUROC, evaluate on SHIFTED test
  learned_norel  leave-task-out regressor on [det id, val-AUROC, dataset feats]
  learned_rel    + reliability features measured at deployment (shifted-test score
                 dispersion/disagreement + val->test KS drift)  <-- the ablation
  oracle         best detector on shifted test (upper bound)

Decisive at each severity: learned_rel - val_select (does reliability win under
drift?) and learned_rel - learned_norel (ABLATION: do reliability features carry
the gain?). Detectors are fit ONCE on clean train; only test scoring repeats per
severity. No test labels are used for routing.
"""

from __future__ import annotations

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
OUT = ROOT / "experiments/elara_u/shift_stress_ablation.json"
SEVERITIES = [0.0, 1.0, 2.0, 3.0]   # 0.0 = sanity (should reproduce the i.i.d. null)
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


def corrupt(Xte, severity, seed):
    """Detector-agnostic deployment drift: scale + additive noise on a random
    feature subset (sensor gain drift + degradation). Same corruption seen by all
    detectors. severity 0 -> identity."""
    if severity <= 0:
        return Xte
    rng = np.random.default_rng(seed)
    d = Xte.shape[1]
    k = max(1, int(round(0.5 * d)))                       # half the features drift
    cols = rng.choice(d, k, replace=False)
    Xc = Xte.copy().astype(float)
    sd = Xc[:, cols].std(0) + 1e-9
    gain = 1.0 + 0.25 * severity * rng.standard_normal(k)  # multiplicative gain drift
    Xc[:, cols] = Xc[:, cols] * gain + 0.5 * severity * sd * rng.standard_normal((len(Xc), k))
    return Xc


def score_task(X, y, severity):
    """Fit zoo on clean train; return per-detector (val_auc_clean, test_auc_shift,
    clean val scores, shifted test scores) for one severity."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=RNG, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=RNG, stratify=ytmp)
    sc = StandardScaler().fit(Xtr)
    Xva_s = sc.transform(Xva)
    Xte_s = sc.transform(corrupt(Xte, severity, seed=RNG + 7))   # corrupt raw, then scale
    Xtr_s = sc.transform(Xtr)
    dets, vauc, tauc, sval, stest = [], [], [], [], []
    for name, ctor in detector_zoo().items():
        try:
            m = ctor().fit(Xtr_s)
            sv = m.decision_function(Xva_s); st = m.decision_function(Xte_s)
            va = roc_auc_score(yva, sv); ta = roc_auc_score(yte, st)
        except Exception:
            sv = np.full(len(yva), 0.0); st = np.full(len(yte), 0.0); va = ta = 0.5
        dets.append(name); vauc.append(float(va)); tauc.append(float(ta))
        sval.append(sv); stest.append(st)
    return {"dets": dets, "val_auc": np.array(vauc), "test_auc": np.array(tauc),
            "yval": yva, "ytest": yte, "Sval": np.column_stack(sval),
            "Stest": np.column_stack(stest)}


def meta_features(rec):
    """Reliability features from the UNLABELED shifted-test scores + val->test drift."""
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
    ds = {"log_nval": float(np.log10(len(rec["yval"]))),
          "log_ntest": float(np.log10(len(rec["ytest"]))),
          "anom_rate": float(np.mean(rec["yval"])),
          "auc_spread": float(rec["val_auc"].max() - rec["val_auc"].min()),
          "mean_disagree": float(np.mean([r["disagree"] for r in rel]))}
    return ds, rel


def feature_row(ds, rel_j, det_idx, ndet, use_rel):
    row = [1.0 if k == det_idx else 0.0 for k in range(ndet)] + [rel_j["val_auc"]] + [ds[k] for k in DS_KEYS]
    return row + ([rel_j[k] for k in REL_KEYS] if use_rel else [])


def learned_router(recs, feats, doms, use_rel, lfo=False):
    names = list(recs)
    X, y, tk, dm = [], [], [], []
    for n in names:
        ds, rel = feats[n]; ndet = len(recs[n]["dets"])
        for j in range(ndet):
            X.append(feature_row(ds, rel[j], j, ndet, use_rel))
            y.append(recs[n]["test_auc"][j]); tk.append(n); dm.append(doms[n])
    X, y, tk, dm = np.array(X), np.array(y), np.array(tk), np.array(dm)
    fam = np.array([_family(d) for d in dm])
    out = {}
    for n in names:
        ds, rel = feats[n]; ndet = len(recs[n]["dets"])
        mask = (fam != _family(doms[n])) if lfo else (tk != n)
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
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ci_low_gt_0": bool(np.percentile(boot, 2.5) > 0), "win_rate": float(np.mean(d > 0))}


def main():
    tasks = load_tasks()
    print(f"loaded {len(tasks)} raw tasks ({dict(Counter(_family(d) for _, d, _, _ in tasks))})")
    results = {}
    for sev in SEVERITIES:
        recs, doms = {}, {}
        for name, dom, X, y in tasks:
            r = score_task(X, y, sev)
            if len(np.unique(r["yval"])) < 2 or len(np.unique(r["ytest"])) < 2:
                continue
            recs[name] = r; doms[name] = dom
        feats = {n: meta_features(recs[n]) for n in recs}
        names = list(recs)
        val_sel = {n: float(recs[n]["test_auc"][int(np.argmax(recs[n]["val_auc"]))]) for n in names}
        oracle = {n: float(np.max(recs[n]["test_auc"])) for n in names}
        norel = learned_router(recs, feats, doms, use_rel=False)
        rel = learned_router(recs, feats, doms, use_rel=True)
        col = lambda d: [d[n] for n in names]
        contrasts = {
            "rel_vs_val_select": paired_ci(col(rel), col(val_sel)),
            "rel_vs_norel_ABLATION": paired_ci(col(rel), col(norel)),
            "norel_vs_val_select": paired_ci(col(norel), col(val_sel)),
        }
        results[f"severity_{sev}"] = {
            "n_tasks": len(names),
            "mean_auroc": {"val_select": float(np.mean(col(val_sel))),
                           "learned_norel": float(np.mean(col(norel))),
                           "learned_rel": float(np.mean(col(rel))),
                           "oracle": float(np.mean(col(oracle)))},
            "contrasts": contrasts}
        print(f"\n=== severity {sev} ({len(names)} tasks) ===")
        m = results[f"severity_{sev}"]["mean_auroc"]
        print(f"  val_select={m['val_select']:.3f}  learned_norel={m['learned_norel']:.3f}  "
              f"learned_rel={m['learned_rel']:.3f}  oracle={m['oracle']:.3f}")
        for k, v in contrasts.items():
            print(f"  {k:24}{v['mean']:+.4f}  CI{[round(x,4) for x in v['ci95']]}  "
                  f"sig={v['ci_low_gt_0']}  win={v['win_rate']:.2f}")

    # verdict: reliability wins under drift if at some severity>0 BOTH rel>val_select
    # and the ablation rel>norel clear CI>0.
    won = [s for s in SEVERITIES if s > 0
           and results[f"severity_{s}"]["contrasts"]["rel_vs_val_select"]["ci_low_gt_0"]
           and results[f"severity_{s}"]["contrasts"]["rel_vs_norel_ABLATION"]["ci_low_gt_0"]]
    if won:
        verdict = (f"POSITIVE under drift: at severity {won} reliability routing beats "
                   "val-selection AND the ablation confirms reliability features carry the gain. "
                   "Novel claim earned -- scoped to distribution shift.")
    else:
        directional = [s for s in SEVERITIES if s > 0
                       and results[f"severity_{s}"]["contrasts"]["rel_vs_val_select"]["mean"] > 0]
        verdict = ("NEGATIVE even under drift: reliability routing does not significantly beat "
                   "val-selection at any tested severity"
                   + (f" (directionally positive at {directional} but CI crosses 0)" if directional else "")
                   + ". The honest headline is the strong validation-selection baseline + the negative reliability result.")
    results["verdict"] = verdict
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nVERDICT: {verdict}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
