"""Gate U seed evaluation: a reliability-aware anomaly META-ROUTER over a detector zoo.

DEVELOPMENT SEED for the ELARA-U program (research_lock/GATE_U_CROSS_DOMAIN_PROTOCOL_v1.yaml).
Proves the core thesis on cross-domain data already on disk: no single detector
wins everywhere (negative transfer), and a validation-only router that selects the
right detector per task improves average rank and reduces worst-case regret.

Protocol (no test labels used for any selection):
  per task -> standardize -> split train(fit, unlabeled)/val(labeled, router
  selection)/test(labeled, eval). Fit each pyod detector on train; score val+test.
  Router picks the detector with best VALIDATION AUROC. Compare router vs each fixed
  detector by mean rank, worst-case rank (negative transfer), and regret-to-oracle,
  with a paired bootstrap over tasks.

NOT an official Gate U pass: the official sweep needs the frozen MVS families
(ADBench/TSB-AD/OpenOOD/MVTec AD 2/Real-IAD D3), several not yet downloaded.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
RNG = 0
CAP = 8000  # total rows per task (subsample for speed across ~54 tasks)


def _balance(X, y, cap=CAP, seed=RNG):
    """Cap total rows at `cap` (for speed), preserving the anomaly rate, with a
    floor of 30 positives so val/test splits stay valid for AUROC."""
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    n_pos = min(len(pos), max(int(round(cap * len(pos) / len(y))), 30))
    n_neg = min(len(neg), cap - n_pos)
    keep = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, n_neg, replace=False)])
    keep = rng.permutation(keep)
    return X[keep], y[keep]


def _numeric(df, drop):
    num = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop if c in df.columns],
                                                     errors="ignore")
    return num.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def load_tasks():
    """Yield (task_name, domain, X[n,d], y[n]) for cross-domain anomaly tasks on disk."""
    tasks = []
    # Fraud: creditcard
    d = pd.read_csv(RAW / "fraud/creditcard.csv")
    tasks.append(("creditcard", "fraud", _numeric(d, ["Time", "Class"]), d["Class"].to_numpy()))
    # Tabular behavior: online shoppers (rare-purchase as anomaly)
    d = pd.read_csv(RAW / "behavior/online_shoppers_intention.csv")
    tasks.append(("online_shoppers", "tabular", _numeric(d, ["Revenue"]),
                  d["Revenue"].astype(int).to_numpy()))
    # Cyber: UNSW-NB15 full + per-attack-category sub-tasks vs Normal
    d = pd.read_csv(RAW / "cyber/UNSW-NB15.csv")
    feat_drop = ["id", "label", "attack_cat"]
    tasks.append(("unsw_full", "cyber", _numeric(d, feat_drop), d["label"].to_numpy()))
    if "attack_cat" in d.columns:
        d["attack_cat"] = d["attack_cat"].astype(str).str.strip()
        normal = d[d["attack_cat"].str.lower() == "normal"]
        for cat in ["Exploits", "Fuzzers", "DoS", "Reconnaissance"]:
            sub = d[d["attack_cat"] == cat]
            if len(sub) < 100 or len(normal) < 100:
                continue
            both = pd.concat([normal, sub])
            y = (both["attack_cat"] == cat).astype(int).to_numpy()
            tasks.append((f"unsw_{cat.lower()}", "cyber", _numeric(both, feat_drop), y))
    # ADBench feature-vector benchmarks: tabular (Classical), image-OOD (CV/ResNet18),
    # text (NLP/BERT). Each .npz is one task; all drop into the same (X,y) pipeline.
    import glob
    import os
    for subdir, domain, prefix in [("adbench", "adbench", "adb_"),
                                    ("adbench_cv", "image_ood", "cv_"),
                                    ("adbench_nlp", "text", "nlp_")]:
        for f in sorted(glob.glob(str(RAW / subdir / "*.npz"))):
            if os.path.basename(f).startswith("._"):
                continue
            try:
                z = np.load(f)
                Xa, ya = np.nan_to_num(z["X"].astype(float)), z["y"].astype(int)
            except Exception:
                continue
            if Xa.shape[0] >= 80 and int(ya.sum()) >= 12 and len(np.unique(ya)) == 2:
                tasks.append((prefix + os.path.basename(f).replace(".npz", ""), domain, Xa, ya))

    out = []
    for name, dom, X, y in tasks:
        if X.shape[0] < 80 or len(np.unique(y)) < 2:
            continue
        X, y = _balance(X, y)
        out.append((name, dom, X, y))
    return out


def detector_zoo():
    return {
        "ECOD": lambda: ECOD(),
        "COPOD": lambda: COPOD(),
        "IForest": lambda: IForest(random_state=RNG),
        "LOF": lambda: LOF(),
        "KNN": lambda: KNN(),
        "OCSVM": lambda: OCSVM(),
    }


def eval_task(X, y):
    """Return {detector: (val_auc, test_auc)} for one task."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=RNG, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=RNG, stratify=ytmp)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    res = {}
    for name, ctor in detector_zoo().items():
        try:
            m = ctor().fit(Xtr)
            va = roc_auc_score(yva, m.decision_function(Xva))
            te = roc_auc_score(yte, m.decision_function(Xte))
            res[name] = (float(va), float(te))
        except Exception as e:  # keep the sweep robust
            res[name] = (0.5, 0.5)
            print(f"    [{name}] failed: {e}")
    return res


def main() -> int:
    tasks = load_tasks()
    dets = list(detector_zoo().keys())
    rows = {}  # task -> {det: (val,test)}, plus router
    for name, dom, X, y in tasks:
        t0 = time.time()
        r = eval_task(X, y)
        router_det = max(dets, key=lambda d: r[d][0])      # validation-only selection
        r["ROUTER"] = (r[router_det][0], r[router_det][1])
        r["_router_choice"] = router_det
        rows[name] = {"domain": dom, "scores": r}
        te = {d: round(r[d][1], 3) for d in dets + ["ROUTER"]}
        print(f"[{name:18}] ({dom}) test AUROC {te}  router->{router_det} ({time.time()-t0:.0f}s)",
              flush=True)

    methods = dets + ["ROUTER"]
    # per-task ranks (1=best test AUROC) and regret-to-oracle
    ranks = {m: [] for m in methods}
    regret = {m: [] for m in methods}
    test_auc = {m: [] for m in methods}
    for name, info in rows.items():
        sc = info["scores"]
        order = sorted(methods, key=lambda m: sc[m][1], reverse=True)
        oracle = sc[order[0]][1]
        for m in methods:
            ranks[m].append(1 + order.index(m))
            regret[m].append(oracle - sc[m][1])
            test_auc[m].append(sc[m][1])

    def agg(d):
        return {m: round(float(np.mean(v)), 4) for m, v in d.items()}

    mean_rank = agg(ranks)
    worst_rank = {m: int(np.max(v)) for m, v in ranks.items()}   # negative-transfer proxy
    mean_regret = agg(regret)
    mean_auc = agg(test_auc)
    best_fixed = min((m for m in dets), key=lambda m: mean_rank[m])

    # paired bootstrap over tasks: ROUTER mean rank - best_fixed mean rank
    rr, bf = np.array(ranks["ROUTER"]), np.array(ranks[best_fixed])
    rng = np.random.default_rng(RNG)
    diffs = [np.mean(rr[i]) - np.mean(bf[i])
             for i in (rng.integers(0, len(rr), len(rr)) for _ in range(10000))]
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    result = {
        "protocol": "GATE_U_SEED_EVAL_v1",
        "status": "DEVELOPMENT_SEED_not_official_gate_pass",
        "n_tasks": len(rows), "detectors": dets,
        "mean_rank": mean_rank, "worst_case_rank": worst_rank,
        "mean_test_auroc": mean_auc, "mean_regret_to_oracle": mean_regret,
        "best_fixed_detector": best_fixed,
        "router_vs_best_fixed_rank_delta": round(mean_rank["ROUTER"] - mean_rank[best_fixed], 4),
        "router_vs_best_fixed_rank_ci95": ci,
        "router_choices": {n: rows[n]["scores"]["_router_choice"] for n in rows},
        "per_task": {n: {"domain": rows[n]["domain"],
                         "test_auroc": {m: round(rows[n]["scores"][m][1], 4) for m in methods}}
                     for n in rows},
    }
    out = ROOT / "experiments/elara_u/gate_u_seed_eval.json"
    out.write_text(json.dumps(result, indent=2))

    print("\n=== GATE U SEED (development) ===")
    print(f"tasks={len(rows)}  detectors={dets}")
    print(f"{'method':9} {'mean_rank':>9} {'worst_rank':>10} {'mean_AUROC':>10} {'mean_regret':>11}")
    for m in methods:
        tag = "  <-- ROUTER" if m == "ROUTER" else ("  (best fixed)" if m == best_fixed else "")
        print(f"{m:9} {mean_rank[m]:9.3f} {worst_rank[m]:10d} {mean_auc[m]:10.3f} "
              f"{mean_regret[m]:11.4f}{tag}")
    print(f"\nROUTER mean-rank delta vs best fixed ({best_fixed}): "
          f"{result['router_vs_best_fixed_rank_delta']:+.3f} (lower=better), CI95 {ci}")
    print(f"worst-case rank: ROUTER={worst_rank['ROUTER']} vs best-fixed={worst_rank[best_fixed]} "
          "(negative-transfer robustness)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
