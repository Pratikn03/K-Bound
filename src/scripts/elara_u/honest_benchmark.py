"""Single source of truth for the HONEST ELARA-U paper (123 tasks, 5 families).

Supersedes the flawed results_clean/results_degraded (inert constant-offset "shift")
and the vacuous stacker_ablations (all variants identical). Computes, with NO
test-label leakage and paired bootstrap CIs over tasks:

  fixed/<det>   each detector held fixed
  auto_select   per-task argmax validation AUROC          (strong simple baseline)
  rank_mean     mean of per-detector test-score ranks       (naive ensemble)
  cw_mean       confidence-weighted mean                     (naive ensemble)
  stack         rank-normalized logistic stack, trained on VAL, applied to test
  stack_rel     stack with per-detector reliability GATING (val-AUROC weighting)
                -- the honest reliability ablation: does reliability help the stack?
  oracle        per-task argmax test AUROC                  (upper bound)

Two honest findings this establishes:
  (1) POSITIVE: stack > auto_select > best fixed detector (with CIs).
  (2) NEGATIVE: reliability gating (stack_rel) does NOT beat the plain stack
      -- consistent with learned_router_ablation.json + heterogeneous_degradation.

No test labels are used to fit any router/stacker. The stacker trains a logistic
regression on each task's VALIDATION scores+labels and is applied to that task's
test scores (standard deploy-time combiner; val labels are permitted).
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "experiments/elara_u/score_archive"
OUT = ROOT / "experiments/elara_u/honest_benchmark.json"
RNG = 0
N_BOOT = 10000


def _family(domain):
    if domain.startswith("adb") or domain == "tabular":
        return "tabular"
    return domain  # cyber, fraud, image_ood, text


def load_archive():
    tasks = []
    for f in sorted(glob.glob(str(ARCHIVE / "*.npz"))):
        if os.path.basename(f).startswith("._"):
            continue
        z = np.load(f, allow_pickle=True)
        t = {"name": os.path.basename(f)[:-4], "domain": str(z["domain"]),
             "fam": _family(str(z["domain"])),
             "Sval": np.asarray(z["Sval"], float), "yval": np.asarray(z["yval"], int),
             "Stest": np.asarray(z["Stest"], float), "ytest": np.asarray(z["ytest"], int),
             "dets": [str(d) for d in z["det_names"]], "val_auc": np.asarray(z["val_auc"], float)}
        if len(np.unique(t["yval"])) == 2 and len(np.unique(t["ytest"])) == 2:
            tasks.append(t)
    return tasks


def _ranknorm(S):
    return np.column_stack([rankdata(S[:, j]) / len(S) for j in range(S.shape[1])])


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5


def _ece(p, y, bins=10):
    p = np.clip(p, 0, 1); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p <= edges[b + 1] if b == bins - 1 else p < edges[b + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def strategies_for_task(t):
    """Return {strategy: (test_auc, test_scores_in_[0,1])} for one task. No test labels used to fit."""
    Sval, yval, Stest, ytest = t["Sval"], t["yval"], t["Stest"], t["ytest"]
    vauc = t["val_auc"]; out = {}
    # fixed detectors
    for j, d in enumerate(t["dets"]):
        out[f"fixed/{d}"] = (_auc(ytest, Stest[:, j]), Stest[:, j])
    # auto-select by validation AUROC
    js = int(np.argmax(vauc)); out["auto_select"] = (_auc(ytest, Stest[:, js]), Stest[:, js])
    # naive ensembles
    rm = _ranknorm(Stest).mean(1); out["rank_mean"] = (_auc(ytest, rm), rm)
    w = 2.0 * np.abs(Stest - 0.5); cw = (Stest * w).sum(1) / np.clip(w.sum(1), 1e-9, None)
    out["cw_mean"] = (_auc(ytest, cw), cw)
    # rank-normalized logistic stack (train on VAL, apply to TEST) -- leakage-free
    Xv, Xt = _ranknorm(Sval), _ranknorm(Stest)
    if len(np.unique(yval)) == 2:
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xv, yval)
        ps = clf.predict_proba(Xt)[:, 1]
        # reliability-GATED stack: weight each detector's rank by max(val_auc-0.5,0) before LR
        g = np.clip(vauc - 0.5, 0, None); g = g / g.sum() if g.sum() > 0 else np.ones(len(vauc)) / len(vauc)
        clf_r = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xv * g, yval)
        pr = clf_r.predict_proba(Xt * g)[:, 1]
    else:
        ps = pr = rm
    out["stack"] = (_auc(ytest, ps), ps)
    out["stack_rel"] = (_auc(ytest, pr), pr)
    # oracle
    out["oracle"] = (max(_auc(ytest, Stest[:, j]) for j in range(len(t["dets"]))), None)
    return out


def paired_ci(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d)
    rng = np.random.default_rng(RNG)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "sig": bool(np.percentile(boot, 2.5) > 0), "win_rate": float(np.mean(d > 0))}


def main():
    tasks = load_archive()
    fams = Counter(t["fam"] for t in tasks)
    print(f"loaded {len(tasks)} tasks  families={dict(fams)}")
    dets = tasks[0]["dets"]
    STRAT = ([f"fixed/{d}" for d in dets] +
             ["auto_select", "rank_mean", "cw_mean", "stack", "stack_rel", "oracle"])
    per = {t["name"]: strategies_for_task(t) for t in tasks}
    names = list(per)

    auc = {s: np.array([per[n][s][0] for n in names]) for s in STRAT}
    # average rank among non-oracle strategies
    ranked = [s for s in STRAT if s != "oracle"]
    A = np.column_stack([auc[s] for s in ranked])
    R = np.zeros_like(A)
    for i in range(A.shape[0]):
        R[i] = rankdata(-A[i], method="average")
    avg_rank = {s: float(R[:, k].mean()) for k, s in enumerate(ranked)}
    mean_auc = {s: float(auc[s].mean()) for s in STRAT}
    mean_regret = {s: float((auc["oracle"] - auc[s]).mean()) for s in STRAT}
    best_fixed = min((f"fixed/{d}" for d in dets), key=lambda s: avg_rank[s])

    # ECE per strategy (probabilistic strategies only)
    ece = {}
    for s in ["auto_select", "rank_mean", "cw_mean", "stack", "stack_rel"]:
        ece[s] = float(np.mean([_ece(per[n][s][1], next(t for t in tasks if t["name"] == n)["ytest"])
                                for n in names]))

    # per-family mean rank for the headline strategies
    fam_rank = defaultdict(dict)
    for fam in fams:
        idx = [i for i, n in enumerate(names) if next(t for t in tasks if t["name"] == n)["fam"] == fam]
        for s in ranked:
            fam_rank[fam][s] = float(R[idx][:, ranked.index(s)].mean())

    # negative-transfer rate (strategy worse than best fixed by >0.01)
    neg_tr = {s: float(np.mean(auc[s] < auc[best_fixed] - 0.01))
              for s in ["auto_select", "stack", "stack_rel"]}

    contrasts = {
        "stack_vs_auto_select": paired_ci(auc["stack"], auc["auto_select"]),
        "stack_rel_vs_stack_ABLATION": paired_ci(auc["stack_rel"], auc["stack"]),
        "stack_vs_best_fixed": paired_ci(auc["stack"], auc[best_fixed]),
        "auto_select_vs_best_fixed": paired_ci(auc["auto_select"], auc[best_fixed]),
    }
    worst_rank = {s: int(R[:, ranked.index(s)].max()) for s in ranked}
    per_task_auc = {s: auc[s].tolist() for s in STRAT if s != "oracle"}
    per_task_rank = {s: R[:, ranked.index(s)].tolist() for s in ranked}
    task_families = [t["fam"] for t in tasks]
    result = {
        "protocol": "ELARA_U_HONEST_BENCHMARK_v1", "n_tasks": len(tasks),
        "families": dict(fams), "best_fixed": best_fixed,
        "average_rank": dict(sorted(avg_rank.items(), key=lambda kv: kv[1])),
        "mean_auroc": mean_auc, "mean_regret": mean_regret, "mean_ece": ece,
        "worst_rank": worst_rank, "per_family_rank": dict(fam_rank),
        "negative_transfer_rate": neg_tr,
        "per_task_auc": per_task_auc, "per_task_rank": per_task_rank,
        "task_families": task_families,
        "contrasts": contrasts,
        "verdict": {
            "positive_stack_beats_select": contrasts["stack_vs_auto_select"]["sig"],
            "positive_select_beats_fixed": contrasts["auto_select_vs_best_fixed"]["sig"],
            "reliability_gating_helps_stack": contrasts["stack_rel_vs_stack_ABLATION"]["sig"],
        },
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(f"\n{'strategy':16}{'avg_rank':>9}{'mean_AUC':>9}{'regret':>8}{'ECE':>7}")
    for s in sorted(STRAT, key=lambda s: avg_rank.get(s, -1)):
        ar = f"{avg_rank[s]:9.2f}" if s in avg_rank else f"{'-':>9}"
        ec = f"{ece[s]:7.3f}" if s in ece else f"{'-':>7}"
        print(f"{s:16}{ar}{mean_auc[s]:9.3f}{mean_regret[s]:8.3f}{ec}")
    print(f"\nbest_fixed={best_fixed}")
    print("\n--- contrasts (mean delta, CI95, sig, win) ---")
    for k, v in contrasts.items():
        print(f"  {k:30}{v['mean']:+.4f}  CI{[round(x,4) for x in v['ci95']]}  sig={v['sig']}  win={v['win_rate']:.2f}")
    print("\nper-family avg rank (stack / auto_select / best_fixed):")
    for fam in fams:
        print(f"  {fam:10} stack={fam_rank[fam]['stack']:.2f}  auto_select={fam_rank[fam]['auto_select']:.2f}  "
              f"{best_fixed}={fam_rank[fam][best_fixed]:.2f}")
    print(f"\nHONEST VERDICT: stack>select={result['verdict']['positive_stack_beats_select']}, "
          f"select>fixed={result['verdict']['positive_select_beats_fixed']}, "
          f"reliability_helps_stack={result['verdict']['reliability_gating_helps_stack']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
