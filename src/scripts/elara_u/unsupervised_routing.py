"""Unsupervised meta-combination: can label-free reliability heuristics beat
label-free averaging? (The one regime where reliability could be a real moat.)

No labels are used by ANY method (test labels are used only to score AUROC). For
each task we combine the detector test scores with: average, rank-mean, max, and
three reliability-heuristic combiners (sharpness-weighted, consensus-weighted,
and a degenerate-guarded average). If a reliability heuristic beats plain average
and rank-mean with a paired-bootstrap CI excluding zero, that is a genuine
unsupervised contribution; if not, reliability is inert even without labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "experiments/elara_u/score_archive"


def _ranks(S):
    return np.argsort(np.argsort(S, axis=0), axis=0) / max(S.shape[0] - 1, 1)


def avg(S):            # plain average
    return S.mean(1)


def rank_mean(S):
    return _ranks(S).mean(1)


def smax(S):
    return S.max(1)


def sharp_w(S):        # weight by per-detector sharpness (confidence), label-free
    w = np.abs(S - 0.5).mean(0)
    w = w / w.sum() if w.sum() > 1e-9 else np.ones(S.shape[1]) / S.shape[1]
    return S @ w


def consensus_w(S):    # weight by agreement with the ensemble consensus, label-free
    R = _ranks(S)
    consensus = R.mean(1, keepdims=True)
    # agreement = -mean abs rank distance to consensus (closer = more reliable)
    agree = -np.abs(R - consensus).mean(0)
    agree = agree - agree.min() + 1e-6
    w = agree / agree.sum()
    return S @ w


def guarded_avg(S):    # drop near-constant (saturated) channels, then average
    keep = S.std(0) >= 0.02
    if not keep.any():
        return S.mean(1)
    return S[:, keep].mean(1)


METHODS = {"avg": avg, "rank_mean": rank_mean, "max": smax,
           "sharp_w": sharp_w, "consensus_w": consensus_w, "guarded_avg": guarded_avg}


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    per = {m: [] for m in METHODS}
    n = 0
    for f in sorted(ARCHIVE.glob("*.npz")):
        if f.name.startswith("._"):
            continue
        z = np.load(f, allow_pickle=True)
        S, y = z["Stest"], z["ytest"].astype(int)
        if len(np.unique(y)) < 2:
            continue
        for m, fn in METHODS.items():
            per[m].append(float(roc_auc_score(y, fn(S))))
        n += 1
    pa = {m: np.array(v) for m, v in per.items()}
    mean_auc = {m: round(float(v.mean()), 4) for m, v in pa.items()}

    # the test: do reliability heuristics beat the plain unsupervised baselines?
    contrasts = {}
    for rel in ["sharp_w", "consensus_w", "guarded_avg"]:
        for base in ["avg", "rank_mean"]:
            mean, lo, hi = _boot(pa[rel] - pa[base])
            contrasts[f"{rel}_vs_{base}"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                             "ci_excludes_zero": lo > 0 or hi < 0}
    best_rel = max(["sharp_w", "consensus_w", "guarded_avg"], key=lambda m: mean_auc[m])
    win = any(contrasts[f"{best_rel}_vs_{b}"]["mean"] > 0 and contrasts[f"{best_rel}_vs_{b}"]["ci_excludes_zero"]
              for b in ["avg", "rank_mean"])

    res = {"protocol": "UNSUPERVISED_ROUTING_v1", "n_tasks": n,
           "mean_auroc": mean_auc, "contrasts": contrasts,
           "best_reliability_heuristic": best_rel,
           "reliability_beats_unsupervised_baselines": bool(win)}
    out = ROOT / "experiments/elara_u/unsupervised_routing.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"=== UNSUPERVISED meta-combination ({n} tasks, no labels used by any method) ===")
    for m in METHODS:
        print(f"  {m:13} mean AUROC {mean_auc[m]:.4f}")
    print("\ncontrasts (reliability heuristic - baseline):")
    for k, v in contrasts.items():
        print(f"  {k:24} {v['mean']:+.4f} CI {v['ci95']} excl0={v['ci_excludes_zero']}")
    print(f"\nVERDICT: reliability beats unsupervised baselines? {'YES' if win else 'NO'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
