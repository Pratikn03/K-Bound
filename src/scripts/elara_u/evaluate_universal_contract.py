"""Evaluate the ELARA-U Universal Evidence Contract on the score archive.

Loads experiments/elara_u/score_archive/*.npz and compares, by cross-task rank /
regret / negative-transfer / calibration with a paired bootstrap:
  fixed detectors | auto-select (val-AUROC argmax, the META-BASELINE) |
  CW-mean | rank-mean | ELARA-U fuse | ELARA-U hybrid (leave-family-out policy).

The headline test is ELARA-U hybrid vs the auto-select meta-baseline and vs the
best fixed detector. Policy thresholds for hybrid are fit on meta-train families
(validation only) and applied to the held-out family (leave-family-out).
DEVELOPMENT result until the full MVS families are acquired.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from uais.elara_u.contract import (
    bootstrap_delta,
    ece,
    negative_transfer_rate,
    ranks_and_regret,
)
from uais.elara_u.router import RouterPolicy, fuse, route, select

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "experiments/elara_u/score_archive"


def _cw(S):
    w = 2.0 * np.abs(S - 0.5)
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def _rank_mean(S):
    return (np.argsort(np.argsort(S, axis=0), axis=0) / max(S.shape[0] - 1, 1)).mean(1)


def load_archive():
    tasks = {}
    for f in sorted(ARCHIVE.glob("*.npz")):
        if f.name.startswith("._"):
            continue
        z = np.load(f, allow_pickle=True)
        tasks[f.stem] = {"Sval": z["Sval"], "yval": z["yval"], "Stest": z["Stest"],
                         "ytest": z["ytest"], "val_auc": z["val_auc"],
                         "domain": str(z["domain"]), "det": list(z["det_names"])}
    return tasks


def _family(domain: str) -> str:
    if domain.startswith("adb") or domain == "tabular":
        return "tabular"
    if domain == "cyber":
        return "cyber"
    return domain  # fraud, etc.


def _routed_val_auc(t, policy):
    """Quality of the hybrid policy on a task's VALIDATION split (no test labels)."""
    sval, yval = t["Sval"], t["yval"]
    score, _ = route(sval, yval, sval, policy, action="hybrid")
    return roc_auc_score(yval, score) if len(np.unique(yval)) > 1 else 0.5


def fit_policy(train_tasks):
    """Grid-search hybrid thresholds maximizing mean routed validation AUROC."""
    best, best_auc = RouterPolicy(), -1.0
    for conf, gap, guard in itertools.product([0.6, 0.7, 0.8], [0.0, 0.05, 0.1], [0.5, 0.55]):
        p = RouterPolicy(conf=conf, gap=gap, guard=guard)
        a = float(np.mean([_routed_val_auc(t, p) for t in train_tasks]))
        if a > best_auc:
            best, best_auc = p, a
    return best


def main() -> int:
    tasks = load_archive()
    if not tasks:
        print("no archive found; run build_score_archive.py first")
        return 1
    names = list(tasks)
    fams = {_family(tasks[n]["domain"]) for n in names}

    # leave-family-out hybrid: fit policy on other families, apply to held-out family
    hybrid_score = {}
    for fam in fams:
        train = [tasks[n] for n in names if _family(tasks[n]["domain"]) != fam]
        policy = fit_policy(train) if train else RouterPolicy()
        for n in names:
            if _family(tasks[n]["domain"]) == fam:
                s, act = route(tasks[n]["Sval"], tasks[n]["yval"], tasks[n]["Stest"], policy, "hybrid")
                hybrid_score[n] = (s, act, policy)

    det_names = tasks[names[0]]["det"]
    per_auc: dict[str, list[float]] = {d: [] for d in det_names}
    for extra in ["auto_select", "cw_mean", "rank_mean", "elara_fuse", "elara_hybrid"]:
        per_auc[extra] = []
    ece_acc = {m: [] for m in per_auc}
    hybrid_actions = []

    for n in names:
        t = tasks[n]
        Ste, yte, auc = t["Stest"], t["ytest"], t["val_auc"]
        def A(score):
            return roc_auc_score(yte, score) if len(np.unique(yte)) > 1 else 0.5
        for j, d in enumerate(det_names):
            per_auc[d].append(A(Ste[:, j])); ece_acc[d].append(ece(Ste[:, j], yte))
        per_auc["auto_select"].append(A(select(Ste, auc))); ece_acc["auto_select"].append(ece(select(Ste, auc), yte))
        per_auc["cw_mean"].append(A(_cw(Ste))); ece_acc["cw_mean"].append(ece(_cw(Ste), yte))
        per_auc["rank_mean"].append(A(_rank_mean(Ste))); ece_acc["rank_mean"].append(ece(_rank_mean(Ste), yte))
        per_auc["elara_fuse"].append(A(fuse(Ste, auc))); ece_acc["elara_fuse"].append(ece(fuse(Ste, auc), yte))
        hs, act, _ = hybrid_score[n]
        per_auc["elara_hybrid"].append(A(hs)); ece_acc["elara_hybrid"].append(ece(hs, yte))
        hybrid_actions.append(act)

    rr = ranks_and_regret(per_auc)
    best_fixed = min(det_names, key=lambda d: rr["mean_rank"][d])
    vs_auto = bootstrap_delta(rr["_ranks"]["elara_hybrid"], rr["_ranks"]["auto_select"])
    vs_best = bootstrap_delta(rr["_ranks"]["elara_hybrid"], rr["_ranks"][best_fixed])

    result = {
        "protocol": "ELARA_U_UNIVERSAL_CONTRACT_v1",
        "status": "DEVELOPMENT_leave_family_out (tabular/cyber/fraud only; MVS families pending)",
        "n_tasks": len(names), "families": sorted(fams),
        "mean_rank": rr["mean_rank"], "worst_rank": rr["worst_rank"],
        "mean_auc": rr["mean_auc"], "mean_regret": rr["mean_regret"],
        "mean_ece": {m: round(float(np.mean(v)), 4) for m, v in ece_acc.items()},
        "best_fixed": best_fixed,
        "elara_hybrid_vs_auto_select": vs_auto,
        "elara_hybrid_vs_best_fixed": vs_best,
        "negative_transfer_rate_hybrid_vs_rankmean": negative_transfer_rate(per_auc, "elara_hybrid", "rank_mean"),
        "hybrid_action_counts": {a: hybrid_actions.count(a) for a in set(hybrid_actions)},
    }
    out = ROOT / "experiments/elara_u/results.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\n=== ELARA-U Universal Contract (development, {len(names)} tasks, "
          f"leave-family-out over {sorted(fams)}) ===")
    head = ["auto_select", "cw_mean", "rank_mean", "elara_fuse", "elara_hybrid", best_fixed]
    print(f"{'method':14}{'mean_rank':>10}{'worst':>7}{'auc':>8}{'regret':>9}{'ece':>8}")
    for m in head:
        print(f"{m:14}{rr['mean_rank'][m]:10.3f}{rr['worst_rank'][m]:7d}{rr['mean_auc'][m]:8.3f}"
              f"{rr['mean_regret'][m]:9.4f}{result['mean_ece'][m]:8.4f}")
    print(f"\nELARA-U hybrid vs auto-select (meta-baseline): rank delta {vs_auto['delta']:+.3f} "
          f"CI {vs_auto['ci95']} excl0={vs_auto['ci_excludes_zero']}")
    print(f"ELARA-U hybrid vs best fixed ({best_fixed}): rank delta {vs_best['delta']:+.3f} "
          f"CI {vs_best['ci95']} excl0={vs_best['ci_excludes_zero']}")
    print(f"hybrid actions: {result['hybrid_action_counts']}")
    print(f"neg-transfer rate (hybrid<rank_mean): {result['negative_transfer_rate_hybrid_vs_rankmean']:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
