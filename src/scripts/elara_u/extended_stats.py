"""Extended statistics (A4/A5 checklist): AUPRC, family-balanced mean rank/AUROC,
and an abstention curve, all from the verified 123-task archive (leakage-free).

- AUPRC: per-task average precision for each strategy (complements AUROC).
- Family-balanced mean: mean over families of the per-family mean (equal weight per
  family, so the 61-task image-OOD family does not dominate the 1-task fraud family).
- Abstention curve: rank tasks by router confidence (the stack's margin = top minus
  median predicted-anomaly score); abstain on the least-confident fraction and report
  mean AUROC / regret on the tasks the router DOES act on. A good router's accuracy
  rises as it abstains more.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.elara_u.honest_benchmark import load_archive, strategies_for_task

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/elara_u/extended_stats.json"
FIG = ROOT / "docs/research/figures/elara_u_abstention.png"


def main():
    tasks = load_archive()
    fams = sorted({t["fam"] for t in tasks})
    strby = {t["name"]: strategies_for_task(t) for t in tasks}
    ytest = {t["name"]: t["ytest"] for t in tasks}
    names = [t["name"] for t in tasks]
    fam_of = {t["name"]: t["fam"] for t in tasks}
    METH = ["stack", "auto_select", "rank_mean", "cw_mean"]

    # AUPRC per strategy (mean over tasks) + family-balanced AUROC
    auprc, auroc = {}, {}
    for m in METH:
        ap, au = [], []
        for n in names:
            sc = strby[n][m][1]
            ap.append(float(average_precision_score(ytest[n], sc)))
            au.append(strby[n][m][0])
        auprc[m] = float(np.mean(ap)); auroc[m] = float(np.mean(au))
    # family-balanced (equal weight per family)
    fam_bal = {}
    for m in METH:
        per_fam = []
        for f in fams:
            idx = [n for n in names if fam_of[n] == f]
            per_fam.append(np.mean([strby[n][m][0] for n in idx]))
        fam_bal[m] = float(np.mean(per_fam))

    # abstention curve for the stack: confidence = margin of predicted-anomaly scores
    conf = {}
    for n in names:
        p = np.clip(strby[n]["stack"][1], 0, 1)
        conf[n] = float(np.quantile(p, 0.99) - np.median(p))   # how peaked the top scores are
    order = sorted(names, key=lambda n: conf[n])                # least-confident first
    curve = []
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        k = int(frac * len(order))
        kept = order[k:]
        a = np.mean([strby[n]["stack"][0] for n in kept])
        oracle = np.mean([max(strby[n][f"fixed/{d}"][0] for d in tasks[0]["dets"]) for n in kept])
        curve.append({"abstain_frac": frac, "n_kept": len(kept),
                      "stack_auroc": float(a), "regret_to_oracle": float(oracle - a)})

    result = {"protocol": "ELARA_U_EXTENDED_STATS_v1", "n_tasks": len(tasks), "families": fams,
              "mean_auprc": auprc, "mean_auroc": auroc, "family_balanced_auroc": fam_bal,
              "abstention_curve_stack": curve}
    OUT.write_text(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = [c["abstain_frac"] for c in curve]
    plt.figure(figsize=(5, 3))
    plt.plot(fr, [c["stack_auroc"] for c in curve], "o-", color="#d95f02", label="stack AUROC")
    plt.plot(fr, [c["regret_to_oracle"] for c in curve], "s--", color="#2c7fb8", label="regret to oracle")
    plt.xlabel("abstention fraction (least-confident tasks dropped)"); plt.ylabel("mean over kept tasks")
    plt.title("Abstention curve (ELARA-U Stack)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG, dpi=150); plt.close()

    print("AUPRC :", {m: round(auprc[m], 3) for m in METH})
    print("AUROC :", {m: round(auroc[m], 3) for m in METH})
    print("fam-balanced AUROC:", {m: round(fam_bal[m], 3) for m in METH})
    print("abstention (frac->stack AUROC):", {c["abstain_frac"]: round(c["stack_auroc"], 3) for c in curve})
    print(f"wrote {OUT} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
