"""Industrial-vision family benchmark (MVTec AD + VisA, ResNet-50 embeddings).

Reported as a SEPARATE family (not merged into the 123-task headline). Each category's
ResNet embeddings (data/raw/adbench_industrial/{mvtecad,visa}_*.npz) are scored with the
frozen detector zoo (build_task), then stack vs auto-select vs best-fixed with paired
bootstrap over categories. No test labels are used to fit any router.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

from scripts.elara_u.build_score_archive import build_task
from scripts.elara_u.gate_u_seed_eval import _balance
from scripts.elara_u.honest_benchmark import paired_ci, strategies_for_task

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data/raw/adbench_industrial"
OUT = ROOT / "experiments/elara_u/industrial_benchmark.json"


def main():
    tasks, src_of = [], {}
    for f in sorted(glob.glob(str(SRC / "*.npz"))):
        b = os.path.basename(f)
        if b.startswith("._"):
            continue
        z = np.load(f)
        X = np.nan_to_num(z["X"].astype(float)); y = z["y"].astype(int)
        if X.shape[0] < 80 or int(y.sum()) < 12 or len(np.unique(y)) < 2:
            continue
        X, y = _balance(X, y)
        Sval, yval, Stest, ytest, det_names, vauc = build_task(X, y)
        name = "ind_" + b[:-4]
        tasks.append({"name": name, "Sval": Sval, "yval": yval, "Stest": Stest, "ytest": ytest,
                      "dets": [str(d) for d in det_names], "val_auc": vauc})
        src_of[name] = "mvtec_ad" if b.startswith("mvtecad_") else "visa"
    per = {t["name"]: strategies_for_task(t) for t in tasks}
    names = list(per); dets = tasks[0]["dets"]
    col = lambda s: np.array([per[n][s][0] for n in names])
    bestfix = max((f"fixed/{d}" for d in dets), key=lambda s: col(s).mean())

    def sub(src):
        ns = [n for n in names if src_of[n] == src]
        if not ns:
            return None
        c = lambda s: np.array([per[n][s][0] for n in ns])
        return {"n": len(ns), "stack": float(c("stack").mean()),
                "auto_select": float(c("auto_select").mean()),
                "stack_vs_auto": paired_ci(c("stack"), c("auto_select"))}

    res = {"protocol": "ELARA_U_INDUSTRIAL_v2 (MVTec AD + VisA, separate family)",
           "n_tasks": len(names), "sources": {s: sum(v == s for v in src_of.values()) for s in set(src_of.values())},
           "mean_auroc": {"stack": float(col("stack").mean()), "auto_select": float(col("auto_select").mean()),
                          "best_fixed": float(col(bestfix).mean()),
                          "oracle": float(np.mean([per[n]["oracle"][0] for n in names]))},
           "best_fixed": bestfix,
           "stack_vs_auto_select": paired_ci(col("stack"), col("auto_select")),
           "stack_vs_best_fixed": paired_ci(col("stack"), col(bestfix)),
           "per_source": {"mvtec_ad": sub("mvtec_ad"), "visa": sub("visa")}}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"=== Industrial family ({len(names)} tasks: {res['sources']}) ===")
    m = res["mean_auroc"]
    print(f"  stack={m['stack']:.3f} auto={m['auto_select']:.3f} best_fixed={m['best_fixed']:.3f} oracle={m['oracle']:.3f}")
    print(f"  stack vs auto:  {res['stack_vs_auto_select']['mean']:+.4f} {res['stack_vs_auto_select']['ci95']} sig={res['stack_vs_auto_select']['sig']}")
    print(f"  stack vs fixed: {res['stack_vs_best_fixed']['mean']:+.4f} {res['stack_vs_best_fixed']['ci95']} sig={res['stack_vs_best_fixed']['sig']}")
    for s in ("mvtec_ad", "visa"):
        d = res["per_source"][s]
        if d:
            print(f"  [{s}] n={d['n']} stack={d['stack']:.3f} auto={d['auto_select']:.3f} "
                  f"delta={d['stack_vs_auto']['mean']:+.4f} sig={d['stack_vs_auto']['sig']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
