"""Calibration evaluation (verified): Brier / NLL / ECE, raw and isotonic-calibrated,
plus a pooled reliability diagram. Fills the previously-pending calibration cells.

Per task: probabilistic test scores for ELARA-U Stack, auto-select, and rank-mean are
computed leakage-free (stack = rank-normalized logistic regression fit on validation).
An isotonic calibrator is fit on the VALIDATION scores+labels of each task and applied
to that task's TEST scores (no test labels used to fit). Metrics are averaged over the
123 tasks; the reliability diagram pools calibrated stack predictions across tasks.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from scripts.elara_u.honest_benchmark import load_archive

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/elara_u/calibration_results.json"
FIG = ROOT / "docs/research/figures/elara_u_reliability.png"
EPS = 1e-6


def _rn(S):
    return np.column_stack([rankdata(S[:, j]) / len(S) for j in range(S.shape[1])])


def _brier(p, y):
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


def _nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(p, y, bins=10):
    p = np.clip(p, 0, 1); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p <= edges[b + 1] if b == bins - 1 else p < edges[b + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def task_probs(t):
    """Return {method: (val_prob, test_prob)} -- leakage-free probabilistic scores."""
    Sval, yval, Stest = t["Sval"], t["yval"], t["Stest"]
    out = {}
    js = int(np.argmax(t["val_auc"]))
    out["auto_select"] = (Sval[:, js], Stest[:, js])
    out["rank_mean"] = (_rn(Sval).mean(1), _rn(Stest).mean(1))
    Xv, Xt = _rn(Sval), _rn(Stest)
    if len(np.unique(yval)) == 2:
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xv, yval)
        out["stack"] = (clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xt)[:, 1])
    else:
        out["stack"] = out["rank_mean"]
    return out


def main():
    tasks = load_archive()
    methods = ["stack", "auto_select", "rank_mean"]
    acc = {m: {k: [] for k in ["brier_raw", "nll_raw", "ece_raw",
                               "brier_cal", "nll_cal", "ece_cal"]} for m in methods}
    pooled_p, pooled_y = [], []   # calibrated stack, for the reliability diagram
    for t in tasks:
        probs = task_probs(t)
        yval, ytest = t["yval"], t["ytest"]
        for m in methods:
            pv, pt = probs[m]
            acc[m]["brier_raw"].append(_brier(pt, ytest))
            acc[m]["nll_raw"].append(_nll(pt, ytest))
            acc[m]["ece_raw"].append(_ece(pt, ytest))
            # isotonic calibrator fit on validation only, applied to test
            try:
                ir = IsotonicRegression(out_of_bounds="clip").fit(pv, yval)
                ptc = ir.predict(pt)
            except Exception:
                ptc = pt
            acc[m]["brier_cal"].append(_brier(ptc, ytest))
            acc[m]["nll_cal"].append(_nll(ptc, ytest))
            acc[m]["ece_cal"].append(_ece(ptc, ytest))
            if m == "stack":
                pooled_p.append(np.clip(ptc, 0, 1)); pooled_y.append(ytest)

    result = {"protocol": "ELARA_U_CALIBRATION_v1", "n_tasks": len(tasks),
              "note": "isotonic calibrator fit on validation only; no test labels used to fit.",
              "metrics": {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}}
    OUT.write_text(json.dumps(result, indent=2))

    # reliability diagram (pooled calibrated stack)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.concatenate(pooled_p); y = np.concatenate(pooled_y)
    edges = np.linspace(0, 1, 11); xs, ys = [], []
    for b in range(10):
        m = (p >= edges[b]) & (p <= edges[b + 1] if b == 9 else p < edges[b + 1])
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    plt.figure(figsize=(4.2, 4.2))
    plt.plot([0, 1], [0, 1], "--", color="#888888", label="perfect")
    plt.plot(xs, ys, "o-", color="#d95f02", label="ELARA-U Stack (calibrated)")
    plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
    plt.title("Reliability diagram (pooled, 123 tasks)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()

    print(f"wrote {OUT} and {FIG}")
    for m in methods:
        d = result["metrics"][m]
        print(f"  {m:12} Brier {d['brier_raw']:.3f}->{d['brier_cal']:.3f}  "
              f"NLL {d['nll_raw']:.3f}->{d['nll_cal']:.3f}  ECE {d['ece_raw']:.3f}->{d['ece_cal']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
