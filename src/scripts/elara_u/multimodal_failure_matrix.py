"""B2 multimodal failure-type matrix: does the reliability gate recover under MANY
independent failure modes (not just best-modality dropout), and correctly NOT help
when all modalities fail together (negative control)?

For each dataset cache (Real-IAD-D3 rgb/ps/xyz; MVTec-3D rgb/-/xyz) and each failure
type, we inject the failure on the TEST scores only (validation stays clean), run the
drift-gated fusion, and report H1-H3 (vs equal-weight, stale auto-select, and the
no-test-time-reliability ablation) with paired-bootstrap CIs over categories.

Failure types (score-level, the deployment caches are per-modality scores):
  rgb_noise      modality 0 -> uniform noise           (RGB sensor fails, others clean)
  mid_noise      modality 1 -> uniform noise           (depth/PS fails, others clean)
  xyz_missing    modality 2 -> constant 0.5            (point-cloud missing)
  best_inverted  deployment-best modality -> 1-score   (sign inversion)
  best_saturated deployment-best modality -> constant  (saturation)
  best_noise     deployment-best modality -> noise      (the original D23 dropout)
  all_degraded   ALL modalities -> noise   NEGATIVE CONTROL: no clean channel, the gate
                 must NOT help (reliability cannot manufacture signal).

Pass per (dataset, failure): H1&H2&H3 all CI>0  -> reliability recovers.
Expected: pass on every single-modality failure; FAIL (correctly) on all_degraded.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/elara_u/failure_matrix_results.json"
DRIFT_TH, VAL_FLOOR = 0.25, 0.55
DATASETS = [("experiments/fusion/realiad_d3_score_cache", "*_v2_binpcd.npz", "Real-IAD-D3"),
            ("experiments/fusion/mvtec3d_score_cache", "*.npz", "MVTec-3D")]
FAILURES = ["rgb_noise", "mid_noise", "xyz_missing", "best_inverted",
            "best_saturated", "best_noise", "all_degraded"]


def _cw(S):
    w = 2.0 * np.abs(S - 0.5)
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def _wfuse(S, keep, vauc):
    if not keep.any():
        return S.mean(1)
    w = np.clip(vauc[keep] - 0.5, 1e-6, None)
    return S[:, keep] @ (w / w.sum())


def corrupt(Stest, vauc, ftype, rng):
    St = Stest.copy(); M = St.shape[1]; best = int(np.argmax(vauc))
    if ftype == "rgb_noise":
        St[:, 0] = rng.uniform(0, 1, len(St))
    elif ftype == "mid_noise":
        St[:, min(1, M - 1)] = rng.uniform(0, 1, len(St))
    elif ftype == "xyz_missing":
        St[:, M - 1] = 0.5
    elif ftype == "best_inverted":
        St[:, best] = 1.0 - St[:, best]
    elif ftype == "best_saturated":
        St[:, best] = 0.9
    elif ftype == "best_noise":
        St[:, best] = rng.uniform(0, 1, len(St))
    elif ftype == "all_degraded":
        St = rng.uniform(0, 1, St.shape)
    return St


def run_category(Sval, yval, Stest, ytest, vauc, ftype):
    rng = np.random.default_rng(0)
    St = corrupt(Stest, vauc, ftype, rng)
    A = lambda s: roc_auc_score(ytest, s) if len(np.unique(ytest)) > 1 else 0.5
    drift = np.array([ks_2samp(Sval[:, m], St[:, m]).statistic for m in range(St.shape[1])])
    val_ok = np.array([(vauc[m] >= VAL_FLOOR) and (Sval[:, m].std() >= 0.02) for m in range(St.shape[1])])
    keep_rel = val_ok & (drift <= DRIFT_TH)
    best = int(np.argmax(vauc))
    return {"equal_weight": A(_cw(St)), "stale_auto_select": A(St[:, best]),
            "no_reliability": A(_wfuse(St, val_ok, vauc)), "reliability_gate": A(_wfuse(St, keep_rel, vauc))}


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    out = {}
    for sub, pat, tag in DATASETS:
        files = [f for f in sorted(glob.glob(str(ROOT / sub / pat))) if not os.path.basename(f).startswith("._")]
        cats = []
        for f in files:
            z = np.load(f)
            if int((z["valauc"] > 0.6).sum()) < 2 or len(np.unique(z["ytest"])) < 2:
                continue
            cats.append((z["Sval"], z["yval"], z["Stest"], z["ytest"], z["valauc"]))
        ds = {}
        for ftype in FAILURES:
            acc = {m: [] for m in ["equal_weight", "stale_auto_select", "no_reliability", "reliability_gate"]}
            for Sval, yval, Stest, ytest, vauc in cats:
                r = run_category(Sval, yval, Stest, ytest, vauc, ftype)
                for m in acc:
                    acc[m].append(r[m])
            pa = {m: np.array(v) for m, v in acc.items()}
            H = {}
            for name, base in [("H1_vs_equal_weight", "equal_weight"),
                               ("H2_vs_stale_auto_select", "stale_auto_select"),
                               ("H3_vs_no_reliability", "no_reliability")]:
                mean, lo, hi = _boot(pa["reliability_gate"] - pa[base])
                H[name] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "pass": lo > 0}
            ds[ftype] = {"mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
                         "hypotheses": H, "all_pass": all(h["pass"] for h in H.values())}
        out[tag] = {"n_categories": len(cats), "failures": ds}

    out["summary"] = {
        "independent_failures_all_pass": {tag: all(out[tag]["failures"][ft]["all_pass"]
                                                   for ft in FAILURES if ft != "all_degraded")
                                          for tag, _, _ in [(t, 0, 0) for _, _, t in DATASETS]},
        "all_degraded_control_does_not_help": {tag: not out[tag]["failures"]["all_degraded"]["all_pass"]
                                               for _, _, tag in DATASETS},
    }
    OUT.write_text(json.dumps(out, indent=2))
    for _, _, tag in DATASETS:
        print(f"\n=== {tag} ({out[tag]['n_categories']} cats) ===")
        for ft in FAILURES:
            d = out[tag]["failures"][ft]; H = d["hypotheses"]
            mark = "PASS" if d["all_pass"] else "no  "
            print(f"  {ft:15} gate={d['mean_auroc']['reliability_gate']:.3f} "
                  f"H1={H['H1_vs_equal_weight']['mean']:+.3f} H2={H['H2_vs_stale_auto_select']['mean']:+.3f} "
                  f"H3={H['H3_vs_no_reliability']['mean']:+.3f}  [{mark}]")
    print(f"\nsummary: {json.dumps(out['summary'])}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
