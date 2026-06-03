"""Multimodal independent-modality-failure reliability test (D23).

The last regime where reliability routing could help: 3 real co-observed modality
experts (rgb/ps/xyz) fused; the deployment-best modality fails at test while the
others stay clean. Tests whether test-time drift detection (reliability gating)
beats equal-weight fusion, stale auto-select, and validation-only fusion. No test
labels are used for selection or drift (drift uses the test score distribution).
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
CACHE = ROOT / "experiments/elara_u" / ".."  # placeholder
CACHE = ROOT / "experiments/fusion/realiad_d3_score_cache"
RNG = 0
DRIFT_TH, VAL_FLOOR = 0.25, 0.55


def _cw(S):
    w = 2.0 * np.abs(S - 0.5)
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def _wfuse(S, vauc, keep):
    if not keep.any():
        return S.mean(1)
    w = np.clip(vauc[keep] - 0.5, 1e-6, None)
    return S[:, keep] @ (w / w.sum())


def run_category(Sval, yval, Stest, ytest, vauc, fail):
    """Return dict of method->test AUROC. If fail, corrupt the best-val modality on test."""
    rng = np.random.default_rng(RNG)
    St = Stest.copy()
    best = int(np.argmax(vauc))
    if fail:
        St[:, best] = rng.uniform(0, 1, len(St))      # deployment-best modality fails at test
    A = lambda s: roc_auc_score(ytest, s) if len(np.unique(ytest)) > 1 else 0.5
    # drift per modality: KS(val, test) -- uses test score distribution, no labels
    drift = np.array([ks_2samp(Sval[:, m], St[:, m]).statistic for m in range(St.shape[1])])
    val_ok = np.array([(vauc[m] >= VAL_FLOOR) and (Sval[:, m].std() >= 0.02) for m in range(St.shape[1])])
    keep_rel = val_ok & (drift <= DRIFT_TH)            # reliability gate: val-ok AND not drifted
    return {
        "equal_weight": float(A(_cw(St))),
        "stale_auto_select": float(A(St[:, best])),
        "no_reliability": float(A(_wfuse(St, vauc, val_ok))),     # validation-only fusion
        "reliability_gate": float(A(_wfuse(St, vauc, keep_rel))),  # + test-time drift drop
    }


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    files = [f for f in sorted(glob.glob(str(CACHE / "*_v2_binpcd.npz")))
             if not os.path.basename(f).startswith("._")]
    regimes = {"clean": False, "modality_failure": True}
    out = {r: {m: [] for m in ["equal_weight", "stale_auto_select", "no_reliability", "reliability_gate"]}
           for r in regimes}
    n = 0
    for f in files:
        z = np.load(f)
        Sval, yval, Stest, ytest, vauc = z["Sval"], z["yval"], z["Stest"], z["ytest"], z["valauc"]
        if int((vauc > 0.6).sum()) < 2 or len(np.unique(ytest)) < 2:
            continue
        n += 1
        for r, fail in regimes.items():
            res = run_category(Sval, yval, Stest, ytest, vauc, fail)
            for m in out[r]:
                out[r][m].append(res[m])

    summary = {"protocol": "D23_multimodal_reliability", "n_categories": n, "regimes": {}}
    for r in regimes:
        pa = {m: np.array(v) for m, v in out[r].items()}
        summary["regimes"][r] = {"mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()}}
    # hypotheses in the failure regime
    pf = {m: np.array(out["modality_failure"][m]) for m in out["modality_failure"]}
    H = {}
    for name, base in [("H1_vs_equal_weight", "equal_weight"),
                       ("H2_vs_stale_auto_select", "stale_auto_select"),
                       ("H3_vs_no_reliability", "no_reliability")]:
        mean, lo, hi = _boot(pf["reliability_gate"] - pf[base])
        H[name] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "pass": lo > 0}
    summary["hypotheses_failure_regime"] = H
    summary["reliability_validated"] = all(h["pass"] for h in H.values())
    res_path = ROOT / "experiments/elara_u/multimodal_reliability_results.json"
    res_path.write_text(json.dumps(summary, indent=2))

    print(f"=== D23 MULTIMODAL RELIABILITY ({n} categories, rgb/ps/xyz) ===")
    for r in regimes:
        print(f"  [{r}] " + "  ".join(f"{m}={summary['regimes'][r]['mean_auroc'][m]:.3f}"
                                      for m in ["equal_weight", "stale_auto_select", "no_reliability", "reliability_gate"]))
    print("\n  Hypotheses (modality_failure regime; reliability_gate minus baseline):")
    for k, v in H.items():
        print(f"    {k:26} {v['mean']:+.4f} CI {v['ci95']} pass={v['pass']}")
    print(f"\n  RELIABILITY ROUTING VALIDATED (H1&H2&H3)? {'YES' if summary['reliability_validated'] else 'NO'}")
    print(f"  wrote {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
