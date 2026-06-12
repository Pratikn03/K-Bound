"""Per-Sample Reliability-Gated Stacking (PS-RGS), a NEW method (D30).

Theory motivation: the paper's negative result (reliability routing inert on
single-input) is Proposition 1 -- a per-CHANNEL constant gate r lies inside the linear
class H={sum_m beta_m phi_m} the stacker already optimizes. The escape: make the weight
depend on the SAMPLE x. A combiner g(x)=sum_m w_m(x) phi_m(x) with w_m(x) non-constant
is OUTSIDE H, so Prop 1's bound does not cap it. PS-RGS learns such a combiner from
NO-LABEL per-sample reliability signals:
  rho_m(x)   local validation reliability of detector m near x (kNN in score space)
  dis_m(x)   per-sample disagreement of m vs consensus rank
  sharp_m(x) |phi_m(x)-0.5| decisiveness
The stacker is fit on AUGMENTED features {phi, rho, dis, sharp, phi*rho} -> the phi*rho
cross-term makes the effective weight per-sample (outside H). We compare PS-RGS to plain
logistic stacking (phi only) on the 123-task archive; honest test of whether per-sample
reliability is label-relevant & heterogeneous (T10). No test labels used to fit anything.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
ARCH = ROOT / "experiments/elara_u/score_archive"
K = 25   # neighborhood size for local reliability


def _ranknorm(S):
    return np.argsort(np.argsort(S, 0), 0) / max(len(S) - 1, 1)


def _per_sample_features(Sval, yval, Sq):
    """Build per-sample reliability features for query rows Sq, using validation (Sval,yval).

    rho_m(x): 1 - local error of detector m = local AUROC-proxy in x's val-neighborhood.
    Implemented label-legitimately: neighbors found in val score-space; rho_m = fraction
    of neighbor pairs that m orders correctly is expensive, so use a cheap proxy:
    rho_m = | mean(phi_m | y=1 nbrs) - mean(phi_m | y=0 nbrs) | (local class separation).
    """
    M = Sval.shape[1]
    Rval = _ranknorm(Sval)
    Rq_val = Rq_proxy(Sval, Sq)                            # query mapped into val rank-space (nbr lookup)
    nn = NearestNeighbors(n_neighbors=min(K, len(Sval))).fit(Rval)
    _, idx = nn.kneighbors(Rq_val)
    yv = np.asarray(yval)
    rho = np.zeros((len(Sq), M))
    for i in range(len(Sq)):
        nb = idx[i]; ynb = yv[nb]
        if ynb.any() and (~ynb.astype(bool)).any():       # local class separation of detector m near x
            for m in range(M):
                sm = Sval[nb, m]
                rho[i, m] = abs(sm[ynb == 1].mean() - sm[ynb == 0].mean())
    Rq_int = _ranknorm(Sq)                                 # within-query ranks for disagreement
    dis = np.abs(Rq_int - Rq_int.mean(1, keepdims=True))   # per-sample disagreement vs consensus
    sharp = np.abs(Sq - 0.5)                                # decisiveness
    return rho, dis, sharp


def Rq_proxy(Sval, Sq):
    """Map query rows into validation rank-space for neighbor lookup (no labels)."""
    Rq = np.zeros_like(Sq)
    for m in range(Sq.shape[1]):
        order = np.argsort(Sval[:, m])
        vs = Sval[order, m]
        Rq[:, m] = np.searchsorted(vs, Sq[:, m]) / max(len(vs) - 1, 1)
    return Rq


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    fs = [f for f in sorted(glob.glob(str(ARCH / "*.npz"))) if not os.path.basename(f).startswith("._")]
    per = {"stack": [], "ps_rgs": [], "het_div": [], "het_aucstd": []}
    n = 0
    for f in fs:
        z = np.load(f, allow_pickle=True)
        Sval, yval, Stest, ytest = z["Sval"], z["yval"].astype(int), z["Stest"], z["ytest"].astype(int)
        if len(np.unique(ytest)) < 2 or len(np.unique(yval)) < 2:
            continue
        A = lambda s: roc_auc_score(ytest, s)
        # baseline: plain logistic stack on phi
        base = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sval, yval)
        per["stack"].append(A(base.predict_proba(Stest)[:, 1]))
        # PS-RGS: augmented per-sample reliability features + phi*rho cross terms
        rho_v, dis_v, sharp_v = _per_sample_features(Sval, yval, Sval)
        rho_t, dis_t, sharp_t = _per_sample_features(Sval, yval, Stest)
        Xv = np.hstack([Sval, rho_v, dis_v, sharp_v, Sval * rho_v])
        Xt = np.hstack([Stest, rho_t, dis_t, sharp_t, Stest * rho_t])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xv, yval)
        per["ps_rgs"].append(A(clf.predict_proba(Xt)[:, 1]))
        # a-priori heterogeneity measures (T10 predicts gain grows with these)
        C = np.corrcoef(Stest.T); off = np.abs(C[np.triu_indices(Stest.shape[1], 1)])
        per["het_div"].append(float(1.0 - np.nan_to_num(off).mean()))     # detector score diversity
        va = np.array([roc_auc_score(yval, Sval[:, j]) for j in range(Sval.shape[1])])
        per["het_aucstd"].append(float(va.std()))                         # spread of detector quality
        n += 1
    pa = {k: np.array(v) for k, v in per.items()}
    delta = pa["ps_rgs"] - pa["stack"]
    # T10 test: does the per-task PS-RGS gain correlate with detector heterogeneity?
    t10 = {}
    for hk in ["het_div", "het_aucstd"]:
        rho_s, p_s = spearmanr(delta, pa[hk])
        t10[hk] = {"spearman_r": round(float(rho_s), 4), "p_value": round(float(p_s), 5),
                   "confirms_T10": bool(rho_s > 0 and p_s < 0.05)}
    mean, lo, hi = _boot(delta)
    res = {
        "protocol": "D30_PER_SAMPLE_RELIABILITY_GATED_STACKING", "n_tasks": n,
        "mean_auroc": {k: round(float(pa[k].mean()), 4) for k in ("stack", "ps_rgs")},
        "ps_rgs_minus_stack": {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "beats_stack": lo > 0},
        "wins": int((pa["ps_rgs"] > pa["stack"]).sum()),
        "T10_heterogeneity_correlation": t10,
    }
    out = ROOT / "experiments/elara_u/per_sample_routing_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"=== D30 PS-RGS vs plain stacking ({n} tasks) ===")
    print(f"  stack   mean AUROC {res['mean_auroc']['stack']:.4f}")
    print(f"  PS-RGS  mean AUROC {res['mean_auroc']['ps_rgs']:.4f}")
    print(f"  PS-RGS - stack: {mean:+.4f} CI [{lo:+.4f},{hi:+.4f}] beats_stack={res['ps_rgs_minus_stack']['beats_stack']}")
    print(f"  PS-RGS wins on {res['wins']}/{n} tasks")
    print("  T10 (gain vs heterogeneity):")
    for hk, v in t10.items():
        print(f"    {hk:12} Spearman r={v['spearman_r']:+.3f} p={v['p_value']} confirms_T10={v['confirms_T10']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
