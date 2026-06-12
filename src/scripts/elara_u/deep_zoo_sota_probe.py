"""SOTA-attempt EARLY SIGNAL (D31, development exploration -- NOT the sealed one-shot).

Question: does a HETEROGENEOUS deep+classical zoo (a) raise the absolute level and
(b) make per-sample routing (PS-RGS) significantly beat plain stacking (T10 predicts
yes, because deep + classical detectors disagree about WHERE they are reliable)? Run on
the 47 ADBench classical tabular datasets, re-scored from raw features. Deep detectors:
DeepSVDD, AutoEncoder, VAE (genuinely different mechanisms from the classical zoo).
No test labels used to fit any detector or combiner. Development data (opened): this is
a feasibility probe to decide whether to commit to the full pre-registered SOTA run.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.elara_u.gate_u_seed_eval import detector_zoo, _balance
from scripts.elara_u.per_sample_routing import _per_sample_features

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/adbench_classical"
RNG, CAP = 0, 6000


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _deep_ctors(d):
    """Deep detectors (different mechanisms). Built lazily; robust to version differences."""
    from pyod.models.deep_svdd import DeepSVDD
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    return {
        "DeepSVDD": lambda: DeepSVDD(n_features=d, epochs=20, verbose=0),
        "AutoEncoder": lambda: AutoEncoder(epoch_num=20, verbose=0),
        "VAE": lambda: VAE(epoch_num=20, verbose=0),
    }


def score_task(X, y):
    X, y = _balance(X, y)                      # cap rows preserving anomaly rate (handles pos>CAP)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=RNG, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=RNG, stratify=ytmp)
    sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    names, sval, stest = [], [], []
    ctors = dict(detector_zoo())
    ctors.update(_deep_ctors(Xtr.shape[1]))
    for nm, ctor in ctors.items():
        try:
            m = ctor(); m.fit(Xtr); ref = m.decision_function(Xtr)   # no chaining (deep fit returns None)
            sval.append(_zsig(m.decision_function(Xva), ref)); stest.append(_zsig(m.decision_function(Xte), ref))
        except Exception:
            sval.append(np.full(len(yva), 0.5)); stest.append(np.full(len(yte), 0.5))
        names.append(nm)
    return np.column_stack(sval), yva.astype(int), np.column_stack(stest), yte.astype(int), names


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _stack(Sv, yv, St):
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sv, yv).predict_proba(St)[:, 1]


def _psrgs(Sv, yv, St):
    rv, dv, sv = _per_sample_features(Sv, yv, Sv)
    rt, dt, st = _per_sample_features(Sv, yv, St)
    Xv = np.hstack([Sv, rv, dv, sv, Sv * rv]); Xt = np.hstack([St, rt, dt, st, St * rt])
    return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xv, yv).predict_proba(Xt)[:, 1]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW), help="dir of ADBench *.npz feature files")
    ap.add_argument("--tag", default="ADBench-classical")
    ap.add_argument("--out", default=str(ROOT / "experiments/elara_u/deep_zoo_sota_probe_results.json"))
    args = ap.parse_args()
    fs = [f for f in sorted(glob.glob(os.path.join(args.raw, "*.npz"))) if not os.path.basename(f).startswith("._")]
    per = {k: [] for k in ["best_single", "auto_select", "stack_classical", "stack_all", "psrgs_all"]}
    nclass = len(detector_zoo()); n = 0
    for f in fs:
        z = np.load(f); X, y = np.nan_to_num(z["X"].astype(float)), z["y"].astype(int)
        if X.shape[0] < 80 or int(y.sum()) < 12 or len(np.unique(y)) < 2:
            continue
        try:
            Sval, yval, Stest, ytest, names = score_task(X, y)
        except Exception as e:
            print(f"[{os.path.basename(f)}] FAIL {type(e).__name__}: {e}", flush=True); continue
        if len(np.unique(ytest)) < 2 or len(np.unique(yval)) < 2:
            continue
        A = lambda s: roc_auc_score(ytest, s)
        va = np.array([roc_auc_score(yval, Sval[:, j]) if len(np.unique(yval)) > 1 else .5 for j in range(Sval.shape[1])])
        per["best_single"].append(max(A(Stest[:, j]) for j in range(Stest.shape[1])))
        per["auto_select"].append(A(Stest[:, int(np.argmax(va))]))
        per["stack_classical"].append(A(_stack(Sval[:, :nclass], yval, Stest[:, :nclass])))
        per["stack_all"].append(A(_stack(Sval, yval, Stest)))
        per["psrgs_all"].append(A(_psrgs(Sval, yval, Stest)))
        n += 1
        print(f"[{os.path.basename(f)[:-4]:22}] stack_cls={per['stack_classical'][-1]:.3f} "
              f"stack_all={per['stack_all'][-1]:.3f} psrgs={per['psrgs_all'][-1]:.3f}", flush=True)
    pa = {k: np.array(v) for k, v in per.items()}
    deep_help = _boot(pa["stack_all"] - pa["stack_classical"])
    psrgs_gain = _boot(pa["psrgs_all"] - pa["stack_all"])
    res = {"protocol": "D31_DEEP_ZOO_SOTA_PROBE", "family": args.tag, "n_tasks": n,
           "zoo": f"{nclass} classical + DeepSVDD/AutoEncoder/VAE",
           "mean_auroc": {k: round(float(v.mean()), 4) for k, v in pa.items()},
           "deep_zoo_helps_stacking": {"mean": round(deep_help[0], 4), "ci95": [round(deep_help[1], 4), round(deep_help[2], 4)], "sig": deep_help[1] > 0},
           "psrgs_beats_stack_on_deep_zoo": {"mean": round(psrgs_gain[0], 4), "ci95": [round(psrgs_gain[1], 4), round(psrgs_gain[2], 4)], "sig": psrgs_gain[1] > 0}}
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== D31 DEEP-ZOO PROBE [{args.tag}] ({n} tasks; {res['zoo']}) ===")
    for k in ["best_single", "auto_select", "stack_classical", "stack_all", "psrgs_all"]:
        print(f"  {k:16} {res['mean_auroc'][k]:.4f}")
    print(f"deep zoo helps stacking : {deep_help[0]:+.4f} CI[{deep_help[1]:+.4f},{deep_help[2]:+.4f}] sig={res['deep_zoo_helps_stacking']['sig']}")
    print(f"PS-RGS vs stack (deep)  : {psrgs_gain[0]:+.4f} CI[{psrgs_gain[1]:+.4f},{psrgs_gain[2]:+.4f}] sig={res['psrgs_beats_stack_on_deep_zoo']['sig']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
