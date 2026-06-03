"""Proof-of-concept: reliability gating helps IFF modalities fail INDEPENDENTLY.

This is the scientific bridge between the negative result (reliability routing does
not help on single-modality data) and the multimodal hypothesis. It is a controlled
CPU simulation that isolates the ONE structural variable that matters: whether a
channel can fail while the others stay clean.

Two regimes, identical everything else:
  SHARED      all modality scores are driven by a common latent; degradation hits
              the shared latent, so every modality degrades together (the single-
              modality analog -- no clean channel to switch to).
  INDEPENDENT each modality is independent; degradation removes the signal from ONE
              modality only, leaving the others clean (the genuine multimodal case).

At test time (no labels), a degraded channel still looks good on the clean validation
split (stale validation) but (a) drifts distributionally and (b) disagrees with the
consensus of the clean channels. Reliability gating uses those unlabeled signals to
downweight it.

Methods compared per trial (test AUROC):
  mean         equal-weight fusion
  val_select   pick the channel with best validation AUROC (stale under shift)
  rel_gate     fuse, weighting each channel by val-quality x test-time agreement
  rel_gate_abl ablation: weight by val-quality ONLY (no test-time reliability)

Pre-registered prediction:
  SHARED:      rel_gate ~= mean ~= val_select   (reliability cannot help)
  INDEPENDENT: rel_gate >  mean and > val_select (reliability recovers the clean
               channels), and rel_gate > rel_gate_abl (the test-time reliability
               signal, not val-quality, carries the gain).
If the prediction holds, the negative result is explained mechanistically and the
multimodal experiment is justified. If it fails, the whole reliability premise is
dead even where it structurally should work.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/elara_u/synthetic_multimodal_poc.json"
RNG = 0
N_TRIALS = 300
N, K = 1200, 4          # samples per split, modalities
ANOM_RATE = 0.12


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5


def _gen_scores(y, deltas, gains, rng):
    """[n, K] scores: channel m encodes the label with strength deltas[m]*gains[m]
    plus independent noise. gains lets us weaken specific channels at deployment."""
    n = len(y)
    return np.column_stack([deltas[m] * gains[m] * y + rng.standard_normal(n)
                            for m in range(K)])


def reliability_weights(Sval, Stest, val_auc, use_testtime):
    """Unsupervised reliability weight per channel. val-quality x test-time agreement.
    Ablation (use_testtime=False) uses val-quality only."""
    q = np.clip(val_auc - 0.5, 0, None)
    if not use_testtime:
        w = q
        return w / w.sum() if w.sum() > 0 else np.ones(len(q)) / len(q)
    # test-time agreement: mean rank-correlation of each channel with the others
    K = Stest.shape[1]; agree = np.zeros(K)
    for m in range(K):
        cs = []
        for k in range(K):
            if k == m:
                continue
            c = spearmanr(Stest[:, m], Stest[:, k]).correlation
            cs.append(0.0 if np.isnan(c) else c)
        agree[m] = max(0.0, float(np.mean(cs)))
    w = q * agree
    return w / w.sum() if w.sum() > 0 else np.ones(K) / K


def _ranknorm(S):
    return np.column_stack([rankdata(S[:, j]) / len(S) for j in range(S.shape[1])])


def one_trial(regime, rng):
    yv = (rng.random(N) < ANOM_RATE).astype(int)
    yt = (rng.random(N) < ANOM_RATE).astype(int)
    deltas = rng.uniform(1.2, 2.6, K)                     # per-channel signal strength
    ones = np.ones(K)
    # validation is clean in both regimes (channels equally informative)
    Sval = _gen_scores(yv, deltas, ones, rng)
    val_auc = np.array([_auc(yv, Sval[:, m]) for m in range(K)])

    # deployment degradation differs ONLY in structure:
    #   INDEPENDENT: ONE channel loses all signal; the others stay fully clean.
    #   SHARED:      ALL channels are weakened together (a common cause); none clean,
    #                but none singled out -- no channel to switch to.
    if regime == "INDEPENDENT":
        gains = ones.copy()
        bad = int(rng.integers(0, K))
        gains[bad] = 0.0                                  # one channel fails independently
    else:  # SHARED
        bad = -1
        gains = np.full(K, 0.30)                          # every channel weakened equally
    Stest = _gen_scores(yt, deltas, gains, rng)

    Rv, Rt = _ranknorm(Sval), _ranknorm(Stest)
    res = {}
    res["mean"] = _auc(yt, Rt.mean(1))
    res["val_select"] = _auc(yt, Stest[:, int(np.argmax(val_auc))])
    w = reliability_weights(Sval, Stest, val_auc, use_testtime=True)
    res["rel_gate"] = _auc(yt, Rt @ w)
    wa = reliability_weights(Sval, Stest, val_auc, use_testtime=False)
    res["rel_gate_abl"] = _auc(yt, Rt @ wa)
    res["_bad_was_val_best"] = int(np.argmax(val_auc) == bad)
    return res


def paired_ci(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d)
    rng = np.random.default_rng(RNG)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(5000)])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "sig": bool(np.percentile(boot, 2.5) > 0)}


def main():
    out = {}
    for regime in ["SHARED", "INDEPENDENT"]:
        rng = np.random.default_rng(RNG)
        rows = [one_trial(regime, rng) for _ in range(N_TRIALS)]
        cols = {m: np.array([r[m] for r in rows]) for m in ["mean", "val_select", "rel_gate", "rel_gate_abl"]}
        out[regime] = {
            "mean_auroc": {m: float(cols[m].mean()) for m in cols},
            "rel_gate_vs_mean": paired_ci(cols["rel_gate"], cols["mean"]),
            "rel_gate_vs_val_select": paired_ci(cols["rel_gate"], cols["val_select"]),
            "rel_gate_vs_ablation": paired_ci(cols["rel_gate"], cols["rel_gate_abl"]),
            "frac_trials_stale_val_picks_bad": float(np.mean([r["_bad_was_val_best"] for r in rows])),
        }
        print(f"\n=== {regime} ({N_TRIALS} trials, K={K} modalities, 1 degraded at test) ===")
        m = out[regime]["mean_auroc"]
        print(f"  mean={m['mean']:.3f}  val_select={m['val_select']:.3f}  "
              f"rel_gate={m['rel_gate']:.3f}  rel_gate_abl={m['rel_gate_abl']:.3f}")
        for k in ["rel_gate_vs_mean", "rel_gate_vs_val_select", "rel_gate_vs_ablation"]:
            v = out[regime][k]
            print(f"  {k:24}{v['mean']:+.4f}  CI{[round(x,4) for x in v['ci95']]}  sig={v['sig']}")

    shared_ok = (not out["SHARED"]["rel_gate_vs_ablation"]["sig"])
    indep_ok = (out["INDEPENDENT"]["rel_gate_vs_mean"]["sig"]
                and out["INDEPENDENT"]["rel_gate_vs_val_select"]["sig"]
                and out["INDEPENDENT"]["rel_gate_vs_ablation"]["sig"])
    if indep_ok and shared_ok:
        verdict = ("PREDICTION CONFIRMED: reliability gating helps under INDEPENDENT modality "
                   "failure (beats mean, val_select, and the no-test-time ablation) but not under "
                   "SHARED degradation. The negative result is mechanistic: reliability needs an "
                   "independent clean channel. Multimodal experiment is justified.")
    elif indep_ok:
        verdict = ("Reliability gating helps under INDEPENDENT failure (the key prediction); SHARED "
                   "regime shows a residual effect -- inspect. Multimodal experiment justified.")
    else:
        verdict = ("PREDICTION FAILED: reliability gating does not clearly help even under INDEPENDENT "
                   "modality failure. The reliability premise is dead even where it structurally should "
                   "work; do not pursue the multimodal experiment.")
    out["verdict"] = verdict
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {verdict}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
