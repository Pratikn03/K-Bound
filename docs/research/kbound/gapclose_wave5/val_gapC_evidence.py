"""Validator for Gap C (frozen criteria in PROTOCOL_GAPCLOSE_WAVE5_v1.md).

Synthetic drift family with CONFIDENCE MISCALIBRATION: severity raises both
class-separability loss and a temperature drift, so entropy/MSP confound
"uncertain" with "harmful" — the documented failure mode (gamma != 0).
Harm ground truth: adaptation (entropy-min style sharpening) helps when the
decision boundary is still mostly right, hurts when features have rotated
(sharpening wrong predictions). The validator scores label-free harm detection
AUC using (a) baselines {entropy, msp}, (b) baselines + new features.

C1  AUC(baselines + new) - AUC(baselines) >= +0.05 (pooled over the family).
C2  Stability: uplift >= -0.02 in every severity band (no band where the new
    features hurt).
Exit 0 iff both pass. Seeds fixed.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from evidence_v2 import extract_all  # noqa: E402

RNG = np.random.default_rng(20260702)
N_BATCH, K, DIM, N_COND = 256, 10, 32, 900


def make_condition(rng):
    """One deployment condition -> (logits, harmful?, severity band)."""
    sev = rng.random()  # 0..1
    mu = rng.normal(size=(K, DIM)) * 1.6
    W = mu.T / np.linalg.norm(mu, axis=1)  # source-trained classifier (aligned)
    y = rng.integers(0, K, size=N_BATCH)
    # feature rotation grows with severity (true damage to separability)
    rot = sev * rng.normal(size=(DIM, DIM)) * 0.45
    X = mu[y] + rng.normal(size=(N_BATCH, DIM))
    X = X @ (np.eye(DIM) + rot)
    logits = X @ W
    # miscalibration drift: severity SHRINKS temperature (over-confidence),
    # so entropy/MSP look BETTER exactly when things get worse.
    T = 1.0 / (1.0 + 1.5 * sev)
    logits = logits / T
    acc0 = float((logits.argmax(1) == y).mean())
    # adaptation = sharpening: helps if boundary mostly right, hurts if rotated
    harmful = acc0 < 0.62
    band = int(min(sev * 3, 2))
    return logits, bool(harmful), band


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_auc(F: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    """Cross-validated harm-AUC of a logistic score on feature set F."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    idx = np.arange(len(y)) % folds
    scores = np.empty(len(y))
    for f in range(folds):
        tr, te = idx != f, idx == f
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(F[tr]), y[tr])
        scores[te] = clf.predict_proba(sc.transform(F[te]))[:, 1]
    return auc(scores, y)


def main() -> int:
    feats, harm, bands = [], [], []
    for _ in range(N_COND):
        L, h, b = make_condition(RNG)
        feats.append(extract_all(L))
        harm.append(h)
        bands.append(b)
    names = list(feats[0].keys())
    F = np.array([[f[n] for n in names] for f in feats])
    y = np.array(harm, dtype=int)
    bands = np.array(bands)
    i_base = [names.index("entropy"), names.index("msp")]
    i_all = list(range(len(names)))

    auc_base = cv_auc(F[:, i_base], y)
    auc_all = cv_auc(F[:, i_all], y)
    uplift = auc_all - auc_base

    per_band = {}
    stable = True
    for b in (0, 1, 2):
        m = bands == b
        if m.sum() < 60 or len(set(y[m])) < 2:
            continue
        ub = cv_auc(F[m][:, i_all], y[m]) - cv_auc(F[m][:, i_base], y[m])
        per_band[f"band{b}"] = dict(n=int(m.sum()), uplift=float(ub))
        stable &= ub >= -0.02

    single = {n: auc(F[:, i], y) for i, n in enumerate(names)}
    c1 = uplift >= 0.05
    out = dict(checks=dict(C1_uplift=bool(c1), C2_stable=bool(stable)),
               auc_baselines=float(auc_base), auc_with_new=float(auc_all),
               uplift=float(uplift), per_band=per_band,
               single_feature_auc=single,
               harmful_rate=float(y.mean()),
               PASS=bool(c1 and stable))
    print(json.dumps(out, indent=1))
    with open(__file__.replace(".py", "_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
