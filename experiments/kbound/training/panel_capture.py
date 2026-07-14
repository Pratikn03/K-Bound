"""panel_capture.py — Wave-5 wiring: agreement-matrix + ev2-evidence capture.

Additive helper for the WILDS runners (run_camelyon17_kbound.py,
run_imagenetr_kbound.py). Pure numpy, torch-free at import; tensors are
converted defensively. Fabricates nothing: everything is computed from the
per-candidate predictions / logits the runner already holds in memory.

Two jobs:
  1. panel_fields(preds_mat) -> {"c_ij": KxK list, "n_D": int}
     Empirical pairwise agreement matrix c_ij = 2*mean(pred_i == pred_j) - 1
     over the panel (row 0 = freeze_f0), and the panel disagreement-region
     count n_D (# eval points where not all candidates agree). This is the
     input the Wave-5 self-normalized tau gate needs
     (docs/research/kbound/gapclose_wave5/tau_selfnorm.py).
  2. ev2_vector(logits) -> list[float] with EV2_NAMES
     The Wave-5 evidence features (MaNo / nuclear norm / GdScore proxy +
     entropy/msp baselines) from kga.evidence_v2.
"""
from __future__ import annotations

import numpy as np

# WIN_HUNT_v2 Arm B: extended with per-sample distributional stats (13 dims).
# Order is FEATURE_ORDER from kga.evidence_v2; kept in explicit sync here so a
# stale import fails loudly rather than silently reordering.
EV2_NAMES = ["ev2_entropy", "ev2_msp", "ev2_mano", "ev2_nuclear", "ev2_gdscore",
             "ev2_ent_q10", "ev2_ent_q50", "ev2_ent_q90",
             "ev2_margin_q10", "ev2_margin_q50", "ev2_margin_q90",
             "ev2_energy_mean", "ev2_energy_q10"]


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def panel_fields(preds_mat) -> dict:
    """Agreement matrix + disagreement count from stacked predictions (K, n)."""
    P = _to_numpy(preds_mat)
    K, n = P.shape
    A = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = float(np.mean(P[i] == P[j]))
    C = 2.0 * A - 1.0
    np.fill_diagonal(C, 1.0)
    n_D = int(np.sum(~np.all(P == P[0:1, :], axis=0)))
    return {"c_ij": [[float(v) for v in row] for row in C], "n_D": n_D}


def ev2_vector(logits) -> list:
    """Wave-5 evidence features from a logits matrix (n, K)."""
    L = _to_numpy(logits).astype(float)
    try:
        from kga.evidence_v2 import extract_all
    except Exception:  # repo-root not on sys.path in some launchers
        import os
        import sys
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", "..", ".."))
        sys.path.insert(0, root)
        from kga.evidence_v2 import extract_all
    f = extract_all(L)
    expected = [n[len("ev2_"):] for n in EV2_NAMES]
    missing = [k for k in expected if k not in f]
    if missing:
        raise RuntimeError(f"evidence_v2/panel_capture out of sync: {missing}")
    return [float(f[k]) for k in expected]


def attach_to_last(records: list, k: int, fields: dict) -> None:
    """Attach panel fields to the last k just-appended candidate records."""
    for r in records[-k:]:
        r.update(fields)
