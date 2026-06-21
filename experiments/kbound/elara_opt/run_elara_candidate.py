"""run_elara_candidate.py — produce an ELARA-Opt candidate and let KGA certify it.

This is the integration seam: ELARA-Opt adapts a frozen f0 on an UNLABELED stream,
then the candidate is consumed by KGA exactly like any other candidate —
  * label-free evidence  Z = evidence_vector(f0, fa, probe)        [test side: no labels]
  * paired benefit       X_i = loss(f0_i) - loss(fa_i)             [DEV/CALIB labels only]
  * certificate          KGA.certify(X) -> Delta_hat +/- eps -> ADAPT/FREEZE/ABSTAIN
The multi-cell helper additionally exercises the pipeline decide_kga / policy_metrics.

`dev_y` is the DEV/calibration label split (allowed before the locked eval). No
held-out TEST labels ever enter this function.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ._compat import (
    evidence_vector, eval_frozen, predict_logits, balanced_acc,
    KGA, decide_kga, policy_metrics, label_regime,
)
from .config import ELARA_OPT_DEFAULTS
from .elara_opt import ELARAOptAdapter
from .gate import MetaGate


def run_elara_candidate(
    f0, adapt_stream: List, eval_x, dev_y: np.ndarray, num_classes: int,
    mode: str = "elara_rule", *, steps: int = 1, lr: float = 1e-3,
    cfg: Optional[Dict] = None, meta_model: Optional[MetaGate] = None,
    seed: int = 0, alpha: float = 0.1,
) -> Dict:
    cfg = cfg or ELARA_OPT_DEFAULTS
    adapter = ELARAOptAdapter(mode=mode, cfg=cfg, meta_model=meta_model, seed=seed)
    fa, upd = adapter.adapt(f0, adapt_stream, steps, lr, num_classes)
    tele = adapter.last_telemetry

    probe = adapt_stream[0]
    Z = evidence_vector(f0, fa, probe, num_classes, upd)          # 11-dim, label-free

    a0, preds0, _ = eval_frozen(f0, eval_x, dev_y)                # DEV-labeled
    logits_a = predict_logits(fa, eval_x, train_mode=True)
    preds_a = logits_a.argmax(axis=1)
    aa = balanced_acc(preds_a, dev_y)
    B = float(aa - a0)

    correct0 = (preds0 == dev_y).astype(float)
    correcta = (preds_a == dev_y).astype(float)
    Xi = correcta - correct0                                      # loss(f0)-loss(fa), 0/1 loss

    kga = KGA(alpha=alpha, method="ebern")
    cert = kga.certify(scores=Xi, benefit_range=2.0)
    decision = kga.decide(cert)

    return {
        "mode": mode, "seed": seed,
        "candidate_hash": tele["summary"]["candidate_hash"],
        "update_norm": float(upd),
        "Z": [float(z) for z in Z],
        "a0_frozen": float(a0), "aa_adapted": float(aa), "B_dev_benefit": B,
        "regime_dev": label_regime(B),
        "kga_delta_hat": float(cert.delta_hat), "kga_epsilon": float(cert.epsilon),
        "kga_lower": float(cert.lower), "kga_upper": float(cert.upper),
        "kga_decision": str(decision), "alpha": float(alpha),
        "telemetry": tele,
    }


def kga_decide_multicell(records: List[Dict], alpha: float = 0.1) -> Dict:
    """Pipeline route: stack per-cell (Z, B, a0, aa) and run decide_kga +
    policy_metrics, exactly as the runners do. Label-free on the test side."""
    Z = np.array([r["Z"] for r in records], dtype=float)
    B = np.array([r["B_dev_benefit"] for r in records], dtype=float)
    a0 = np.array([r["a0_frozen"] for r in records], dtype=float)
    aa = np.array([r["aa_adapted"] for r in records], dtype=float)
    Bhat, eps, dec = decide_kga(Z, B, alpha=alpha)
    metrics = policy_metrics(dec, a0, aa, B=B, alpha=alpha)
    return {
        "n_cells": len(records),
        "Bhat": [float(x) for x in Bhat], "eps": float(eps),
        "decisions": [str(d) for d in dec], "policy_metrics": metrics,
    }
