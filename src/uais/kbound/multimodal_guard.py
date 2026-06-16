"""Target-Label-Light Multimodal Safety Guard.

Composes ELARA-U reliability routing with the KGA adapt/freeze/abstain
certificate.  When a labeled micro-probe is available, uses
:meth:`kga.KGA.certify_probe`; otherwise falls back to label-free
:meth:`kga.KGA.certify` on the full placement-benefit pool.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from kga import KGA
from kga.policy import Decision
from src.uais.elara_u.router import RouterPolicy, _rank_mean, _reliab_weights, reliability_features

Action = Literal["adapt", "freeze", "abstain"]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def placements(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-positive placement; mean equals AUROC (Mann-Whitney)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.array([])
    ns = np.sort(neg)
    below = np.searchsorted(ns, pos, side="left")
    ties = np.searchsorted(ns, pos, side="right") - below
    return (below + 0.5 * ties) / len(neg)


def placement_benefits(y: np.ndarray, s_frozen: np.ndarray, s_adapt: np.ndarray) -> np.ndarray:
    """Per-sample benefit scores whose mean is AUROC(s_adapt) - AUROC(s_frozen)."""
    return placements(y, s_adapt) - placements(y, s_frozen)


def cw_fuse(S: np.ndarray) -> np.ndarray:
    w = 2.0 * np.abs(S - 0.5)
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def relgate_fuse(
    s_val: np.ndarray,
    s_test: np.ndarray,
    valauc: np.ndarray,
    *,
    drift_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Reliability-gated fusion; returns (fused_scores, active_channel_mask)."""
    val_ok = (valauc >= 0.55) & (s_val.std(0) >= 0.02)
    drift = np.array([ks_2samp(s_val[:, m], s_test[:, m]).statistic for m in range(s_val.shape[1])])
    keep = val_ok & (drift <= drift_threshold)
    if keep.sum() == 0:
        keep = val_ok.copy()
    if keep.sum() == 0:
        keep = np.ones(s_val.shape[1], bool)
    w = np.clip(valauc[keep] - 0.5, 1e-6, None)
    fused = s_test[:, keep] @ (w / w.sum())
    return fused, keep


@dataclass
class GuardResult:
    decision: Decision
    action: Action
    test_scores: np.ndarray
    certificate: dict
    channel_mask: np.ndarray | None
    probe_k: int
    auroc_frozen: float
    auroc_adapt: float


@dataclass
class MultimodalGuard:
    """KGA-guarded multimodal fusion router."""

    alpha: float = 0.10
    probe_k: int | None = None
    probe_seed: int = 20260615
    policy: RouterPolicy | None = None

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = RouterPolicy()

    def _frozen_scores(self, s_val: np.ndarray, y_val: np.ndarray, s_test: np.ndarray) -> tuple[np.ndarray, int]:
        f = reliability_features(s_val, y_val)
        best_m = int(np.nanargmax(f["val_auc"]))
        return s_test[:, best_m], best_m

    def guard_category(
        self,
        s_val: np.ndarray,
        y_val: np.ndarray,
        s_test: np.ndarray,
        y_test: np.ndarray,
        valauc: np.ndarray,
        *,
        probe_k: int | None = None,
    ) -> GuardResult:
        """Run certificate + routing for one category with labels on test (probe)."""
        k_use = self.probe_k if probe_k is None else probe_k
        frozen, _ = self._frozen_scores(s_val, y_val, s_test)
        fused, mask = relgate_fuse(s_val, s_test, valauc, drift_threshold=self.policy.drift_threshold)

        benefits = placement_benefits(y_test, frozen, fused)
        kga = KGA(alpha=self.alpha)
        br = float(min(2.0, max(float(np.max(benefits) - np.min(benefits)) + 0.05, 0.1))) if benefits.size >= 2 else 2.0
        if benefits.size < 2:
            cert = None
            decision = Decision.ABSTAIN
        elif k_use is not None and k_use > 0:
            cert = kga.certify_probe(benefits, k=min(k_use, benefits.size), seed=self.probe_seed, benefit_range=br)
            decision = kga.decide(cert)
        else:
            cert = kga.certify(scores=benefits, benefit_range=br)
            decision = kga.decide(cert)

        action: Action
        if decision == Decision.ADAPT:
            action = "adapt"
            scores = fused
        elif decision == Decision.FREEZE:
            action = "freeze"
            scores = frozen
        else:
            action = "abstain"
            scores = _rank_mean(s_test)

        cert_dict = None if cert is None else {
            "n": cert.n, "delta_hat": cert.delta_hat, "epsilon": cert.epsilon,
            "lower": cert.lower, "upper": cert.upper, "method": cert.method,
        }
        return GuardResult(
            decision=decision,
            action=action,
            test_scores=scores,
            certificate=cert_dict or {},
            channel_mask=mask,
            probe_k=0 if k_use is None else int(k_use),
            auroc_frozen=auroc(y_test, frozen),
            auroc_adapt=auroc(y_test, fused),
        )

    def guard_track(
        self,
        cache_files: list[str],
        *,
        probe_sizes: list[int] | None = None,
    ) -> dict:
        """Aggregate k-sweep over cached multimodal .npz categories."""
        probe_sizes = probe_sizes or [0, 8, 16, 32, 64]
        rows = []
        for f in sorted(cache_files):
            z = np.load(f)
            s_val, y_val = z["Sval"], z["yval"]
            s_test, y_test = z["Stest"], z["ytest"]
            valauc = z["valauc"]
            if len(np.unique(y_test)) < 2:
                continue
            if (valauc > 0.6).sum() < 2:
                continue
            for k in probe_sizes:
                r = self.guard_category(s_val, y_val, s_test, y_test, valauc, probe_k=k if k > 0 else None)
                rows.append({
                    "file": f, "k": k, "action": r.action,
                    "decision": r.decision.value,
                    "auroc_out": auroc(y_test, r.test_scores),
                    "auroc_frozen": r.auroc_frozen,
                    "auroc_adapt": r.auroc_adapt,
                    "cert": r.certificate,
                })
        return {"rows": rows, "probe_sizes": probe_sizes}


def load_track_cache(repo_root: str, cache_dir: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(repo_root, cache_dir, pattern)))
