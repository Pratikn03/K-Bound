"""Compatibility facade for the canonical KGA-ELARA integration.

New code should import :class:`kga.integrations.elara.ELARAKGAGuard`. This
module retains the historical API used by experiment scripts. A full-target
decision is explicitly retrospective; it is never labeled as label-free.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from kga.integrations.elara import ELARAKGAGuard, EvaluationMode
from kga.policy import Decision
from src.uais.elara_u.router import RouterPolicy, reliability_features

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
    evaluation_mode: str
    labels_used_for_decision: int


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
        """Delegate one category to the canonical KGA-ELARA integration."""
        k_use = self.probe_k if probe_k is None else probe_k
        if k_use is not None and k_use > 0:
            rng = np.random.default_rng(self.probe_seed)
            n_probe = min(int(k_use), len(y_test))
            probe_indices = np.sort(rng.choice(len(y_test), size=n_probe, replace=False))
            mode = EvaluationMode.TARGET_LABEL_LIGHT
        else:
            probe_indices = None
            mode = EvaluationMode.RETROSPECTIVE_AUDIT
        canonical = ELARAKGAGuard(
            alpha=self.alpha,
            probe_seed=self.probe_seed,
            policy=self.policy,
        ).decide(
            s_val=s_val,
            y_val=y_val,
            s_test=s_test,
            y_test=y_test,
            mode=mode,
            probe_indices=probe_indices,
        )
        action = canonical.decision.value.lower()
        return GuardResult(
            decision=canonical.decision,
            action=action,
            test_scores=canonical.deployed_scores,
            certificate=canonical.certificate,
            channel_mask=None,
            probe_k=canonical.labels_used_for_decision if mode is EvaluationMode.TARGET_LABEL_LIGHT else 0,
            auroc_frozen=auroc(y_test, canonical.frozen_scores),
            auroc_adapt=auroc(y_test, canonical.candidate_scores),
            evaluation_mode=mode.value,
            labels_used_for_decision=canonical.labels_used_for_decision,
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
