"""kbound_edge.replay -- offline recorded-stream runner (the Tier-0/Tier-1 chain).

Runs the full per-window decision chain on a recorded stream of windows:

    frames -> frozen pred p0 (OFFICIAL) -> episodic Tent candidate -> pa
           -> label-free evidence Z -> Bhat = estimator(Z)
           -> kga_decide(Bhat, eps) -> JSONL log

The ONLINE chain (:func:`run_window`) never receives labels.  For offline
evaluation, the caller separately measures the true benefit B from held-out
labels it owns -- those labels are never passed into :func:`run_window`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from kbound_edge.model import predict_proba
from kbound_edge.evidence import edge_evidence_vector, EDGE_EVIDENCE_NAMES
from kbound_edge.policy import kga_decide, Decision, PolicyContext, apply_policy, POLICIES
from kbound_edge.logging import assert_no_labels
from kbound_edge.dataset import frames_to_tensor


@dataclass
class WindowOutcome:
    """Everything produced for one window (model outputs only, no labels)."""

    window_id: int
    decision: Decision
    evidence: Dict[str, float]
    p0: np.ndarray            # frozen softmax (N,C) -- OFFICIAL model output
    pa: np.ndarray            # adapted-candidate softmax (N,C) -- shadow only
    upd_norm: float
    latency_ms: float

    @property
    def frozen_pred(self) -> List[int]:
        return self.p0.argmax(1).tolist()

    @property
    def candidate_pred(self) -> List[int]:
        return self.pa.argmax(1).tolist()


def _to_tensor(window: Any, image_size: int):
    """Coerce a window payload to a model tensor, guarding against label leaks."""
    import torch

    if isinstance(window, dict):
        assert_no_labels(window, where="online window payload")
        if "frames" not in window:
            raise KeyError("window dict must contain a 'frames' entry")
        return _to_tensor(window["frames"], image_size)
    if isinstance(window, torch.Tensor):
        return window
    # assume a sequence of BGR frames
    return frames_to_tensor(window, image_size)


def run_window(
    window_id: int,
    window: Any,
    f0,
    adapter,
    estimator,
    eps: float,
    image_size: int = 64,
) -> WindowOutcome:
    """Run the online decision chain for ONE window.  No labels permitted.

    ``window`` may be a preprocessed tensor, a list of BGR frames, or a dict with
    a ``'frames'`` key (which is guarded against label leakage).
    """
    x = _to_tensor(window, image_size)

    t0 = perf_counter()
    p0 = predict_proba(f0, x)            # frozen model = official output
    res = adapter.adapt(x)               # isolated candidate (f0 untouched)
    pa = predict_proba(res.model, x)     # candidate output (shadow)
    z = edge_evidence_vector(p0, pa, res.upd_norm)
    bhat = estimator.predict_one(z)
    decision = kga_decide(bhat, eps)
    latency_ms = (perf_counter() - t0) * 1000.0

    evidence = {name: float(v) for name, v in zip(EDGE_EVIDENCE_NAMES, z)}
    return WindowOutcome(
        window_id=window_id,
        decision=decision,
        evidence=evidence,
        p0=p0,
        pa=pa,
        upd_norm=float(res.upd_norm),
        latency_ms=latency_ms,
    )


def policy_decisions_for(outcome: WindowOutcome, eps: float,
                         conf_tau: float = 0.5, entropy_tau: float = 0.05) -> Dict[str, str]:
    """Decisions of EVERY comparison policy for one window (for the ablation table)."""
    ctx = PolicyContext(
        bhat=outcome.decision.bhat,
        eps=eps,
        evidence=outcome.evidence,
        conf_tau=conf_tau,
        entropy_tau=entropy_tau,
    )
    return {name: apply_policy(name, ctx) for name in POLICIES}


def replay_windows(
    windows: Sequence[Any],
    f0,
    adapter,
    estimator,
    eps: float,
    logger=None,
    image_size: int = 64,
    collect_policies: bool = True,
    conf_tau: float = 0.5,
    entropy_tau: float = 0.05,
) -> Dict[str, Any]:
    """Replay a sequence of windows; log each; return outcomes + policy decisions.

    Returns a dict with:
        outcomes          list[WindowOutcome]
        decisions         list[str]                 (kga_full decisions)
        latencies_ms      list[float]
        policy_decisions  dict[policy_name -> list[str]]   (if collect_policies)
    """
    outcomes: List[WindowOutcome] = []
    decisions: List[str] = []
    latencies: List[float] = []
    policy_decisions: Dict[str, List[str]] = {name: [] for name in POLICIES}

    for wid, window in enumerate(windows):
        outcome = run_window(wid, window, f0, adapter, estimator, eps, image_size)
        outcomes.append(outcome)
        decisions.append(outcome.decision.decision)
        latencies.append(outcome.latency_ms)

        if logger is not None:
            logger.log(
                window_id=wid,
                decision=outcome.decision.as_dict(),
                evidence=outcome.evidence,
                latency_ms=outcome.latency_ms,
                frozen_pred=outcome.frozen_pred,
                extra={"shadow_candidate_pred": outcome.candidate_pred,
                       "upd_norm": outcome.upd_norm},
            )
        if collect_policies:
            for name, d in policy_decisions_for(outcome, eps, conf_tau, entropy_tau).items():
                policy_decisions[name].append(d)

    return {
        "outcomes": outcomes,
        "decisions": decisions,
        "latencies_ms": latencies,
        "policy_decisions": policy_decisions if collect_policies else None,
    }
