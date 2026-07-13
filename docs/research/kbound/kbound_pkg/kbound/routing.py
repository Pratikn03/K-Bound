"""kbound.routing -- Frozen mirror of ``kga.routing`` (Wave 4 multicandidate panel).

Re-vendor from ``kga/routing.py`` when the reproduction package is re-released.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from kbound.certificate import conformal_radius

Selector = Literal["argmax_lcb", "first_positive"]


@dataclass(frozen=True)
class CandidateCertificate:
    index: int
    delta_hat: float
    epsilon: float

    @property
    def lcb(self) -> float:
        return self.delta_hat - self.epsilon


@dataclass(frozen=True)
class RoutingDecision:
    selected: int | None
    certificates: tuple[CandidateCertificate, ...]
    alpha: float
    bonferroni_alpha: float
    committed: bool

    @property
    def decision(self) -> str:
        if self.selected is None:
            return "abstain"
        return "adapt"


def split_conformal_rank_radius(cal_errors: np.ndarray, level: float) -> float:
    return conformal_radius(np.abs(np.asarray(cal_errors, dtype=float).ravel()), alpha=level)


def candidate_lcb_from_calibration(
    deploy_score: float,
    cal_scores: np.ndarray,
    cal_truth: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float, float]:
    cal_scores = np.asarray(cal_scores, dtype=float).ravel()
    cal_truth = np.asarray(cal_truth, dtype=float).ravel()
    if cal_scores.shape != cal_truth.shape:
        raise ValueError("cal_scores and cal_truth must have the same shape")
    residuals = cal_scores - cal_truth
    eps = split_conformal_rank_radius(residuals, alpha)
    delta_hat = float(deploy_score)
    return delta_hat, eps, delta_hat - eps


def multiclass_benefit(mu_d: float, pa: float, p0: float) -> float:
    return float(mu_d) * (float(pa) - float(p0))


def multiclass_harmful(delta: float, pa: float, p0: float, mu_d: float) -> bool:
    if mu_d > 0:
        return float(pa) <= float(p0)
    return float(delta) <= 0.0


def bonferroni_multicandidate_route(
    lcbs: Sequence[float],
    *,
    alpha: float,
    selector: Selector = "argmax_lcb",
) -> int | None:
    if not lcbs:
        return None
    arr = np.asarray(lcbs, dtype=float)
    positive = np.where(arr > 0.0)[0]
    if positive.size == 0:
        return None
    if selector == "first_positive":
        return int(positive[0])
    return int(positive[np.argmax(arr[positive])])


def route_panel(
    deploy_scores: np.ndarray,
    cal_scores: np.ndarray,
    cal_truth: np.ndarray,
    *,
    alpha: float,
    selector: Selector = "argmax_lcb",
) -> RoutingDecision:
    deploy_scores = np.asarray(deploy_scores, dtype=float).ravel()
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_truth = np.asarray(cal_truth, dtype=float)
    k = deploy_scores.size
    if cal_scores.shape != cal_truth.shape or cal_scores.shape[0] != k:
        raise ValueError("deploy_scores length must match cal_scores/cal_truth rows")
    bonf = alpha / k
    certs: list[CandidateCertificate] = []
    lcbs: list[float] = []
    for i in range(k):
        dh, eps, lcb = candidate_lcb_from_calibration(
            deploy_scores[i], cal_scores[i], cal_truth[i], alpha=bonf
        )
        certs.append(CandidateCertificate(index=i, delta_hat=dh, epsilon=eps))
        lcbs.append(lcb)
    selected = bonferroni_multicandidate_route(lcbs, alpha=alpha, selector=selector)
    return RoutingDecision(
        selected=selected,
        certificates=tuple(certs),
        alpha=float(alpha),
        bonferroni_alpha=float(bonf),
        committed=selected is not None,
    )


class _BettingEProcess:
    def __init__(self, alpha: float, a: float = -1.0, b: float = 1.0, cap: float = 0.5) -> None:
        self.a, self.b = a, b
        self.lam_max = cap / (-a)
        self.s1 = 0.0
        self.s2 = 0.25
        self.cnt = 0.0
        self.logw = 0.0
        self.alpha = alpha

    def update(self, x: float) -> float:
        x = float(max(self.a, min(self.b, x)))
        mu = self.s1 / self.cnt if self.cnt > 0 else 0.0
        s2 = self.s2 / max(self.cnt, 1.0)
        lam = float(np.clip(mu / s2 if s2 > 0 else 0.0, 0.0, self.lam_max))
        self.logw += math.log(max(1.0 + lam * x, 1e-300))
        self.s1 += x
        self.s2 += x * x
        self.cnt += 1.0
        return self.logw

    def rejected_null(self, global_alpha: float, k: int) -> bool:
        return self.logw >= math.log(k / global_alpha)


class AnytimeMulticandidatePanel:
    def __init__(self, k: int, alpha: float = 0.1) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.alpha = float(alpha)
        self._procs = [_BettingEProcess(alpha / k) for _ in range(k)]
        self._steps = 0

    def update(self, benefits: Sequence[float]) -> int | None:
        if len(benefits) != self.k:
            raise ValueError(f"expected {self.k} benefits, got {len(benefits)}")
        self._steps += 1
        for i, x in enumerate(benefits):
            self._procs[i].update(float(x))
            if self._procs[i].rejected_null(self.alpha, self.k):
                return i
        return None

    @property
    def steps(self) -> int:
        return self._steps
