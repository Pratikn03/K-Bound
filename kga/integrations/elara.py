"""Claim-safe ELARA-U candidate routing behind the KGA decision layer.

ELARA-U constructs a validation-fitted detector candidate. KGA then certifies
whether that candidate should replace the validation-selected frozen expert.
The integration is optional: importing :mod:`kga` does not import ELARA-U.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, cast

import numpy as np

from kga._validation import as_float_array
from kga.benefit import FrozenLinearBenefitEstimator
from kga.kga import KGA
from kga.policy import Decision


class EvaluationMode(str, Enum):
    """Information boundary used to construct the KGA certificate."""

    RETROSPECTIVE_AUDIT = "retrospective_audit"
    TARGET_LABEL_LIGHT = "target_label_light"
    LABEL_FREE = "label_free"


@dataclass
class ELARAKGAResult:
    """One frozen ELARA candidate and the KGA decision over it."""

    mode: EvaluationMode
    decision: Decision
    deployed_action: str
    router_action: str
    frozen_expert: int
    frozen_scores: np.ndarray
    candidate_scores: np.ndarray
    deployed_scores: np.ndarray
    certificate: dict[str, float | int | str | None]
    evidence: dict[str, float]
    labels_used_for_decision: int
    claim_tier: str
    claim_eligible: bool
    claim_reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe decision record without per-example scores."""

        return {
            "mode": self.mode.value,
            "decision": self.decision.value,
            "deployed_action": self.deployed_action,
            "router_action": self.router_action,
            "frozen_expert": self.frozen_expert,
            "certificate": self.certificate,
            "evidence": self.evidence,
            "labels_used_for_decision": self.labels_used_for_decision,
            "claim_tier": self.claim_tier,
            "claim_eligible": self.claim_eligible,
            "claim_reasons": list(self.claim_reasons),
        }


def _as_scores(value: np.ndarray, name: str) -> np.ndarray:
    arr = as_float_array(value)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D score array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_binary_labels(value: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = as_float_array(value).ravel()
    if arr.size != n:
        raise ValueError(f"{name} length {arr.size} does not match score rows {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if not set(np.unique(arr)).issubset({0, 1}):
        raise ValueError(f"{name} must contain binary labels 0/1")
    return arr.astype(int)


def _load_router_api():
    try:
        from uais.elara_u.router import RouterPolicy, reliability_features, route
    except ImportError:
        try:
            from src.uais.elara_u.router import RouterPolicy, reliability_features, route
        except ImportError as exc:  # pragma: no cover - exercised in standalone installs
            raise ImportError(
                "The ELARA integration requires the optional `uais.elara_u` package. "
                "Install the full repository package to use ELARAKGAGuard."
            ) from exc
    return RouterPolicy, reliability_features, route


def _brier_benefits(y: np.ndarray, frozen: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    for name, scores in (("frozen", frozen), ("candidate", candidate)):
        if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(
                f"Brier-bound {name} scores must be probabilities in [0, 1]; "
                "unbounded detector scores do not justify benefit_range=2.0"
            )
    return np.asarray((frozen - y) ** 2 - (candidate - y) ** 2, dtype=float)


def _certificate_record(cert) -> dict[str, float | int | str | None]:
    def finite(value: Any) -> float | None:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None

    return {
        "delta_hat": finite(cert.delta_hat),
        "epsilon": finite(cert.epsilon),
        "lower": finite(cert.lower),
        "upper": finite(cert.upper),
        "method": str(cert.method),
        "alpha": float(cert.alpha),
        "n": int(cert.n),
    }


def _candidate_unavailable(
    mode: EvaluationMode, frozen_expert: int, frozen: np.ndarray, router_action: str
) -> ELARAKGAResult:
    """Retain an already validated fallback without inventing a candidate score."""
    return ELARAKGAResult(
        mode=mode,
        decision=Decision.ABSTAIN,
        deployed_action="retain_frozen",
        router_action=router_action,
        frozen_expert=frozen_expert,
        frozen_scores=frozen.copy(),
        candidate_scores=np.full_like(frozen, np.nan),
        deployed_scores=frozen.copy(),
        certificate={"availability": "unavailable", "reason": "candidate scores must be finite and match test rows"},
        evidence={},
        labels_used_for_decision=0,
        claim_tier="unavailable",
        claim_eligible=False,
        claim_reasons=("candidate_scores_unavailable",),
    )


@dataclass
class ELARAKGAGuard:
    """Use KGA to guard deployment of an ELARA-U routed candidate."""

    alpha: float = 0.10
    router_action: str = "hybrid"
    probe_seed: int = 20260615
    policy: object | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.router_action not in {"select", "fuse", "hybrid"}:
            raise ValueError("router_action must be select, fuse, or hybrid")

    def decide(
        self,
        *,
        s_val: np.ndarray,
        y_val: np.ndarray,
        s_test: np.ndarray,
        mode: EvaluationMode | str,
        y_test: np.ndarray | None = None,
        probe_indices: np.ndarray | None = None,
        estimator: FrozenLinearBenefitEstimator | None = None,
        protocol_sha256: str | None = None,
        expected_estimator_payload_sha256: str | None = None,
    ) -> ELARAKGAResult:
        """Build the ELARA candidate and apply KGA under a declared mode."""

        try:
            mode = EvaluationMode(mode)
        except ValueError as exc:
            raise ValueError(f"unknown evaluation mode: {mode!r}") from exc

        # Validate this boundary before any unavailable-candidate early return.
        if mode is EvaluationMode.LABEL_FREE:
            if y_test is not None:
                raise ValueError("label_free mode must not receive y_test")
            if probe_indices is not None:
                raise ValueError("label_free mode must not receive probe_indices")

        s_val = _as_scores(s_val, "s_val")
        s_test = _as_scores(s_test, "s_test")
        if s_val.shape[1] != s_test.shape[1]:
            raise ValueError("s_val and s_test must have the same number of experts")
        y_val = _as_binary_labels(y_val, s_val.shape[0], "y_val")

        RouterPolicy, reliability_features, route = _load_router_api()
        policy = self.policy if self.policy is not None else RouterPolicy()
        reliability = reliability_features(s_val.copy(), y_val.copy())
        if not isinstance(reliability, Mapping) or "val_auc" not in reliability:
            raise ValueError("reliability must provide val_auc for frozen-expert selection")
        try:
            val_auc = as_float_array(reliability["val_auc"])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("val_auc must contain finite numeric values") from exc
        if val_auc.shape != (s_val.shape[1],) or not np.all(np.isfinite(val_auc)):
            raise ValueError("val_auc must contain one finite, unmasked value per expert")
        frozen_expert = int(np.argmax(val_auc))
        frozen = np.array(s_test[:, frozen_expert], dtype=float, copy=True)
        # Candidate construction must not mutate the saved fallback, source
        # inputs, or the evidence subsequently computed on those inputs.
        candidate, router_action = route(s_val.copy(), y_val.copy(), s_test.copy(), policy, action=self.router_action)
        candidate = as_float_array(candidate).copy().ravel()
        if candidate.shape != frozen.shape or not np.all(np.isfinite(candidate)):
            return _candidate_unavailable(mode, frozen_expert, frozen, str(router_action))

        kga = KGA(alpha=self.alpha)
        evidence = kga.evidence(
            s_val,
            s_test,
            extra={
                "best_val_auc": float(reliability["best_auc"]),
                "val_gap": float(reliability["gap"]),
                "val_disagreement": float(reliability["disagreement"]),
                "n_experts": int(s_val.shape[1]),
            },
        )
        feature_map = {
            "ks_mean": float(evidence.ks_mean),
            "ks_max": float(evidence.ks_max),
            "disagree": float(evidence.disagree),
            "entropy_shift": float(evidence.entropy_shift),
            "conf_shift": float(evidence.conf_shift),
            "ess_frac": float(evidence.ess_frac),
            "best_val_auc": float(reliability["best_auc"]),
            "val_gap": float(reliability["gap"]),
            "val_disagreement": float(reliability["disagreement"]),
            "n_experts": float(s_val.shape[1]),
        }

        claim_reasons: tuple[str, ...]
        unavailable_reason: str | None = None
        if mode is EvaluationMode.LABEL_FREE:
            try:
                if estimator is None:
                    raise ValueError("label_free mode requires a frozen estimator")
                if protocol_sha256 is None:
                    raise ValueError("label_free mode requires the externally authorized protocol SHA-256")
                cert = kga.certify_evidence(
                    estimator,
                    protocol_sha256=protocol_sha256,
                    expected_estimator_payload_sha256=expected_estimator_payload_sha256,
                    features=feature_map,
                    evidence_schema_version=estimator.evidence_schema_version,
                )
            except (ValueError, TypeError, OSError, FloatingPointError, OverflowError) as exc:
                # The frozen scores are already available. Missing authority
                # does not justify a negative-benefit assertion (FREEZE).
                cert = None
                unavailable_reason = str(exc)
            labels_used = 0
            claim_tier = "label_free_candidate"
            claim_reasons = ("requires_heldout_aggregate_promotion_check",)
        elif mode is EvaluationMode.TARGET_LABEL_LIGHT:
            if y_test is None:
                raise ValueError("target_label_light mode requires y_test for the fixed probe")
            labels = _as_binary_labels(y_test, s_test.shape[0], "y_test")
            if probe_indices is None:
                raise ValueError("target_label_light mode requires fixed probe_indices")
            # Object dtype preserves mixed Python bool/int inputs until they
            # are checked, unlike NumPy's automatic integer promotion.
            raw_idx: np.ma.MaskedArray = np.ma.asarray(probe_indices, dtype=object)
            if raw_idx.ndim != 1 or np.any(np.ma.getmaskarray(raw_idx)):
                raise ValueError("probe_indices must be an unmasked 1-D integer array")
            if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in raw_idx.data):
                raise ValueError("probe_indices must contain genuine integers, not booleans or coerced values")
            idx = np.asarray(raw_idx.data)
            if idx.size == 0 or np.unique(idx).size != idx.size:
                raise ValueError("probe_indices must be non-empty and unique")
            if np.any(idx < 0) or np.any(idx >= labels.size):
                raise ValueError("probe_indices are out of range")
            idx = idx.astype(np.intp)
            benefits = _brier_benefits(labels[idx], frozen[idx], candidate[idx])
            cert = kga.certify_probe(benefits, k=None, benefit_range=2.0)
            labels_used = int(idx.size)
            claim_tier = "target_label_light_candidate"
            claim_reasons = ("requires_heldout_aggregate_promotion_check",)
        else:
            if y_test is None:
                raise ValueError("retrospective_audit mode requires y_test")
            if probe_indices is not None:
                raise ValueError("retrospective_audit mode does not accept probe_indices")
            labels = _as_binary_labels(y_test, s_test.shape[0], "y_test")
            benefits = _brier_benefits(labels, frozen, candidate)
            cert = kga.certify(scores=benefits, benefit_range=2.0)
            labels_used = int(labels.size)
            claim_tier = "retrospective_only"
            claim_reasons = ("uses_full_target_labels_for_decision", "not_deployment_eligible")

        decision = kga.decide(cert) if cert is not None else Decision.ABSTAIN
        if cert is None:
            claim_tier = "unavailable"
            claim_reasons = ("estimator_authority_unavailable",)
        deploy_candidate = decision is Decision.ADAPT
        deployed_action = (
            "adapt" if decision is Decision.ADAPT else "freeze" if decision is Decision.FREEZE else "retain_frozen"
        )
        deployed_scores = candidate if deploy_candidate else frozen
        return ELARAKGAResult(
            mode=mode,
            decision=decision,
            deployed_action=deployed_action,
            router_action=str(router_action),
            frozen_expert=frozen_expert,
            frozen_scores=frozen,
            candidate_scores=candidate,
            deployed_scores=deployed_scores,
            certificate=(
                _certificate_record(cert)
                if cert is not None
                else {"availability": "unavailable", "reason": unavailable_reason}
            ),
            evidence=feature_map,
            labels_used_for_decision=labels_used,
            claim_tier=claim_tier,
            claim_eligible=False,
            claim_reasons=claim_reasons,
        )


def _binary_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    from scipy.stats import rankdata

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_result(result: ELARAKGAResult, y_test: np.ndarray) -> dict[str, float | bool | None]:
    """Evaluate a frozen decision; labels cannot alter the stored decision."""

    labels = _as_binary_labels(y_test, result.frozen_scores.size, "y_test")
    auc_frozen = _binary_auroc(labels, result.frozen_scores)
    auc_candidate = _binary_auroc(labels, result.candidate_scores)
    auc_kga = _binary_auroc(labels, result.deployed_scores)
    oracle = max(auc_frozen, auc_candidate)
    benefit = float(np.mean(_brier_benefits(labels, result.frozen_scores, result.candidate_scores)))
    false_adapt = bool(result.decision is Decision.ADAPT and benefit <= 0.0)
    false_freeze = bool(result.decision is Decision.FREEZE and benefit >= 0.0)
    lower, upper = result.certificate.get("lower"), result.certificate.get("upper")
    finite_interval = all(
        isinstance(x, (float, int)) and not isinstance(x, bool) and np.isfinite(x) for x in (lower, upper)
    )
    # This is offline inclusion of the measured Brier benefit, not population
    # coverage and not the frequency of strict decisions.
    inclusion = bool(float(cast(float, lower)) <= benefit <= float(cast(float, upper))) if finite_interval else None
    return {
        "auroc_frozen": auc_frozen,
        "auroc_candidate": auc_candidate,
        "auroc_kga": auc_kga,
        "auroc_oracle": oracle,
        "regret_frozen": oracle - auc_frozen,
        "regret_candidate": oracle - auc_candidate,
        "regret_kga": oracle - auc_kga,
        "brier_benefit": benefit,
        "harmful_candidate": benefit <= 0.0,
        "false_adapt": false_adapt,
        "false_freeze": false_freeze,
        "committed": result.decision is not Decision.ABSTAIN,
        "cell_interval_included": inclusion,
    }
