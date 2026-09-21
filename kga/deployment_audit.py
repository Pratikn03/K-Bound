"""Local request-scoped KGA lifecycle with evidence binding and audit receipts.

This is a reference integration boundary for trusted in-process callbacks, not
a sandbox or a proof of calibration validity. Repeated requests are limited and
deduplicated for engineering reasons; per-request coverage does not imply a
simultaneous statistical guarantee. A candidate never replaces the source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, NoReturn


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class DeploymentContract:
    source_sha256: str
    adapter_sha256: str
    estimator_sha256: str
    calibration_sha256: str
    schema_sha256: str
    valid_from: float
    expires_at: float
    alpha: float
    max_assessments: int
    target: str = "declared_cell_benefit"
    max_retained_receipts: int = 10_000
    max_journal_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for name in ("source", "adapter", "estimator", "calibration", "schema"):
            value = getattr(self, name + "_sha256")
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"invalid {name} identity")
        if not (
            math.isfinite(self.valid_from) and math.isfinite(self.expires_at) and self.valid_from < self.expires_at
        ):
            raise ValueError("invalid validity window")
        if isinstance(self.alpha, bool) or not 0 < self.alpha < 1:
            raise ValueError("invalid alpha")
        for name in ("max_assessments", "max_retained_receipts", "max_journal_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.target != "declared_cell_benefit":
            raise ValueError("this integration implements cell-benefit evidence only")

    @property
    def sha256(self):
        return sha256_bytes(_canonical(asdict(self)))


@dataclass(frozen=True)
class AssessmentReceipt:
    request_id: str
    batch_sha256: str
    contract_sha256: str
    action: str
    reason: str
    selected_model_sha256: str
    candidate_sha256: str | None
    evidence_sha256: str | None
    observed_at: float | None
    timings_seconds: tuple[tuple[str, float], ...]
    previous_receipt_sha256: str | None
    receipt_sha256: str = ""

    def to_dict(self):
        value = asdict(self)
        value["timings_seconds"] = dict(self.timings_seconds)
        return value


class SessionCapacityError(RuntimeError):
    """No receipt can be safely issued; use the immutable source as fallback.

    The session remains closed. Recovery requires a new reviewed session;
    catching this exception must never authorize candidate selection or retry
    the candidate outside the gate.
    """


class GateSession:
    """Serializes lifecycle transactions; immutable bytes isolate source state.

    Call ``assess`` at each request's actual time, including retries. A receipt
    is an observation at ``observed_at``; retrieving its bytes does not renew its
    validity. Callbacks receive only checkpoint/input bytes and must themselves
    enforce their declared adapter/estimator implementation and data contract.
    For large checkpoints, integrate an external content-addressed store before
    scaling beyond the configured in-memory assessment limit.
    A journal I/O failure permanently closes this session. Its file may contain
    a partial or unacknowledged tail and must be marked incomplete; it is not a
    complete history of deployed actions. Recovery requires a new reviewed
    session, never silent reuse of that journal.

    Receipt and journal-byte limits include rejected requests. A request that
    would exceed either limit permanently closes the session and raises
    ``SessionCapacityError``; all prior selections then fall back to source.
    Cached retries consume no capacity while the session remains open. Receipts
    are never evicted or silently recomputed. The journal records observations,
    not proof of deployed actions: an expiry fallback can itself exhaust the
    budget after an earlier observation was written but never returned.
    Limits cover this session's writes, not external modification of its file.
    """

    def __init__(self, contract: DeploymentContract, source_model: bytes, *, journal: Path | None = None):
        if not isinstance(source_model, bytes) or sha256_bytes(source_model) != contract.source_sha256:
            raise ValueError("source checkpoint does not match contract")
        self.contract = contract
        self.source_model = source_model
        self._journal = Path(journal) if journal is not None else None
        if self._journal is not None:
            self._journal.parent.mkdir(parents=True, exist_ok=True)
            self._journal.open("x").close()
        self._lock = threading.RLock()
        self._cache: dict[str, AssessmentReceipt] = {}
        self._receipts: dict[str, AssessmentReceipt] = {}
        self._deadlines: dict[str, float] = {}
        self._models = {contract.source_sha256: source_model}
        self._head: str | None = None
        self._assessment_count = 0
        self._journal_failed = False
        self._capacity_reason: str | None = None
        self._journal_bytes_reserved = 0

    @property
    def assessment_count(self):
        return self._assessment_count

    @property
    def journal_healthy(self) -> bool:
        return not self._journal_failed

    @property
    def capacity_exhausted(self) -> bool:
        return self._capacity_reason is not None

    @property
    def retained_receipt_count(self) -> int:
        return len(self._receipts)

    @property
    def journal_bytes_reserved(self) -> int:
        """Upper bound on bytes attempted, including any failed partial write."""
        return self._journal_bytes_reserved

    def _close_capacity(self, reason: str) -> NoReturn:
        self._capacity_reason = reason
        raise SessionCapacityError(reason)

    def _require_receipt_capacity(self) -> None:
        if self._capacity_reason is not None:
            raise SessionCapacityError(self._capacity_reason)
        if len(self._receipts) >= self.contract.max_retained_receipts:
            self._close_capacity("retained_receipt_capacity")
        if (
            self._journal is not None
            and not self._journal_failed
            and self._journal_bytes_reserved >= self.contract.max_journal_bytes
        ):
            self._close_capacity("journal_byte_capacity")

    def _receipt(self, request_id, batch_sha, action, reason, now, timings, candidate_sha=None, evidence_sha=None):
        self._require_receipt_capacity()
        if self._journal_failed:
            action, reason = "ABSTAIN", "journal_unavailable"
        selected = candidate_sha if action == "ADAPT" else self.contract.source_sha256
        result = AssessmentReceipt(
            request_id,
            batch_sha,
            self.contract.sha256,
            action,
            reason,
            selected,
            candidate_sha,
            evidence_sha,
            now if math.isfinite(now) else None,
            tuple(sorted(timings.items())),
            self._head,
        )
        result = replace(result, receipt_sha256=sha256_bytes(_canonical(result.to_dict())))
        if self._journal is not None and not self._journal_failed:
            payload = _canonical(result.to_dict()) + b"\n"
            if self._journal_bytes_reserved + len(payload) > self.contract.max_journal_bytes:
                self._close_capacity("journal_byte_capacity")
            # Reserve before attempting I/O: a failed append may already have
            # written a prefix, and the latched journal must not be reused.
            self._journal_bytes_reserved += len(payload)
            try:
                with self._journal.open("ab") as stream:
                    if stream.write(payload) != len(payload):
                        raise OSError("incomplete journal write")
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                self._journal_failed = True
                result = replace(
                    result,
                    action="ABSTAIN",
                    reason="journal_write_failure",
                    selected_model_sha256=self.contract.source_sha256,
                    receipt_sha256="",
                )
                result = replace(result, receipt_sha256=sha256_bytes(_canonical(result.to_dict())))
                self._receipts[result.receipt_sha256] = result
                return result
        self._head = result.receipt_sha256
        self._receipts[result.receipt_sha256] = result
        return result

    def assess(
        self,
        request_id: str,
        batch: bytes,
        generate_candidate: Callable[[bytes, bytes], bytes],
        evaluate: Callable[[bytes, bytes], Mapping],
        *,
        now: float | None = None,
    ):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("request_id must be a nonempty bounded string")
        if not isinstance(batch, bytes):
            raise ValueError("batch must be immutable bytes")
        arrived = time.monotonic()
        submitted_at = time.time() if now is None else float(now)
        with self._lock:
            if self._capacity_reason is not None:
                raise SessionCapacityError(self._capacity_reason)
            observed = submitted_at + (time.monotonic() - arrived)
            start = time.perf_counter()
            batch_sha = sha256_bytes(batch)
            timings = {}

            def finish(action, reason, candidate_sha=None, evidence_sha=None):
                completed_at = submitted_at + (time.monotonic() - arrived)
                if not (
                    math.isfinite(completed_at) and self.contract.valid_from <= completed_at < self.contract.expires_at
                ):
                    action, reason = "ABSTAIN", "calibration_outside_validity_window"
                timings["total_assessment"] = time.perf_counter() - start
                receipt = self._receipt(
                    request_id, batch_sha, action, reason, completed_at, timings, candidate_sha, evidence_sha
                )
                deadline = arrived + (self.contract.expires_at - submitted_at)
                self._deadlines[receipt.receipt_sha256] = deadline
                if receipt.action == "ADAPT" and time.monotonic() >= deadline:
                    # The append/fsync itself can cross expiry. Publish a fallback
                    # observation rather than return the earlier ADAPT observation.
                    receipt = self._receipt(
                        request_id,
                        batch_sha,
                        "ABSTAIN",
                        "calibration_expired_during_journal",
                        submitted_at + (time.monotonic() - arrived),
                        timings,
                        candidate_sha,
                        evidence_sha,
                    )
                return receipt

            if not (math.isfinite(observed) and self.contract.valid_from <= observed < self.contract.expires_at):
                return finish("ABSTAIN", "calibration_outside_validity_window")
            if self._journal_failed:
                return finish("ABSTAIN", "journal_unavailable")
            if request_id in self._cache:
                cached = self._cache[request_id]
                if cached.batch_sha256 != batch_sha:
                    return finish("ABSTAIN", "request_id_conflict")
                if cached.action == "ADAPT" and time.monotonic() >= self._deadlines[cached.receipt_sha256]:
                    return finish("ABSTAIN", "calibration_outside_validity_window")
                return cached
            self._require_receipt_capacity()
            if self._assessment_count >= self.contract.max_assessments:
                return finish("ABSTAIN", "assessment_limit_reached")
            self._assessment_count += 1
            stamp = time.perf_counter()
            try:
                candidate = generate_candidate(self.source_model, batch)
                if not isinstance(candidate, bytes) or not candidate:
                    raise ValueError("candidate must be nonempty immutable bytes")
            except Exception as error:
                timings["candidate_generation"] = time.perf_counter() - stamp
                receipt = finish("ABSTAIN", "candidate_failure:" + type(error).__name__)
                self._cache[request_id] = receipt
                return receipt
            timings["candidate_generation"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            candidate_sha = sha256_bytes(candidate)
            timings["candidate_hash"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            try:
                value = evaluate(candidate, batch)
            except Exception as error:
                timings["evidence_generation"] = time.perf_counter() - stamp
                receipt = finish("ABSTAIN", "estimator_failure:" + type(error).__name__, candidate_sha)
                self._cache[request_id] = receipt
                return receipt
            timings["evidence_generation"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            action, reason, evidence_sha = self._validate_evidence(value, candidate_sha, batch_sha)
            timings["evidence_validation_and_gate"] = time.perf_counter() - stamp
            receipt = finish(action, reason, candidate_sha, evidence_sha)
            if receipt.action == "ADAPT":
                self._models.setdefault(candidate_sha, candidate)
            self._cache[request_id] = receipt
            return receipt

    def _validate_evidence(self, value, candidate_sha, batch_sha):
        identities = {
            name + "_sha256": getattr(self.contract, name + "_sha256")
            for name in ("source", "adapter", "estimator", "calibration", "schema")
        }
        identities.update(candidate_sha256=candidate_sha, batch_sha256=batch_sha)
        required = set(identities) | {"prediction", "radius", "alpha", "target"}
        if not isinstance(value, Mapping) or set(value) != required:
            return "ABSTAIN", "malformed_evidence", None
        if any(type(value[key]) is not str or value[key] != expected for key, expected in identities.items()):
            return "ABSTAIN", "evidence_identity_mismatch", None
        if type(value["target"]) is not str or value["target"] != self.contract.target:
            return "ABSTAIN", "evidence_target_mismatch", None
        try:
            if any(isinstance(value[key], bool) for key in ("prediction", "radius", "alpha")):
                raise ValueError("boolean numeric value")
            prediction, radius, alpha = (float(value[key]) for key in ("prediction", "radius", "alpha"))
            if (
                not math.isfinite(prediction)
                or math.isnan(radius)
                or radius < 0
                or not math.isfinite(alpha)
                or alpha != self.contract.alpha
            ):
                raise ValueError("invalid numerical evidence")
            canonical = dict(
                value,
                prediction=prediction,
                alpha=alpha,
                radius=radius if math.isfinite(radius) else None,
                radius_status="finite" if math.isfinite(radius) else "unavailable",
            )
            digest = sha256_bytes(_canonical(canonical))
        except (ValueError, TypeError, OverflowError):
            return "ABSTAIN", "malformed_evidence", None
        if prediction - radius > 0:
            return "ADAPT", "strict_positive_lower_bound", digest
        if prediction + radius < 0:
            return "FREEZE", "strict_negative_upper_bound", digest
        return "ABSTAIN", "interval_not_directional", digest

    def selected_model(self, receipt: AssessmentReceipt) -> bytes:
        with self._lock:
            if self._receipts.get(receipt.receipt_sha256) != receipt:
                raise ValueError("receipt was not issued by this session")
            if self._journal_failed or self.capacity_exhausted:
                return self.source_model
            if receipt.action == "ADAPT" and time.monotonic() >= self._deadlines[receipt.receipt_sha256]:
                return self.source_model
            return self._models[receipt.selected_model_sha256]
