"""kbound_edge.logging -- JSONL window logger with a hard no-live-labels guard.

Every processed window appends exactly one JSON object to a ``.jsonl`` file.
Each record is guaranteed to carry the fields the audit/report tooling and
:mod:`tests.test_log_integrity` rely on:

    schema_version, timestamp, window_id, model_version, config_hash,
    decision, bhat, eps, lower, upper, reason, latency_ms, evidence{...}

LABEL HYGIENE (the guarantee enforced here)
-------------------------------------------
The online path (capture -> adapt -> evidence -> decision -> log) must NEVER see
ground-truth labels.  :func:`assert_no_labels` recursively rejects any payload
whose keys look like labels, and both the runtime (before processing a window)
and the logger (before writing a record) call it.  A leak raises
:class:`LabelLeakError` instead of being silently written.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from kbound_edge import SCHEMA_VERSION

#: Keys that indicate ground-truth labels leaking onto the online path.
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "label", "labels",
        "y", "y_true", "ytrue", "y_label",
        "target", "targets",
        "gt", "ground_truth", "groundtruth",
        "class_id", "class_ids", "true_class", "true_label",
    }
)


class LabelLeakError(ValueError):
    """Raised when a ground-truth label is detected on the online path."""


def assert_no_labels(payload: Any, where: str = "online payload") -> None:
    """Recursively raise :class:`LabelLeakError` if any label-like key is present.

    Dict keys are matched case-insensitively against :data:`FORBIDDEN_LABEL_KEYS`.
    Lists/tuples are scanned element-wise.  Non-container payloads (tensors,
    arrays, scalars) are fine -- only *named* labels are the hazard.
    """
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(k, str) and k.strip().lower() in FORBIDDEN_LABEL_KEYS:
                raise LabelLeakError(
                    f"label-like key '{k}' found in {where}: ground-truth labels "
                    "must never reach the online path (capture/adapt/evidence/decision/log)"
                )
            assert_no_labels(v, where)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            assert_no_labels(item, where)


def config_hash(config: Dict[str, Any]) -> str:
    """Stable short hash of a config dict (used as ``config_hash`` in records)."""
    import hashlib

    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class WindowLogger:
    """Append-only JSONL logger for online windows."""

    def __init__(
        self,
        path: str,
        model_version: str,
        config_hash: str,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.path = path
        self.model_version = model_version
        self.config_hash = config_hash
        self.schema_version = schema_version
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._n = 0

    def log(
        self,
        window_id: int,
        decision: Dict[str, Any],
        evidence: Dict[str, float],
        latency_ms: float,
        frozen_pred: Optional[list] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write one window record and return it.

        ``decision`` is :meth:`kbound_edge.policy.Decision.as_dict` (carries the
        ``decision`` string plus bhat/eps/lower/upper/reason).
        """
        record: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "timestamp": time.time(),
            "window_id": int(window_id),
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "latency_ms": float(latency_ms),
            "evidence": {k: float(v) for k, v in evidence.items()},
        }
        record.update(decision)  # decision, bhat, eps, lower, upper, reason
        if frozen_pred is not None:
            record["frozen_pred"] = list(frozen_pred)
        if extra:
            record.update(extra)

        # Defence in depth: never persist a label, even via `extra`.
        assert_no_labels(record, where="log record")

        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        self._n += 1
        return record

    @property
    def n_written(self) -> int:
        return self._n

    def close(self) -> None:
        if not self._fh.closed:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except Exception:
                pass
            self._fh.close()

    def __enter__(self) -> "WindowLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str) -> list:
    """Read a JSONL window log back into a list of dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
