"""telemetry.py — structured, label-free telemetry for ELARA-Opt.

Records per-step losses, reliability features, gate weights, trust-region radius,
update norm, gradient cosines, the candidate hash and the seed.  A hard guard
rejects any attempt to log a target label (forbidden keys, or non-JSON values
such as raw tensors).  Tests scan the emitted telemetry to prove no leakage.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List

import numpy as np
import torch

#: substrings forbidden in any telemetry key (target-label guard).
_FORBIDDEN_KEY_SUBSTR = ("label", "target", "y_true", "y_test", "ground_truth", "gt_", "_gt")


def _check_jsonable(key: str, value):
    kl = key.lower()
    for bad in _FORBIDDEN_KEY_SUBSTR:
        if bad in kl:
            raise ValueError(f"telemetry rejected: key '{key}' looks like a target label")
    if isinstance(value, (str, bool, int, float)) or value is None:
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            _check_jsonable(key, v)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _check_jsonable(k, v)
        return
    raise ValueError(
        f"telemetry rejected: value for '{key}' is {type(value)}; only JSON scalars/"
        f"lists/dicts allowed (raw tensors/arrays could smuggle labels)"
    )


class TelemetryCollector:
    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self.seed = int(seed)
        self.steps: List[Dict] = []
        self.summary: Dict = {}

    def log_step(self, record: Dict):
        for k, v in record.items():
            _check_jsonable(k, v)
        self.steps.append(record)

    def finalize(self, summary: Dict):
        for k, v in summary.items():
            _check_jsonable(k, v)
        self.summary = {"mode": self.mode, "seed": self.seed, **summary, "n_steps": len(self.steps)}
        return self.summary

    def to_dict(self) -> Dict:
        return {"summary": self.summary, "steps": self.steps}

    def write_jsonl(self, path: str):
        with open(path, "w") as fh:
            fh.write(json.dumps({"_record": "summary", **self.summary}) + "\n")
            for i, s in enumerate(self.steps):
                fh.write(json.dumps({"_record": "step", "_i": i, **s}) + "\n")


def candidate_hash(affine_params, ndigits: int = 6) -> str:
    """Deterministic content hash of the adapted affine parameters."""
    h = hashlib.sha256()
    for p in affine_params:
        a = p.detach().cpu().to(torch.float64).numpy()
        a = np.round(a, ndigits)
        h.update(a.tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]
