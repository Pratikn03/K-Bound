#!/usr/bin/env python3
"""Hermetic end-to-end audit of the released KGA interval decision rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kga.certificate import conformal_split
from kga.policy import decide


def _scenario(scenario_id: str, delta_hat: float, delta_true: float) -> dict:
    residuals = np.linspace(0.005, 0.025, 20)
    certificate = conformal_split(delta_hat, residuals, alpha=0.1)
    action = decide(certificate, alpha=0.1).value
    expected = "ADAPT" if delta_true > 0 else "FREEZE"
    return {
        "scenario_id": scenario_id,
        "delta_hat": delta_hat,
        "delta_true": delta_true,
        "epsilon": certificate.epsilon,
        "decision": action,
        "expected_decision": expected,
        "audit_pass": action == expected and abs(delta_hat - delta_true) <= certificate.epsilon,
    }


def run_audit(_root: Path | None = None) -> dict:
    scenarios = [
        _scenario("helpful_synthetic", delta_hat=0.20, delta_true=0.19),
        _scenario("harmful_synthetic", delta_hat=-0.20, delta_true=-0.19),
    ]
    return {
        "schema": "kbound-gate-e2e-audit-v1",
        "decision_rule": "ADAPT iff lower>0; FREEZE iff upper<0; otherwise ABSTAIN",
        "scenarios": scenarios,
        "audit_pass": all(row["audit_pass"] for row in scenarios),
    }


def main() -> int:
    payload = run_audit(Path.cwd())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
