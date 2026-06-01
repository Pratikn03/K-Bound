#!/usr/bin/env python3
"""M4 temporal / deployment monitoring audit (Phase 3 scaffold + healthcare proxy)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "research_lock").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def main() -> int:
    root = _repo_root()
    gap1 = root / "experiments/fusion/healthcare_gap1_patient_stratified_validation.json"
    gdr = root / "experiments/fusion/gate_decision_rule_e2e_audit.json"
    protocol = root / "research_lock/M4_TEMPORAL_STREAM_PROTOCOL_v1.yaml"

    monitoring: dict = {"evaluated": False}
    if gap1.is_file():
        g = json.loads(gap1.read_text(encoding="utf-8"))
        dep = g.get("deployment_audit") or {}
        mon = dep.get("monitoring") or {}
        temporal = g.get("temporal_validation") or dep.get("temporal_validation") or {}
        monitoring = {
            "evaluated": True,
            "source": str(gap1.relative_to(root)),
            "calibration_monitor_ready": mon.get("calibration_monitor_ready"),
            "calibration_alert": mon.get("calibration_alert"),
            "calibration_reasons": mon.get("calibration_reasons", []),
            "thresholds": mon.get("calibration_thresholds"),
            "temporal_order_valid": temporal.get("temporal_order_valid"),
            "split_leakage_count": (g.get("reference") or {}).get("split_leakage_count"),
            "claim_boundary": g.get("claim_boundary"),
        }

    gdr_ok = gdr.is_file()
    report = {
        "protocol": str(protocol.relative_to(root)) if protocol.is_file() else None,
        "status": "SCAFFOLD_WITH_PROXY_EVIDENCE" if monitoring.get("evaluated") else "SCAFFOLD_ONLY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "m4_dedicated_stream_acquired": False,
        "proxy_monitoring": monitoring,
        "gate_decision_rule_e2e_present": gdr_ok,
        "tier_3_p6_partial": bool(monitoring.get("evaluated") and gdr_ok),
        "next_step": "Acquire dedicated industrial temporal RGB+depth stream; run prospective monitoring holdout.",
    }
    out = root / "elara_master_c/audits/m4_temporal_monitoring_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
