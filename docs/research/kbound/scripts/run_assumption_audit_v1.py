#!/usr/bin/env python3
"""Run assumption_audit_v1 stress suite on cached K-Bound artifacts.

Pre-registered: research_lock/assumption_audit_v1.yaml
Does NOT verify exchangeability — falsification-oriented warnings only.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
KB = ROOT / "docs/research/kbound"
sys.path.insert(0, str(KB / "kbound_pkg"))

from assumption_audit import run_audit, run_stress_suite  # noqa: E402


def load_stress_grid_z(seed=0, adapter="tent", n_cal=200, n_dep=100):
    p = (ROOT / "experiments/kbound/results/stress_grid_multiseed_v1"
         / f"seed{seed}" / f"per_condition_cifar10c_{adapter}_seed{seed}.json")
    recs = json.load(open(p))["records"]
    Z = np.array([r["Z"] for r in recs], float)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Z))
    cal = Z[idx[:n_cal]]
    dep_same = Z[idx[n_cal:n_cal + n_dep]]
    # simulated support shift: scale one feature
    dep_shift = dep_same.copy()
    dep_shift[:, 0] *= 1.8
    return cal, dep_same, dep_shift


def main():
    z_cal, z_benign, z_shift = load_stress_grid_z()
    resid_cal = np.abs(np.random.default_rng(1).normal(0, 0.02, 200))
    resid_dep = np.abs(np.random.default_rng(2).normal(0, 0.02, 100))
    resid_drift = np.abs(np.random.default_rng(3).normal(0.08, 0.02, 100))

    conditions = [
        {"id": "benign_transfer", "z_calib": z_cal, "z_deploy": z_benign[:100]},
        {"id": "evidence_support_shift", "z_calib": z_cal, "z_deploy": z_shift},
        {"id": "residual_drift", "z_calib": z_cal, "z_deploy": z_benign[:100],
         "residuals_calib": resid_cal, "residuals_deploy": resid_drift},
        {"id": "concept_shift_witness", "z_calib": z_cal, "z_deploy": z_shift},
        {"id": "mild_helpful_shift", "z_calib": z_cal, "z_deploy": z_benign[:80]},
        {"id": "low_margin_shift", "z_calib": z_cal, "z_deploy": z_benign[80:120]},
    ]
    suite = run_stress_suite(conditions)
    out = {
        "protocol_id": "assumption_audit_v1",
        "status": "executed_cached_arm",
        "note": "Falsification-oriented; does not verify exchangeability or risk alignment.",
        "stress_suite": suite,
        "summary": {
            "n_warning": sum(1 for s in suite if s["assumption_status"] == "warning"),
            "n_not_falsified": sum(1 for s in suite if s["assumption_status"] == "not_falsified"),
        },
    }
    out_json = KB / "results/assumption_audit_v1.json"
    out_md = KB / "reports/assumption_audit_v1.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    lines = ["# Assumption audit v1 (cached arm)\n", f"Protocol: `assumption_audit_v1`\n\n"]
    for s in suite:
        lines.append(f"- **{s['condition_id']}**: `{s['assumption_status']}` → "
                     f"action `{s['recommended_safe_action']}`; guarantee `{s['guarantee_wording']}`\n")
    out_md.write_text("".join(lines))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"summary: {out['summary']}")


if __name__ == "__main__":
    main()
