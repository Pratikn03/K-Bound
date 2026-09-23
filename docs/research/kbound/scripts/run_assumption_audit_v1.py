#!/usr/bin/env python3
"""Replay the legacy assumption diagnostics with corrected claim semantics.

This posthoc replay is not execution under the original source/protocol identity.
It preserves the original v1 JSON/Markdown and writes a separate v2 correction.
Does NOT verify exchangeability or manufacture the advertised concept-shift test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
KB = ROOT / "docs/research/kbound"
sys.path.insert(0, str(KB / "kbound_pkg"))

from assumption_audit import run_stress_suite  # noqa: E402


def load_stress_grid_z(seed=0, adapter="tent", n_cal=200, n_dep=100):
    p = (
        ROOT
        / "experiments/kbound/results/stress_grid_multiseed_v1"
        / f"seed{seed}"
        / f"per_condition_cifar10c_{adapter}_seed{seed}.json"
    )
    recs = json.load(open(p))["records"]
    Z = np.array([r["Z"] for r in recs], float)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Z))
    cal = Z[idx[:n_cal]]
    dep_same = Z[idx[n_cal : n_cal + n_dep]]
    # simulated support shift: scale one feature
    dep_shift = dep_same.copy()
    dep_shift[:, 0] *= 1.8
    return cal, dep_same, dep_shift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KB / "theorem_assumption_revision/legacy_assumption_audit_replay_v2",
        help="Fresh correction directory; original v1 authorities are never replaced.",
    )
    args = parser.parse_args()
    z_cal, z_benign, z_shift = load_stress_grid_z()
    resid_cal = np.abs(np.random.default_rng(1).normal(0, 0.02, 200))
    resid_drift = np.abs(np.random.default_rng(3).normal(0.08, 0.02, 100))

    conditions = [
        {"id": "benign_transfer", "z_calib": z_cal, "z_deploy": z_benign[:100]},
        {"id": "evidence_support_shift", "z_calib": z_cal, "z_deploy": z_shift},
        {
            "id": "residual_drift",
            "z_calib": z_cal,
            "z_deploy": z_benign[:100],
            "residuals_calib": resid_cal,
            "residuals_deploy": resid_drift,
        },
        {"id": "concept_shift_witness", "z_calib": z_cal, "z_deploy": z_shift},
        {"id": "mild_helpful_shift", "z_calib": z_cal, "z_deploy": z_benign[:80]},
        {"id": "low_margin_shift", "z_calib": z_cal, "z_deploy": z_benign[80:120]},
    ]
    suite = run_stress_suite(conditions)
    out = {
        "schema": "kbound-legacy-assumption-audit-correction/2",
        "historical_protocol_id": "assumption_audit_v1",
        "status": "posthoc_diagnostic_replay_not_new_statistical_validation",
        "note": "No diagnostic outcome establishes coverage or authorizes ADAPT.",
        "supersession_scope": "Withdraw only the v1 guarantee='applies' and ADAPT recommendations; preserve original measurements and bytes.",
        "limitations": [
            "The historical concept_shift_witness passes the same shifted Z as evidence_support_shift; it does not construct opposite-benefit matched-evidence worlds.",
            "The mild_helpful_shift and low_margin_shift labels are not supported by benefit inputs in this generator; no benefit or margin test was performed.",
            "The feature-range diagnostic does not warn on the recorded one-feature shift; failure to detect that shift is retained, not converted into a pass of the intended safeguard.",
            "Residual drift uses simulated residual arrays, not a new measured target study.",
            "No-warning diagnostics leave exchangeability, risk alignment and coverage unresolved.",
        ],
        "source_bindings": [
            {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (
                KB / "results/assumption_audit_v1.json",
                KB / "reports/assumption_audit_v1.md",
                ROOT / "research_lock/assumption_audit_v1.yaml",
                KB / "kbound_pkg/assumption_audit/__init__.py",
                Path(__file__).resolve(),
                ROOT
                / "experiments/kbound/results/stress_grid_multiseed_v1/seed0/per_condition_cifar10c_tent_seed0.json",
            )
        ],
        "stress_suite": suite,
        "summary": {
            "n_warning": sum(1 for s in suite if s["assumption_status"] == "warning"),
            "n_not_falsified": sum(1 for s in suite if s["assumption_status"] == "not_falsified"),
        },
    }
    out_json = args.output_dir / "assumption_audit_v2.json"
    out_md = args.output_dir / "assumption_audit_v2.md"
    if out_json.exists() or out_md.exists():
        raise FileExistsError("Use a fresh correction directory; existing evidence is preserved")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Corrected legacy assumption diagnostic replay\n\n",
        "This posthoc correction preserves the original v1 artifacts. Their guarantee and ADAPT recommendations are not valid scientific authority.\n\n",
    ]
    for s in suite:
        lines.append(
            f"- **{s['condition_id']}**: `{s['assumption_status']}` → "
            f"action `{s['recommended_safe_action']}`; guarantee `{s['guarantee_wording']}`\n"
        )
    lines.extend(["\n## Scope limitations\n\n", *(f"- {item}\n" for item in out["limitations"])])
    out_md.write_text("".join(lines))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"summary: {out['summary']}")


if __name__ == "__main__":
    main()
