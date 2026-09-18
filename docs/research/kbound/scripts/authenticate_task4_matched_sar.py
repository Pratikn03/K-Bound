#!/usr/bin/env python3
"""Authenticate raw outputs from the Task 4 matched SAR control study.

Verifies checks A through K:
- A: All declared records exist per arm.
- B: Schema integrity: all required fields present.
- C: Numeric bounds: no NaN or Inf in numeric fields.
- D: Frozen accuracy identity across arms per condition.
- E: Sample hash identity across arms per condition.
- F: Arm 1 BN-only parameters (lr=0, grad=False, upd_norm=0).
- G: Arm 2 Ref-SAR parameters (lr=2.5e-4, grad=True, freeze_layer4=True).
- H: Arm 3 Rate-only parameters (lr=4.0e-3, grad=True, freeze_layer4=True).
- I: Arm 4 Mask-only parameters (lr=2.5e-4, grad=True, freeze_layer4=False).
- J: Arm 5 Historical combined (lr=4.0e-3, grad=True, freeze_layer4=False).
- K: Model update distinctness across arms.

Generates TASK4_MATCHED_SAR_MANIFEST.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = [
    "arm_id",
    "arm_name",
    "seed",
    "condition_id",
    "corruption",
    "severity",
    "composition",
    "batch_regime",
    "sample_hash",
    "frozen_accuracy",
    "candidate_accuracy",
    "candidate_benefit",
    "oracle_action",
    "oracle_accuracy",
    "always_adapt_regret",
    "always_freeze_regret",
    "candidate_state_hash_before",
    "candidate_state_hash_after",
    "parameter_update_norm",
    "bn_statistics_active",
    "gradient_update_active",
    "learning_rate",
    "freeze_layer4",
    "runtime_sec",
    "failure_status",
]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def authenticate_task4(results_dir: Path, expected_cells: int = 27) -> tuple[bool, dict[str, Any]]:
    raw_dir = results_dir / "raw"
    arm_files = {
        "arm1": raw_dir / "arm1_bn_only.jsonl",
        "arm2": raw_dir / "arm2_reference_sar.jsonl",
        "arm3": raw_dir / "arm3_rate_only.jsonl",
        "arm4": raw_dir / "arm4_mask_only.jsonl",
        "arm5": raw_dir / "arm5_combined.jsonl",
    }

    checks: dict[str, dict[str, Any]] = {}
    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    file_hashes: dict[str, str] = {}

    # Check A: Existence and record count
    missing_files = []
    for arm_key, fpath in arm_files.items():
        if not fpath.exists():
            missing_files.append(str(fpath))
            continue
        file_hashes[arm_key] = sha256_file(fpath)
        recs = []
        for line in fpath.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
        records_by_arm[arm_key] = recs

    if missing_files:
        checks["check_a_file_presence"] = {
            "status": "FAIL",
            "detail": f"Missing files: {missing_files}",
        }
        return False, {"checks": checks}

    arm_counts = {ak: len(recs) for ak, recs in records_by_arm.items()}
    count_ok = all(cnt == expected_cells for cnt in arm_counts.values())
    checks["check_a_record_counts"] = {
        "status": "PASS" if count_ok else "FAIL",
        "expected_cells_per_arm": expected_cells,
        "observed_counts": arm_counts,
    }

    # Check B & C: Schema integrity and Numeric Bounds
    schema_ok = True
    bounds_ok = True
    for arm_key, recs in records_by_arm.items():
        for i, rec in enumerate(recs):
            for f in REQUIRED_FIELDS:
                if f not in rec:
                    schema_ok = False
                    checks[f"check_b_schema_{arm_key}"] = {
                        "status": "FAIL",
                        "detail": f"Record {i} missing field {f}",
                    }
                    break
            for k, v in rec.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    bounds_ok = False
                    checks[f"check_c_bounds_{arm_key}"] = {
                        "status": "FAIL",
                        "detail": f"Record {i} field {k} is {v}",
                    }
                    break

    checks["check_b_schema"] = {"status": "PASS" if schema_ok else "FAIL"}
    checks["check_c_numeric_bounds"] = {"status": "PASS" if bounds_ok else "FAIL"}

    # Check D & E: Frozen accuracy identity and sample hash identity across arms
    first_arm = list(records_by_arm.keys())[0]
    first_recs = records_by_arm[first_arm]
    frozen_identity_ok = True
    sample_identity_ok = True

    for i in range(min(arm_counts.values())):
        cid0 = first_recs[i]["condition_id"]
        fa0 = first_recs[i]["frozen_accuracy"]
        sh0 = first_recs[i]["sample_hash"]
        for ak in records_by_arm:
            if ak == first_arm:
                continue
            rec_ak = records_by_arm[ak][i]
            if rec_ak["condition_id"] != cid0:
                checks["check_alignment"] = {
                    "status": "FAIL",
                    "detail": f"Condition mismatch at index {i}: {cid0} vs {rec_ak['condition_id']}",
                }
                return False, {"checks": checks}
            if rec_ak["frozen_accuracy"] != fa0:
                frozen_identity_ok = False
            if rec_ak["sample_hash"] != sh0:
                sample_identity_ok = False

    checks["check_d_frozen_accuracy_identity"] = {"status": "PASS" if frozen_identity_ok else "FAIL"}
    checks["check_e_sample_hash_identity"] = {"status": "PASS" if sample_identity_ok else "FAIL"}

    # Check F-J: Arm specific configurations
    arm_specs = {
        "arm1": {"lr": 0.0, "grad": False, "freeze_l4": True, "upd_norm_zero": True},
        "arm2": {"lr": 0.00025, "grad": True, "freeze_l4": True, "upd_norm_zero": False},
        "arm3": {"lr": 0.004, "grad": True, "freeze_l4": True, "upd_norm_zero": False},
        "arm4": {"lr": 0.00025, "grad": True, "freeze_l4": False, "upd_norm_zero": False},
        "arm5": {"lr": 0.004, "grad": True, "freeze_l4": False, "upd_norm_zero": False},
    }

    for ak, spec in arm_specs.items():
        recs = records_by_arm[ak]
        cfg_ok = True
        for rec in recs:
            if rec["learning_rate"] != spec["lr"]:
                cfg_ok = False
            if rec["gradient_update_active"] != spec["grad"]:
                cfg_ok = False
            if rec["freeze_layer4"] != spec["freeze_l4"]:
                cfg_ok = False
            if spec["upd_norm_zero"] and rec["parameter_update_norm"] != 0.0:
                cfg_ok = False
            if not spec["upd_norm_zero"] and rec["parameter_update_norm"] < 0.0:
                cfg_ok = False
        checks[f"check_{ak}_configuration"] = {"status": "PASS" if cfg_ok else "FAIL"}

    # Check K: Distinct updates between adapted arms
    distinct_ok = True
    for i in range(min(arm_counts.values())):
        h2 = records_by_arm["arm2"][i]["candidate_state_hash_after"]
        h3 = records_by_arm["arm3"][i]["candidate_state_hash_after"]
        h4 = records_by_arm["arm4"][i]["candidate_state_hash_after"]
        h5 = records_by_arm["arm5"][i]["candidate_state_hash_after"]
        hashes = {h2, h3, h4, h5}
        if len(hashes) < 2 and any(records_by_arm[ak][i]["parameter_update_norm"] > 0 for ak in ["arm2", "arm3", "arm4", "arm5"]):
            distinct_ok = False
    checks["check_k_distinct_updates"] = {"status": "PASS" if distinct_ok else "FAIL"}

    # Overall status
    all_passed = all(c.get("status") == "PASS" for c in checks.values())

    # Summary metrics per arm
    arm_summaries: dict[str, dict[str, Any]] = {}
    for ak, recs in records_by_arm.items():
        n = len(recs)
        if n == 0:
            continue
        acc_a = [r["candidate_accuracy"] for r in recs if r["candidate_accuracy"] is not None]
        acc_0 = [r["frozen_accuracy"] for r in recs if r["frozen_accuracy"] is not None]
        benefits = [r["candidate_benefit"] for r in recs if r["candidate_benefit"] is not None]
        regrets = [r["always_adapt_regret"] for r in recs if r["always_adapt_regret"] is not None]
        norms = [r["parameter_update_norm"] for r in recs if r["parameter_update_norm"] is not None]
        wins = sum(1 for b in benefits if b > 0)
        losses = sum(1 for b in benefits if b < 0)
        ties = sum(1 for b in benefits if b == 0)

        arm_summaries[ak] = {
            "name": recs[0]["arm_name"],
            "arm_id": recs[0]["arm_id"],
            "n_records": n,
            "mean_frozen_accuracy": sum(acc_0) / len(acc_0) if acc_0 else 0.0,
            "mean_candidate_accuracy": sum(acc_a) / len(acc_a) if acc_a else 0.0,
            "mean_candidate_benefit": sum(benefits) / len(benefits) if benefits else 0.0,
            "mean_oracle_regret": sum(regrets) / len(regrets) if regrets else 0.0,
            "mean_update_norm": sum(norms) / len(norms) if norms else 0.0,
            "win_cells": wins,
            "loss_cells": losses,
            "tie_cells": ties,
            "win_rate": wins / n if n else 0.0,
            "loss_rate": losses / n if n else 0.0,
        }

    manifest = {
        "schema_version": "kbound_task4_matched_sar_manifest_v1",
        "status": "PASS" if all_passed else "FAIL",
        "file_sha256": file_hashes,
        "arm_counts": arm_counts,
        "arm_summaries": arm_summaries,
        "verification_checks": checks,
    }

    manifest_path = results_dir / "TASK4_MATCHED_SAR_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest written to: {manifest_path} (Status: {manifest['status']})")

    return all_passed, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/kbound/results/task4_matched_sar_v1"))
    parser.add_argument("--expected-cells", type=int, default=27)
    args = parser.parse_args()

    passed, manifest = authenticate_task4(args.results_dir, expected_cells=args.expected_cells)
    print(f"\nAuthentication result: {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
