#!/usr/bin/env python3
"""Generate a deterministic, role-disjoint randomized-condition unit manifest.

The output is protocol metadata only.  It does not run a model, inspect labels,
or create an empirical result.  Commit and seal the manifest before collecting
confirmatory units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DISTRIBUTION = {
    "corruption_family": [
        "defocus_blur",
        "motion_blur",
        "snow",
        "fog",
        "brightness",
        "jpeg_compression",
    ],
    "severity": [1, 2, 3, 4, 5],
    "composition": ["iid", "imbalanced", "single_class"],
    "batch_size": [16, 32, 64, 128],
    "candidate": ["tent", "eata", "sar"],
    "model_seed": [0, 1, 2, 3, 4],
}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def generate_manifest(
    *,
    seed: int,
    n_fit: int,
    n_calibration: int,
    n_test: int,
    alpha: float,
    distribution: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    if min(n_fit, n_calibration, n_test) <= 0:
        raise ValueError("all role counts must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    distribution = distribution or DEFAULT_DISTRIBUTION
    required = set(DEFAULT_DISTRIBUTION)
    if set(distribution) != required or any(not distribution[name] for name in required):
        raise ValueError(f"distribution must contain non-empty fields {sorted(required)}")

    rng = np.random.default_rng(seed)
    roles = (
        ("estimator_fit", n_fit),
        ("residual_calibration", n_calibration),
        ("test", n_test),
    )
    units: list[dict[str, Any]] = []
    for role, count in roles:
        for draw_index in range(count):
            draw = {
                key: distribution[key][int(rng.integers(0, len(distribution[key])))]
                for key in sorted(distribution)
            }
            identity = {
                "protocol": "KBOUND_EXACT_CONFIRMATION_v1",
                "generator_seed": seed,
                "role": role,
                "draw_index": draw_index,
                "draw": draw,
            }
            units.append(
                {
                    "unit_id": f"xec-{canonical_sha256(identity)[:20]}",
                    "role": role,
                    "draw_index": draw_index,
                    **draw,
                }
            )

    body = {
        "schema_version": 1,
        "protocol_id": "KBOUND_EXACT_CONFIRMATION_v1",
        "status": "DRAFT_UNSEALED",
        "generator_seed": seed,
        "alpha": alpha,
        "sampling": "independent draws with replacement from declared uniform categorical laws",
        "distribution": distribution,
        "role_counts": {
            "estimator_fit": n_fit,
            "residual_calibration": n_calibration,
            "test": n_test,
        },
        "decision_rule": {
            "adapt": "delta_hat - epsilon > 0",
            "freeze": "delta_hat + epsilon < 0",
            "otherwise": "abstain",
        },
        "radius_rule": "r_(ceil((n_cal+1)*(1-alpha))); infeasible rank => +inf",
        "historical_stress_grid_reused": False,
        "units": units,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        return ["manifest has no units"]
    ids = [unit.get("unit_id") for unit in units]
    if len(ids) != len(set(ids)):
        errors.append("unit IDs are not unique")
    role_sets = {
        role: {unit.get("unit_id") for unit in units if unit.get("role") == role}
        for role in ("estimator_fit", "residual_calibration", "test")
    }
    for left, right in (("estimator_fit", "residual_calibration"), ("estimator_fit", "test"), ("residual_calibration", "test")):
        if role_sets[left] & role_sets[right]:
            errors.append(f"roles overlap: {left} and {right}")
    expected_counts = payload.get("role_counts", {})
    for role, ids_for_role in role_sets.items():
        if len(ids_for_role) != expected_counts.get(role):
            errors.append(f"role count mismatch for {role}")
    expected_hash = payload.get("manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    if expected_hash != canonical_sha256(unhashed):
        errors.append("manifest hash mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--n-fit", type=int, default=120)
    parser.add_argument("--n-calibration", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=240)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / "research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json",
    )
    args = parser.parse_args()
    payload = generate_manifest(
        seed=args.seed,
        n_fit=args.n_fit,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        alpha=args.alpha,
    )
    errors = validate_manifest(payload)
    if errors:
        raise SystemExit("invalid generated manifest: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote DRAFT manifest {args.output}")
    print(f"manifest_sha256={payload['manifest_sha256']}")
    print("No model was run and no empirical claim was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
