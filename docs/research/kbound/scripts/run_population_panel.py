#!/usr/bin/env python3
"""Replay the population-transfer interval on an outcome-blind benchmark panel.

This runner deliberately consumes only fitted predictions, cell radii, and the
fresh-sample size used to justify the sampling radius.  It refuses target or
outcome fields so a scored label cannot enter a population decision.  The
result is an action/interval receipt; it is not population evidence unless the
input panel's sampling and exchangeability provenance is separately sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# Running a repository script by absolute path sets ``sys.path[0]`` to this
# ``scripts`` directory, not to the repository root.  Add the root explicitly
# so the documented CLI works without requiring callers to set PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kga.population_transfer import (
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)


SCHEMA = "kbound_population_panel_v1"
OUTCOME_KEYS = frozenset({"outcome", "target", "label", "y", "benefit", "delta"})
REQUIRED_KEYS = frozenset({"cell_id", "delta_hat", "epsilon", "sample_size"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_record(record: dict[str, Any], index: int) -> None:
    missing = sorted(REQUIRED_KEYS - record.keys())
    if missing:
        raise ValueError(f"record {index} missing required fields: {', '.join(missing)}")
    forbidden = sorted(OUTCOME_KEYS.intersection(record.keys()))
    if forbidden:
        raise ValueError(
            "outcome-dependent fields are forbidden in the population decision input: "
            + ", ".join(forbidden)
        )
    if not isinstance(record["cell_id"], str) or not record["cell_id"]:
        raise ValueError(f"record {index} cell_id must be a non-empty string")
    if isinstance(record["sample_size"], bool) or not isinstance(record["sample_size"], int):
        raise ValueError(f"record {index} sample_size must be an integer")
    if record["sample_size"] <= 0:
        raise ValueError(f"record {index} sample_size must be positive")
    for name in ("delta_hat", "epsilon"):
        if isinstance(record[name], bool) or not isinstance(record[name], (int, float)):
            raise ValueError(f"record {index} {name} must be numeric")


def build_population_panel(
    rows: Iterable[dict[str, Any]],
    *,
    alpha_cell: float,
    delta_sampling: float,
    alpha_population: float,
) -> dict[str, Any]:
    """Build outcome-blind compound intervals and strict actions for each row."""
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"record {index} must be an object")
        _validate_record(row, index)
        sampling_radius = hoeffding_paired_accuracy_radius(
            n=row["sample_size"], delta=delta_sampling
        )
        interval = compose_conditional_population_interval(
            delta_hat=float(row["delta_hat"]),
            epsilon=float(row["epsilon"]),
            r_samp=sampling_radius,
            alpha_cell=alpha_cell,
            delta_sampling=delta_sampling,
            alpha_population=alpha_population,
        )
        records.append(
            {
                "cell_id": row["cell_id"],
                "delta_hat": interval.delta_hat,
                "epsilon": interval.epsilon,
                "sample_size": row["sample_size"],
                "sampling_radius": interval.r_samp,
                "population_radius": interval.population_radius,
                "interval_lower": interval.interval_lower,
                "interval_upper": interval.interval_upper,
                "action": interval.action.value,
            }
        )
    return {
        "schema": SCHEMA,
        "claim_scope": (
            "population-interval action replay only; no population guarantee is asserted "
            "without independent sampling and exchangeability provenance"
        ),
        "outcome_blind_input": True,
        "settings": {
            "alpha_cell": float(alpha_cell),
            "delta_sampling": float(delta_sampling),
            "alpha_population": float(alpha_population),
        },
        "n": len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-cell", type=float, default=0.05)
    parser.add_argument("--delta-sampling", type=float, default=0.05)
    parser.add_argument("--alpha-population", type=float, default=0.10)
    args = parser.parse_args()

    document = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("input must be a JSON array of outcome-blind cell records")
    result = build_population_panel(
        document,
        alpha_cell=args.alpha_cell,
        delta_sampling=args.delta_sampling,
        alpha_population=args.alpha_population,
    )
    result["input_sha256"] = _sha256(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
