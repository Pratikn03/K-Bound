#!/usr/bin/env python3
"""Generate the controlled population-frontier versus empirical-KGA bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kga.frontier import assess_frontier
from kga.policy import decide_batch

SCHEMA = "kbound_frontier_kga_bridge_v1"
ROWS = (
    # M, beta, realized gamma, Delta_hat, epsilon
    (-0.30, 0.10, 0.04, -0.24, 0.05),
    (-0.10, 0.10, 0.02, -0.07, 0.08),
    (-0.05, 0.10, -0.02, -0.09, 0.04),
    (0.00, 0.10, 0.00, 0.01, 0.04),
    (0.05, 0.10, 0.02, 0.09, 0.04),
    (0.10, 0.10, -0.02, 0.07, 0.08),
    (0.30, 0.10, -0.04, 0.24, 0.05),
)


def build() -> dict:
    records = []
    for index, (margin, beta, gamma, delta_hat, epsilon) in enumerate(ROWS):
        benefit = margin + gamma
        population = assess_frontier(margin, beta)
        empirical = str(decide_batch([delta_hat], [epsilon])[0])
        records.append(
            {
                "cell_id": f"bridge-{index:02d}",
                "M": margin,
                "beta": beta,
                "gamma": gamma,
                "Delta": benefit,
                "Delta_hat": delta_hat,
                "epsilon": epsilon,
                "population_action": population.action.value,
                "empirical_action": empirical,
                "same_action": population.action.value == empirical,
                "interpretation": (
                    "distinct layers agree on this cell"
                    if population.action.value == empirical
                    else "distinct layers disagree; neither radius substitutes for the other"
                ),
            }
        )
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "generator_sha256": source_hash,
        "claim_scope": (
            "controlled algebraic bridge only; beta is externally declared and epsilon is an "
            "empirical radius; this is not a real-data beta estimate"
        ),
        "records": records,
        "summary": {
            "n": len(records),
            "agreement": sum(row["same_action"] for row in records),
            "disagreement": sum(not row["same_action"] for row in records),
            "population_abstain": sum(row["population_action"] == "ABSTAIN" for row in records),
            "empirical_abstain": sum(row["empirical_action"] == "ABSTAIN" for row in records),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json"),
    )
    args = parser.parse_args()
    document = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(document["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
