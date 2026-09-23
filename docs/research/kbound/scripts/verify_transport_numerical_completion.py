#!/usr/bin/env python3
"""Fresh bounded numerical replay of the preserved synthetic transport design.

This does not replace the 4,860 historical trials or access a real target. It
checks the new arithmetic implementation on one deterministic count draw for
each existing scenario/sample-size pair and all declared rho values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from kga.paired_transport import (  # noqa: E402
    TransportSpec,
    accuracy_contrasts,
    confidence_bounds,
    solve_benefit,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "docs/research/kbound/next_phase/paired_transport_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    sources = [
        ROOT / "kga/paired_transport.py",
        ROOT / "kga/transport_numerics.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_paired_transport.py",
        ROOT / "tests/test_transport_numerics.py",
        protocol_path,
    ]
    bindings = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sources]
    rng = np.random.default_rng(20260923)
    rows = []
    for scenario in protocol["scenarios"]:
        source_probabilities = np.asarray(scenario["source_conditional"], dtype=float)
        target_probabilities = np.asarray(
            scenario.get("target_unlabeled", np.asarray(scenario["target_joint"]).sum(axis=1)), dtype=float
        )
        classes = source_probabilities.shape[1]
        contrasts = accuracy_contrasts(scenario["pairs"], classes)
        for n in protocol["sample_sizes"]:
            source_counts = np.column_stack([rng.multinomial(n, source_probabilities[:, y]) for y in range(classes)])
            # Preserve the missing-source-class diagnostic's sampling declaration.
            for y in scenario.get("missing_source_classes", []):
                source_counts[:, int(y)] = 0
            target_counts = rng.multinomial(classes * n, target_probabilities)
            start = time.perf_counter()
            bands = confidence_bounds(source_counts, target_counts, alpha=protocol["alpha"])
            bounds_seconds = time.perf_counter() - start
            for rho in protocol["rho_grid"]:
                spec = TransportSpec.from_confidence(
                    bands,
                    contrasts,
                    rho=rho,
                    assumption_contract="synthetic_numerical_replay_fixed_pair_iid_counts_and_declared_rho",
                )
                start = time.perf_counter()
                result = solve_benefit(spec)
                solve_seconds = time.perf_counter() - start
                rows.append(
                    {
                        "scenario": scenario["id"],
                        "n_per_class": n,
                        "rho": rho,
                        "source_counts": source_counts.tolist(),
                        "target_counts": target_counts.tolist(),
                        "confidence_bounds_seconds": bounds_seconds,
                        "solve_seconds": solve_seconds,
                        "result": asdict(result),
                    }
                )
    for binding in bindings:
        if digest(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError("Source changed during numerical replay")
    summary = {
        "schema": "kbound-transport-numerical-completion/1",
        "scope": "new arithmetic verification only; historical scientific trials are unchanged",
        "seed": 20260923,
        "source_bindings": bindings,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "cases": len(rows),
        "exact_feasible_verified": sum(r["result"]["verified"] for r in rows),
        "abstained_without_numerical_certificate": sum(not r["result"]["verified"] for r in rows),
        "actions": {
            action: sum(r["result"]["action"] == action for r in rows) for action in ("ADAPT", "FREEZE", "ABSTAIN")
        },
        "max_bounds_seconds": max(r["confidence_bounds_seconds"] for r in rows),
        "max_solve_seconds": max(r["solve_seconds"] for r in rows),
        "rows": rows,
    }
    output = args.output / "transport_numerical_verification.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "SHA256SUMS.txt").write_text(f"{digest(output)}  {output.name}\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "source_bindings")}))


if __name__ == "__main__":
    main()
