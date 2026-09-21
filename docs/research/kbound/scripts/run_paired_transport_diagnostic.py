#!/usr/bin/env python3
"""Seal, decide, and score a synthetic paired-transport diagnostic; no target files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import scipy  # noqa: E402

from kga.paired_transport import (  # noqa: E402
    TransportSpec,
    accuracy_contrasts,
    binomial_interval,
    break_even_budget,
    confidence_bounds,
    separate_accuracy_interval,
    solve_benefit,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_protocol(protocol):
    if protocol["version"] != 1 or not 0 < protocol["alpha"] < 1:
        raise ValueError("Unsupported protocol version or alpha")
    if not isinstance(protocol["repetitions"], int) or protocol["repetitions"] < 1:
        raise ValueError("A positive predeclared replication count is required")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in protocol["sample_sizes"]):
        raise ValueError("Invalid predeclared sample sizes")
    if any(not np.isfinite(rho) or not 0 <= rho <= 1 for rho in protocol["rho_grid"]):
        raise ValueError("Invalid predeclared sensitivity grid")
    ids = [s["id"] for s in protocol["scenarios"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Scenarios must have unique identities")
    for scenario in protocol["scenarios"]:
        a = np.asarray(scenario["source_conditional"], dtype=float)
        t = np.asarray(scenario["target_joint"], dtype=float)
        if a.shape != t.shape or a.ndim != 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(t)):
            raise ValueError("Invalid generating probability table")
        if (
            np.min(a) < 0
            or np.min(t) < 0
            or not np.allclose(a.sum(axis=0), 1, atol=1e-12, rtol=0)
            or abs(t.sum() - 1) > 1e-12
        ):
            raise ValueError("Generating tables must be probability distributions")
        accuracy_contrasts(scenario["pairs"], a.shape[1])
        q = np.asarray(scenario.get("target_unlabeled", t.sum(axis=1)), dtype=float)
        if q.shape != (a.shape[0],) or not np.allclose(q, t.sum(axis=1), atol=1e-12, rtol=0):
            raise ValueError("Hidden labels and observable target distribution are inconsistent")


def seal(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    sources = [ROOT / "kga/paired_transport.py", Path(__file__).resolve(), ROOT / "tests/test_paired_transport.py"]
    theory = ROOT / "docs/research/kbound/next_phase/paired_transport_theory.md"
    if theory.exists():
        sources.append(theory)
    write_json(
        output / "seal.json",
        {
            "sealed_at_utc": now(),
            "protocol_sha256": digest(output / "protocol.json"),
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "scope": "pre-Monte-Carlo synthetic diagnostic; no external target records",
            "runtime": {
                "python": sys.version,
                "executable": sys.executable,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
    )
    print(json.dumps({"sealed": str(output), "protocol_sha256": digest(output / "protocol.json")}))


def verify_seal(output):
    record = json.loads((output / "seal.json").read_text())
    if digest(output / "protocol.json") != record["protocol_sha256"]:
        raise ValueError("Protocol hash changed after seal")
    for name, expected in record["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Sealed implementation changed: {name}")
    return record


def decide(source_counts, target_counts, pairs, rho, alpha):
    """This interface has no target-label, truth, target-joint, or scenario arguments."""
    bands = confidence_bounds(source_counts, target_counts, alpha=alpha)
    k = source_counts.shape[1]
    c = accuracy_contrasts(pairs, k)
    spec = TransportSpec.from_confidence(
        bands, c, rho=rho, assumption_contract="fixed-pair-independent-counts-and-declared-conditional-TV"
    )
    paired = solve_benefit(spec)
    pair_array = np.asarray(pairs)
    labels = np.arange(k)
    frozen = (pair_array[:, 0, None] == labels).astype(float)
    candidate = (pair_array[:, 1, None] == labels).astype(float)
    separate = separate_accuracy_interval(spec, frozen, candidate)
    # Descriptive plug-in: minimum-norm least squares, projected to the simplex.
    totals = source_counts.sum(axis=0)
    ahat = np.divide(
        source_counts, totals, out=np.full(source_counts.shape, 1 / source_counts.shape[0]), where=totals > 0
    )
    qhat = target_counts / target_counts.sum()
    pi = np.maximum(0.0, np.linalg.lstsq(np.vstack([ahat, np.ones(k)]), np.append(qhat, 1.0), rcond=None)[0])
    pi = pi / pi.sum() if pi.sum() else np.ones(k) / k
    estimate = float(np.sum(c * ahat * pi))
    plugin = {
        "lower": None,
        "upper": None,
        "point_estimate": estimate,
        "action": "ADAPT" if estimate > 1e-10 else "FREEZE" if estimate < -1e-10 else "ABSTAIN",
        "status": "uncertified_plugin",
        "verified": False,
        "witnesses": {},
    }
    return {"paired_lp": asdict(paired), "separate_accuracy": asdict(separate), "plugin": plugin}, spec


def summarize(output):
    decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    outcomes = {r["id"]: r for r in map(json.loads, (output / "outcomes.jsonl").read_text().splitlines())}
    groups = {}
    for decision in decisions:
        key = (decision["scenario"], decision["n_per_class"], decision["rho"], decision["arm"])
        groups.setdefault(key, []).append((decision, outcomes[decision["id"]]))
    rows = []
    for (scenario, n, rho, arm), values in sorted(groups.items()):
        count = len(values)
        adapt = sum(d["action"] == "ADAPT" for d, _ in values)
        freeze = sum(d["action"] == "FREEZE" for d, _ in values)
        wrong = sum(
            (d["action"] == "ADAPT" and o["benefit"] <= 1e-12) or (d["action"] == "FREEZE" and o["benefit"] >= -1e-12)
            for d, o in values
        )
        fa = sum(d["action"] == "ADAPT" and o["benefit"] <= 1e-12 for d, o in values)
        # Failure/empty sets have no reported interval; they are not counted as covered.
        coverage = sum(
            d["lower"] is not None and d["lower"] - 1e-12 <= o["benefit"] <= d["upper"] + 1e-12 for d, o in values
        )
        widths = [d["upper"] - d["lower"] for d, _ in values if d["lower"] is not None]
        row = {
            "scenario": scenario,
            "n_per_class": n,
            "rho": rho,
            "arm": arm,
            "trials": count,
            "true_benefit": values[0][1]["benefit"],
            "true_transport_tv": values[0][1]["actual_rho"],
            "within_declared_transport_class": values[0][1]["actual_rho"] <= rho + 1e-12,
            "adapt_count": adapt,
            "freeze_count": freeze,
            "abstain_count": count - adapt - freeze,
            "wrong_direction_count": int(wrong),
            "wrong_direction_rate": wrong / count,
            "wrong_direction_mc_95_interval": list(binomial_interval(int(wrong), count, 0.05)),
            "false_adapt_count": int(fa),
            "conditional_false_adapt": fa / adapt if adapt else None,
            "interval_inclusion_rate": coverage / count if arm != "plugin" else None,
            "mean_interval_width": float(np.mean(widths)) if widths else None,
            "mean_oracle_regret": float(
                np.mean([max(o["benefit"], 0) - (o["benefit"] if d["action"] == "ADAPT" else 0) for d, o in values])
            ),
            "mean_decision_seconds": float(np.mean([d["decision_seconds"] for d, _ in values])),
            "infeasible_count": sum(d["status"] == "infeasible" for d, _ in values),
            "numerical_failure_count": sum(d["status"] == "numerical_verification_failed" for d, _ in values),
        }
        rows.append(row)
    write_json(
        output / "summary.json",
        {
            "rows": rows,
            "interpretation": "Synthetic diagnostics only. Conditional theorem applies only within the declared transport class. Monte Carlo intervals are descriptive per regime, not simultaneous. Failed/empty LPs abstain. Plugin has no coverage claim.",
            "external_baselines": {
                "official_MaC_LP": "unavailable: no author implementation link located in verified primary article; not replaced with a proxy"
            },
        },
    )
    import csv

    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Paired transport synthetic diagnostic",
        "",
        "All regimes were sealed before sampling. No real target was read.",
        "",
        "| Scenario | n/class | rho | Arm | True benefit | ADAPT / FREEZE / ABSTAIN | Wrong direction | Inclusion | Width |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        inclusion = "NA" if r["interval_inclusion_rate"] is None else f"{r['interval_inclusion_rate']:.3f}"
        width = "NA" if r["mean_interval_width"] is None else f"{r['mean_interval_width']:.3f}"
        lines.append(
            f"| {r['scenario']} | {r['n_per_class']} | {r['rho']} | {r['arm']} | {r['true_benefit']:.3f} | {r['adapt_count']} / {r['freeze_count']} / {r['abstain_count']} | {r['wrong_direction_rate']:.3f} | {inclusion} | {width} |"
        )
    lines += [
        "",
        "Official MaC-LP execution remains unavailable; the separate-accuracy arm is a specified internal coupling relaxation. The plug-in is uncertified. No new LP machinery or deployment guarantee is claimed.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return rows


def run(output):
    seal_record = verify_seal(output)
    protocol = json.loads((output / "protocol.json").read_text())
    validate_protocol(protocol)
    started = time.monotonic()
    with (output / "RUN_STARTED.json").open("x") as handle:
        json.dump({"started_at_utc": now(), "seal_sha256": digest(output / "seal.json")}, handle)
    completed = 0
    # Decision and latent-truth streams are physically separate. Scoring is a later pass.
    with (
        (output / "decisions.jsonl").open("x") as dh,
        (output / "outcomes.jsonl").open("x") as oh,
        (output / "input_counts.jsonl").open("x") as ih,
        (output / "witnesses.jsonl").open("x") as wh,
    ):
        for scenario in protocol["scenarios"]:
            a = np.asarray(scenario["source_conditional"])
            t = np.asarray(scenario["target_joint"])
            q = np.asarray(scenario.get("target_unlabeled", t.sum(axis=1)))
            contrast = accuracy_contrasts(scenario["pairs"], a.shape[1])
            truth = float(np.sum(contrast * t))
            prior = t.sum(axis=0)
            actual_rho = float(0.5 * np.abs(t - a * prior).sum())
            for n in protocol["sample_sizes"]:
                for trial in range(protocol["repetitions"]):
                    key = f"{protocol['seed']}:{scenario['evidence_group']}:{n}:{trial}"
                    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
                    source = np.column_stack(
                        [
                            rng.multinomial(0 if y in scenario.get("missing_source_classes", []) else n, a[:, y])
                            for y in range(a.shape[1])
                        ]
                    )
                    target = rng.multinomial(a.shape[1] * n, q)
                    inputs = {
                        "source_counts": source.tolist(),
                        "target_counts": target.tolist(),
                        "pairs": scenario["pairs"],
                    }
                    input_hash = hashlib.sha256(encoded(inputs)).hexdigest()
                    ident = f"{scenario['id']}:{n}:{trial}"
                    ih.write(encoded({"id": ident, "sha256": input_hash, **inputs}).decode() + "\n")
                    for rho in protocol["rho_grid"]:
                        stamp = time.monotonic()
                        arms, spec = decide(source, target, scenario["pairs"], rho, protocol["alpha"])
                        elapsed = time.monotonic() - stamp
                        for arm, record in arms.items():
                            witness = record.pop("witnesses", {})
                            record.update(
                                {
                                    "id": ident,
                                    "scenario": scenario["id"],
                                    "n_per_class": n,
                                    "trial": trial,
                                    "rho": rho,
                                    "arm": arm,
                                    "input_sha256": input_hash,
                                    "decision_seconds": elapsed,
                                    "timing_scope": "all three arms combined; not end-to-end adaptation",
                                }
                            )
                            dh.write(encoded(record).decode() + "\n")
                            if trial == 0:
                                wh.write(
                                    encoded({"id": ident, "rho": rho, "arm": arm, "witnesses": witness}).decode() + "\n"
                                )
                        if trial == 0 and rho == protocol["rho_grid"][0]:
                            wh.write(
                                encoded({"id": ident, "break_even": asdict(break_even_budget(spec))}).decode() + "\n"
                            )
                        completed += 1
                    oh.write(
                        encoded(
                            {
                                "id": ident,
                                "benefit": truth,
                                "actual_rho": actual_rho,
                                "target_joint": t.tolist(),
                                "target_prior": prior.tolist(),
                            }
                        ).decode()
                        + "\n"
                    )
                print(
                    json.dumps({"scenario": scenario["id"], "n_per_class": n, "completed_regimes": completed}),
                    flush=True,
                )
    decisions_sha = digest(output / "decisions.jsonl")
    rows = summarize(output)
    if digest(output / "decisions.jsonl") != decisions_sha:
        raise RuntimeError("Scoring changed the decision authority")
    verify_seal(output)
    files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "receipt.json")
    write_json(
        output / "receipt.json",
        {
            "status": "complete",
            "finished_at_utc": now(),
            "wall_seconds": time.monotonic() - started,
            "completed_regimes": completed,
            "summary_rows": len(rows),
            "protocol_sha256": seal_record["protocol_sha256"],
            "artifact_sha256": {p.name: digest(p) for p in files},
            "platform": platform.platform(),
            "command": sys.argv,
            "decision_hash_unchanged_by_scoring": True,
            "population_scope": "synthetic conditional transport model only",
        },
    )
    print(json.dumps({"complete": str(output), "summary_rows": len(rows), "wall_seconds": time.monotonic() - started}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["seal", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.operation == "seal":
        if args.protocol is None:
            parser.error("seal requires --protocol")
        seal(args.protocol, args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
