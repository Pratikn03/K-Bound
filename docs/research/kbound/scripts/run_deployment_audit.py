#!/usr/bin/env python3
"""Run local lifecycle fault injection and overhead measurement, not model TTA.

Real image/model candidate timings are supplied separately by the natural study.
This harness deliberately labels byte-fixture callbacks and never describes
their latency as end-to-end adaptation latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from kga.deployment_audit import DeploymentContract, GateSession, sha256_bytes  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": ordered[min(len(values) - 1, int(0.95 * len(values)))] * 1000,
        "p99_ms": ordered[min(len(values) - 1, int(0.99 * len(values)))] * 1000,
        "max_ms": max(values) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", default=1000, type=int)
    parser.add_argument("--source-checkpoint", type=Path)
    args = parser.parse_args()
    if args.repetitions < 10:
        parser.error("at least ten repetitions required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = (ROOT / "kga/deployment_audit.py", Path(__file__), ROOT / "tests/test_deployment_audit.py")
    snapshots = args.output_dir / "execution_sources"
    snapshots.mkdir()
    for source_path in sources:
        shutil.copyfile(source_path, snapshots / source_path.name)
    protocol = {
        "schema": "kga-local-lifecycle-audit-v1",
        "repetitions": args.repetitions,
        "warmups": 20,
        "byte_fixture_size": 8192,
        "wallclock": time.time(),
        "measurement": "trusted in-process lifecycle with durable JSONL writes; callbacks are byte fixtures",
        "scope": "engineering only; no statistical repeated-use or deployment-benefit claim",
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    protocol_hash = digest(args.output_dir / "protocol.json")
    source = b"s" * 8192
    candidate = b"c" * 8192
    batch = b"outcome-free-input-fixture"
    started = time.time()
    contract = DeploymentContract(
        sha256_bytes(source),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        started - 1,
        started + 3600,
        0.1,
        args.repetitions + 50,
    )
    session = GateSession(contract, source, journal=args.output_dir / "receipts.jsonl")

    def evaluate(model, inputs, prediction=0.2):
        return {
            "source_sha256": contract.source_sha256,
            "candidate_sha256": sha256_bytes(model),
            "batch_sha256": sha256_bytes(inputs),
            "adapter_sha256": contract.adapter_sha256,
            "estimator_sha256": contract.estimator_sha256,
            "calibration_sha256": contract.calibration_sha256,
            "schema_sha256": contract.schema_sha256,
            "prediction": prediction,
            "radius": 0.05,
            "alpha": 0.1,
            "target": contract.target,
        }

    def candidate_fixture(*_):
        return candidate

    for i in range(20):
        session.assess(f"warmup-{i}", batch, candidate_fixture, evaluate)
    latencies, dedup, stages, counts = [], [], {}, {"ADAPT": 0, "FREEZE": 0, "ABSTAIN": 0}
    for i in range(args.repetitions):
        prediction = [0.2, -0.2, 0.0][i % 3]
        start = time.perf_counter()
        receipt = session.assess(
            f"measured-{i}", batch, candidate_fixture, lambda model, inputs, p=prediction: evaluate(model, inputs, p)
        )
        latencies.append(time.perf_counter() - start)
        counts[receipt.action] += 1
        for stage, seconds in receipt.timings_seconds:
            stages.setdefault(stage, []).append(seconds)
        start = time.perf_counter()
        again = session.assess(f"measured-{i}", batch, candidate_fixture, evaluate)
        dedup.append(time.perf_counter() - start)
        assert receipt == again
        assert session.source_model == source
        assert session.selected_model(receipt) == (candidate if receipt.action == "ADAPT" else source)
    faults = {}
    for name, key, value in [
        ("tampered_calibration", "calibration_sha256", "e" * 64),
        ("tampered_candidate", "candidate_sha256", "e" * 64),
        ("nan_prediction", "prediction", float("nan")),
        ("negative_radius", "radius", -0.1),
        ("outcome_injection", "realized_benefit", 0.3),
    ]:
        receipt = session.assess(
            name, batch, candidate_fixture, lambda m, x, k=key, v=value: dict(evaluate(m, x), **{k: v})
        )
        faults[name] = {"action": receipt.action, "reason": receipt.reason}
        assert receipt.action == "ABSTAIN" and session.selected_model(receipt) == source

    def fail_candidate(*_):
        raise RuntimeError("injected candidate failure")

    receipt = session.assess("failed_candidate", batch, fail_candidate, evaluate)
    faults["candidate_failure"] = {"action": receipt.action, "reason": receipt.reason}
    receipt = session.assess("measured-0", batch, candidate_fixture, evaluate, now=started + 4000)
    faults["expired_cached_decision"] = {"action": receipt.action, "reason": receipt.reason}
    assert all(x["action"] == "ABSTAIN" for x in faults.values())
    journal_rows = [json.loads(line) for line in (args.output_dir / "receipts.jsonl").read_text().splitlines()]
    journal_head = None
    for row in journal_rows:
        assert row["previous_receipt_sha256"] == journal_head
        journal_head = row["receipt_sha256"]
        unsigned = dict(row, receipt_sha256="")
        assert (
            sha256_bytes(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            == journal_head
        )
    assert session.journal_healthy
    hash_measurement = None
    if args.source_checkpoint:
        start = time.perf_counter()
        payload = args.source_checkpoint.read_bytes()
        read_seconds = time.perf_counter() - start
        values = []
        for _ in range(31):
            start = time.perf_counter()
            payload_hash = sha256_bytes(payload)
            values.append(time.perf_counter() - start)
        hash_measurement = {
            "path": str(args.source_checkpoint),
            "bytes": len(payload),
            "sha256": payload_hash,
            "read_ms": read_seconds * 1000,
            "hash_latency": distribution(values),
            "scope": "checkpoint read/hash only; no inference/adaptation",
        }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "schema": "kga-local-lifecycle-audit-result-v1",
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "platform": platform.platform(),
        "python": sys.version,
        "repetitions": args.repetitions,
        "warmups": 20,
        "outcomes": counts,
        "full_lifecycle_latency": distribution(latencies),
        "duplicate_request_latency": distribution(dedup),
        "stage_latencies": {k: distribution(v) for k, v in stages.items()},
        "fault_injection": faults,
        "assessment_count": session.assessment_count,
        "source_unchanged": sha256_bytes(session.source_model) == contract.source_sha256,
        "journal_verification": {
            "status": "PASS",
            "records": len(journal_rows),
            "head_sha256": journal_head,
            "file_sha256": digest(args.output_dir / "receipts.jsonl"),
            "scope": "complete successful journal for this run; interrupted journals are separately tested and marked incomplete",
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "checkpoint_hash_measurement": hash_measurement,
        "total_wall_seconds": time.time() - started,
        "limitations": [
            "fixture callbacks; real candidate/inference measured in natural study",
            "single local machine, serialized lifecycle, no network SLA",
            "bounded in-memory candidate retention; not a distributed model store",
            "per-request calibration does not establish repeated-use statistical control",
            "no energy meter, multi-tenant load, or long-duration field deployment",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output_dir),
                "latency": report["full_lifecycle_latency"],
                "faults": len(faults),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
