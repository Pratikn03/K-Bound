"""Baseline load test for the UAIS API (Gate P P15).

Runs against an in-process TestClient — no live server required. For multi-replica
load against a deployed stack, use ``locustfile.py`` with Locust.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_baseline(*, n_requests: int = 200, concurrency: int = 1) -> dict:
    import os

    os.environ.setdefault("UAIS_API_KEYS", "loadtest-key")
    os.environ.setdefault("UAIS_CORS_ORIGINS", "https://ops.example")
    os.environ["UAIS_PRODUCTION_MODE"] = "false"

    from fastapi.testclient import TestClient

    from deploy.api.main import app

    client = TestClient(app)
    headers = {"X-API-Key": "loadtest-key"}
    latencies: list[float] = []
    errors = 0

    paths = [("/health", "GET", None), ("/kga/health", "GET", None)]
    t0 = time.perf_counter()
    for i in range(n_requests):
        path, method, body = paths[i % len(paths)]
        start = time.perf_counter()
        if method == "GET":
            resp = client.get(path, headers=headers if path != "/health" else None)
        else:
            resp = client.post(path, headers=headers, json=body)
        latencies.append((time.perf_counter() - start) * 1000)
        if resp.status_code >= 500:
            errors += 1
    elapsed = time.perf_counter() - t0

    return {
        "gate": "P15_load_baseline",
        "n_requests": n_requests,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 3),
        "rps": round(n_requests / elapsed, 2) if elapsed > 0 else 0,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 2),
            "max": round(max(latencies), 2),
        },
        "errors_5xx": errors,
        "pass": errors == 0 and statistics.median(latencies) < 500,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "BASELINE_RESULTS.json")
    args = ap.parse_args()
    report = run_baseline(n_requests=args.n)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
