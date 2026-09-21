#!/usr/bin/env python3
"""Loopback-only production-mode API smoke and latency measurement."""

import argparse
import hashlib
import json
import os
import secrets
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(*args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, check=True, **kwargs).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
    docker = "/usr/local/bin/docker"
    image_id = run(docker, "image", "inspect", args.image, "--format", "{{.Id}}")
    protocol = {
        "image_id": image_id,
        "scope": "local production-mode certificate API only",
        "sequential_repetitions": 100,
        "concurrency": 4,
        "concurrent_repetitions": 100,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": "synthetic residuals and estimates; no real adaptation endpoint",
        "isolation": "loopback host port; readonly rootfs; no capabilities; no-new-privileges",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    env = dict(os.environ, KGA_API_KEYS=secrets.token_hex(32))
    cid = run(
        docker,
        "run",
        "--detach",
        "--rm",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--publish",
        "127.0.0.1::8000",
        "--env",
        "KGA_API_KEYS",
        "--env",
        "KGA_PRODUCTION_MODE=true",
        "--env",
        "KGA_CORS_ORIGINS=http://localhost",
        "--env",
        "KGA_RATE_LIMIT_REQUESTS=5000",
        args.image,
        env=env,
    )
    try:
        address = run(docker, "port", cid, "8000/tcp").splitlines()[0]
        base = "http://" + address

        def request(path, payload=None, authenticated=True):
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["X-API-Key"] = env["KGA_API_KEYS"]
            data = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(base + path, data=data, headers=headers)
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.status, json.loads(response.read()), time.perf_counter() - started
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read()), time.perf_counter() - started

        for _attempt in range(50):
            try:
                status, health, _ = request("/health")
                if status == 200:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("container did not become healthy")
        status, ready, _ = request("/ready")
        assert status == 200 and ready["ready"] and ready["mode"] == "production"
        payload = {
            "calib_scores": [0.1, 0.2, 0.3],
            "test_scores": [0.2, 0.3, 0.4],
            "cert_mode": "full",
            "calib_residuals": [0.05] * 100,
            "delta_hat": 0.2,
            "alpha": 0.1,
        }
        assert request("/decide", payload, authenticated=False)[0] == 403
        assert request("/decide", dict(payload, alpha=2))[0] == 422
        proxy = {"calib_scores": [0.1, 0.2, 0.3], "test_scores": [0.2, 0.3, 0.4], "cert_mode": "proxy"}
        assert request("/decide", proxy)[1]["decision"] == "ABSTAIN"
        for _ in range(10):
            assert request("/decide", payload)[0] == 200
        sequential = [request("/decide", payload) for _ in range(100)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(lambda _: request("/decide", payload), range(100)))
        for status, body, _ in sequential + concurrent:
            assert status == 200 and body["decision"] == "ADAPT"

        def summarize(rows):
            x = sorted(row[2] * 1000 for row in rows)
            return {"n": len(x), "median_ms": statistics.median(x), "p95_ms": x[95], "p99_ms": x[99]}

        stats = json.loads(run(docker, "stats", "--no-stream", "--format", "{{json .}}", cid))
        uid = run(docker, "exec", cid, "id", "-u")
        assert uid != "0"
        image_source = json.loads(
            run(
                docker,
                "exec",
                cid,
                "python",
                "-c",
                (
                    "import hashlib,json,platform;from pathlib import Path;"
                    "root=Path('/app');paths=sorted([*root.joinpath('kga').rglob('*.py'),"
                    "*root.joinpath('deploy/api').rglob('*.py'),root/'requirements-api-py311-linux.lock.txt']);"
                    "print(json.dumps({'python':platform.python_version(),'files':"
                    "{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}))"
                ),
            )
        )
        report = {
            "status": "PASS",
            "protocol": protocol,
            "image_id": image_id,
            "local_nonroot_uid": uid,
            "image_source": image_source,
            "ready": ready,
            "sequential": summarize(sequential),
            "concurrency_4": summarize(concurrent),
            "docker_stats": stats,
            "checks": {
                "no_key_rejected": True,
                "malformed_rejected": True,
                "proxy_abstains": True,
                "fixture_full_certificate_decides": True,
            },
            "limitations": [
                "one-machine short load test; no availability/SLA claim",
                "certificate API does not execute a neural candidate",
                "new lifecycle module tested separately; not wired into this endpoint",
                "not a public deployment or field validation",
            ],
        }
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        logs = run(docker, "logs", cid)
        (args.output_dir / "container.log").write_text(logs + "\n")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "image_id": image_id,
                    "sequential": report["sequential"],
                    "concurrency_4": report["concurrency_4"],
                },
                indent=2,
            )
        )
    finally:
        subprocess.run([docker, "stop", "--time", "5", cid], capture_output=True, check=False)


if __name__ == "__main__":
    main()
