#!/usr/bin/env python3
"""Fail-closed launcher for the locked K-Bound multi-seed completion matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "research_lock" / "MULTISEED_COMPLETION_PROTOCOL_v1.json"
DEFAULT_RUN_ROOT = ROOT / "experiments" / "kbound" / "runs" / "multiseed_completion_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "kbound_multiseed_completion_protocol_v1":
        raise ValueError(f"unsupported protocol schema in {path}")
    return payload


def resolve_python() -> Path:
    candidates = [
        os.environ.get("KBOUND_PYTHON"),
        str(Path.home() / ".venv_wilds" / "bin" / "python"),
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate).expanduser().absolute()
    raise FileNotFoundError("no Python runtime found; set KBOUND_PYTHON")


def marker_exists(root: Path, marker: str) -> bool:
    if marker.startswith("glob:"):
        return any(root.glob(marker.removeprefix("glob:")))
    return (root / marker).exists()


def resolve_dataset(protocol: dict, key: str) -> Path | None:
    spec = protocol["datasets"][key]
    candidates = []
    override = os.environ.get(spec["environment"])
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(Path(item).expanduser() for item in spec["candidates"])
    for candidate in candidates:
        if candidate.is_dir() and all(marker_exists(candidate, marker) for marker in spec["markers"]):
            return candidate.resolve()
    return None


def resolve_class_index(imagenetr_root: Path | None) -> Path | None:
    override = os.environ.get("KBOUND_IMAGENET_CLASS_INDEX")
    candidates = [Path(override).expanduser()] if override else []
    if imagenetr_root is not None:
        candidates.extend([
            imagenetr_root / "imagenet_class_index.json",
            imagenetr_root.parent / "imagenet_class_index.json",
        ])
    candidates.extend([
        Path("/Volumes/T9/uav/data/imagenet_class_index.json"),
        Path("/Volumes/T9/uav/datasets/imagenet_class_index.json"),
    ])
    return next((path.resolve() for path in candidates if path.is_file()), None)


def check_python(runtime: Path) -> tuple[bool, str]:
    probe = (
        "import json,torch,torchvision,sklearn;"
        "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,"
        "'sklearn':sklearn.__version__,'mps':torch.backends.mps.is_available(),"
        "'cuda':torch.cuda.is_available()}))"
    )
    result = subprocess.run([str(runtime), "-c", probe], text=True, capture_output=True)
    if result.returncode:
        return False, result.stderr.strip() or result.stdout.strip()
    return True, result.stdout.strip()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def preflight(protocol: dict, jobs: list[str], run_root: Path, device: str) -> dict:
    runtime = resolve_python()
    python_ok, python_detail = check_python(runtime)
    dataset_keys = sorted({protocol["jobs"][job]["dataset"] for job in jobs})
    datasets = {key: resolve_dataset(protocol, key) for key in dataset_keys}
    class_index = resolve_class_index(datasets.get("imagenetr"))
    mount_ok = Path("/Volumes/T9").is_dir()
    run_root.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(run_root.parent)
    checks = {
        "t9_mounted": mount_ok,
        "python_research_stack": python_ok,
        "run_disk_free_gib": round(usage.free / (1024**3), 2),
        "run_disk_at_least_10_gib": usage.free >= 10 * 1024**3,
        "datasets": {key: str(value) if value else None for key, value in datasets.items()},
        "imagenet_class_index": str(class_index) if class_index else None,
    }
    try:
        runtime_info = json.loads(python_detail) if python_ok else {}
    except json.JSONDecodeError:
        runtime_info = {}
    available = {
        "mps": bool(runtime_info.get("mps")),
        "cuda": bool(runtime_info.get("cuda")),
    }
    accelerator_ok = device == "cpu" or (
        available[device] if device in available else any(available.values())
    )
    checks["requested_device"] = device
    checks["accelerator_available"] = available
    checks["device_ready"] = accelerator_ok
    failures = []
    if not mount_ok:
        failures.append("T9 is not mounted at /Volumes/T9")
    if not python_ok:
        failures.append(f"research Python probe failed: {python_detail}")
    elif not accelerator_ok:
        failures.append(
            f"requested device {device!r} is unavailable; pass --device cpu only for an intentional CPU run"
        )
    if not checks["run_disk_at_least_10_gib"]:
        failures.append("less than 10 GiB free for run outputs/checkpoints")
    for key, value in datasets.items():
        if value is None:
            failures.append(f"dataset {key} not found; set {protocol['datasets'][key]['environment']}")
    if "imagenetr" in dataset_keys and class_index is None:
        failures.append("ImageNet class-index JSON not found; set KBOUND_IMAGENET_CLASS_INDEX")
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "python": str(runtime),
        "python_detail": python_detail,
        "datasets_resolved": datasets,
        "class_index_resolved": class_index,
    }


def selected_jobs(protocol: dict, requested: list[str]) -> list[str]:
    jobs = []
    for value in requested:
        jobs.extend(item.strip() for item in value.split(",") if item.strip())
    jobs = jobs or list(protocol["default_jobs"])
    unknown = sorted(set(jobs) - set(protocol["jobs"]))
    if unknown:
        raise ValueError(f"unknown jobs: {', '.join(unknown)}")
    return jobs


def format_arguments(
    job_name: str,
    job: dict,
    seed: int,
    run_root: Path,
    data_root: Path,
    class_index: Path | None,
    device: str,
) -> list[str]:
    job_dir = run_root / job_name
    seed_dir = job_dir / f"seed{seed}" if job["output_layout"] == "per_seed" else job_dir
    values = {
        "data": str(data_root),
        "seed": str(seed),
        "seed_dir": str(seed_dir),
        "job_dir": str(job_dir),
        "job_parent": str(job_dir.parent),
        "job_name": job_dir.name,
        "device": device,
        "class_index": str(class_index) if class_index else "<missing-class-index>",
    }
    return [item.format(**values) for item in job["arguments"]]


def expected_paths(run_root: Path, job_name: str, job: dict, seed: int) -> list[Path]:
    job_dir = run_root / job_name
    base = job_dir / f"seed{seed}" if job["output_layout"] == "per_seed" else job_dir
    expected = job["expected"]
    names = [expected] if isinstance(expected, str) else expected
    return [base / name.format(seed=seed) for name in names]


def bootstrap_job(protocol: dict, job_name: str, run_root: Path) -> list[Path]:
    job = protocol["jobs"][job_name]
    spec = job.get("bootstrap")
    if not spec:
        return []
    source = ROOT / spec["source"]
    if not source.is_dir():
        raise FileNotFoundError(f"bootstrap source missing: {source}")
    copied = []
    for seed in spec["seeds"]:
        if job["methods"]:
            matches = [
                source / f"per_condition_{job['dataset_name']}_{method}_seed{seed}.json"
                for method in job["methods"]
            ]
            matches = [path for path in matches if path.is_file()]
        else:
            matches = sorted(source.glob(f"per_condition_{job['dataset_name']}_*_seed{seed}.json"))
        if not matches:
            raise FileNotFoundError(f"no bootstrap seed {seed} records under {source}")
        target_dir = run_root / job_name
        if job["output_layout"] == "per_seed":
            target_dir = target_dir / f"seed{seed}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in matches:
            target = target_dir / path.name
            if target.exists() and sha256(target) != sha256(path):
                raise RuntimeError(f"immutable bootstrap mismatch: {target}")
            if not target.exists():
                shutil.copy2(path, target)
            copied.append(target)
    return copied


def write_job_lock(protocol_path: Path, job_name: str, job_dir: Path, commands: list[list[str]]) -> None:
    lock = {
        "schema": "kbound_run_lock_v1",
        "job": job_name,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256(protocol_path),
        "git_head": git_head(),
        "commands": commands,
    }
    path = job_dir / "RUN_LOCK.json"
    encoded = json.dumps(lock, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"run lock mismatch; use a new immutable run directory: {path}")
    job_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def run_command(command: list[str], log_path: Path) -> None:
    prefix = ["caffeinate", "-is"] if shutil.which("caffeinate") else []
    full = prefix + command
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {json.dumps(full)}\n")
        log.flush()
        process = subprocess.Popen(
            full,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END rc={code}\n")
    if code:
        raise subprocess.CalledProcessError(code, full)


def command_matrix(
    protocol: dict,
    jobs: list[str],
    run_root: Path,
    datasets: dict[str, Path | None],
    class_index: Path | None,
    runtime: Path,
    device: str,
) -> dict[str, list[tuple[int, list[str]]]]:
    matrix = {}
    for job_name in jobs:
        job = protocol["jobs"][job_name]
        data = datasets.get(job["dataset"])
        if data is None:
            continue
        rows = []
        for seed in job["run_seeds"]:
            arguments = format_arguments(job_name, job, seed, run_root, data, class_index, device)
            rows.append((seed, [str(runtime), str(ROOT / job["runner"]), *arguments]))
        matrix[job_name] = rows
    return matrix


def print_status(protocol: dict, jobs: list[str], run_root: Path) -> int:
    missing = 0
    for job_name in jobs:
        job = protocol["jobs"][job_name]
        print(f"{job_name}:")
        for seed in job["seeds"]:
            paths = expected_paths(run_root, job_name, job, seed)
            bootstrap_seeds = set(job.get("bootstrap", {}).get("seeds", []))
            if all(path.is_file() for path in paths):
                state = "complete"
            elif seed in bootstrap_seeds:
                state = "bootstrap-ready"
            else:
                state = "pending"
                missing += 1
            print(f"  seed {seed}: {state}  {', '.join(str(path) for path in paths)}")
    return int(missing > 0)


def analyze(protocol: dict, jobs: list[str], run_root: Path, runtime: Path, nboot: int) -> None:
    scorer = ROOT / "experiments" / "kbound" / "training" / "uniform_multiseed.py"
    for job_name in jobs:
        bootstrap_job(protocol, job_name, run_root)
        job = protocol["jobs"][job_name]
        job_dir = run_root / job_name
        command = [
            str(runtime), str(scorer),
            "--run-dir", str(job_dir),
            "--dataset", job["dataset_name"],
            "--seeds", *[str(seed) for seed in job["seeds"]],
            "--alpha", str(protocol["alpha"]),
            "--nboot", str(nboot),
        ]
        if job["methods"]:
            command.extend(["--methods", *job["methods"]])
        run_command(command, job_dir / "analyze.log")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "plan", "run", "status", "analyze"])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--jobs", nargs="*", default=[])
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=os.environ.get("KBOUND_DEVICE", "auto"))
    parser.add_argument("--nboot", type=int, default=10000)
    parser.add_argument("--yes", action="store_true", help="confirm execution of the locked long-run queue")
    args = parser.parse_args(argv)

    protocol_path = args.protocol.expanduser().resolve()
    protocol = load_protocol(protocol_path)
    jobs = selected_jobs(protocol, args.jobs)
    run_root = args.run_root.resolve()
    if ROOT not in run_root.parents:
        raise SystemExit("run root must remain inside the clean repository")
    if args.action == "status":
        return print_status(protocol, jobs, run_root)

    report = preflight(protocol, jobs, run_root, args.device)
    public_report = {
        "passed": report["passed"],
        "failures": report["failures"],
        "checks": report["checks"],
        "python": report["python"],
        "python_detail": report["python_detail"],
    }
    if args.action == "preflight":
        print(json.dumps(public_report, indent=2))
        return int(not report["passed"])

    planning_datasets = dict(report["datasets_resolved"])
    if args.action == "plan":
        for key, value in planning_datasets.items():
            if value is None:
                planning_datasets[key] = Path(protocol["datasets"][key]["candidates"][0])
    planning_class_index = report["class_index_resolved"]
    if args.action == "plan" and planning_class_index is None:
        planning_class_index = Path("/Volumes/T9/uav/data/imagenet_class_index.json")
    matrix = command_matrix(
        protocol,
        jobs,
        run_root,
        planning_datasets,
        planning_class_index,
        Path(report["python"]),
        args.device,
    )
    if args.action == "plan":
        print(json.dumps(public_report, indent=2))
        for job_name in jobs:
            for seed, command in matrix.get(job_name, []):
                print(f"{job_name}/seed{seed}: {json.dumps(command)}")
        return 0
    if not report["passed"]:
        print(json.dumps(public_report, indent=2))
        return 2
    if args.action == "analyze":
        analyze(protocol, jobs, run_root, Path(report["python"]), args.nboot)
        return 0
    if not args.yes:
        raise SystemExit("refusing to start long runs without --yes")

    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".accelerator.lock"
    with lock_path.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another K-Bound accelerator queue holds the run lock") from exc
        for job_name in jobs:
            bootstrap_job(protocol, job_name, run_root)
            rows = matrix[job_name]
            job_dir = run_root / job_name
            write_job_lock(protocol_path, job_name, job_dir, [command for _, command in rows])
            job = protocol["jobs"][job_name]
            for seed, command in rows:
                expected = expected_paths(run_root, job_name, job, seed)
                if all(path.is_file() for path in expected):
                    print(f"[skip] {job_name}/seed{seed}: all expected artifacts exist")
                    continue
                run_command(command, job_dir / f"seed{seed}.log")
    print_status(protocol, jobs, run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
