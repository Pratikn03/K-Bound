"""Phase 3 — calibration-transfer closure orchestrator.

This runner executes the pre-registered held-out calibration-transfer
evaluation using the existing Family-D execution stack.

Default mode is `dry-run` to avoid accidental held-out test execution.
Use `--full-run` to authorize one-time test evaluation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PHASE2_V3_MANIFEST = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_PARTITION_MANIFEST_v3.json"


def _load_protocol(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Protocol YAML not found: {path}")
    data = yaml.safe_load(path.read_text())
    if "protocol" not in data:
        raise SystemExit("Protocol YAML missing top-level 'protocol' key")
    return data["protocol"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PRECHECK FAIL: {message}")


def _validate_freeze(protocol: dict[str, Any]) -> None:
    calib = protocol.get("calibration_provenance", {})
    ks = calib.get("ks_reference_fit", {})
    th = calib.get("threshold_selection", {})

    _require(ks.get("uses_test_scores") is False, "ks_reference_fit.uses_test_scores must be false")
    _require(ks.get("uses_test_labels") is False, "ks_reference_fit.uses_test_labels must be false")
    _require(
        ks.get("frozen_before_test_metric_read") is True,
        "ks_reference_fit.frozen_before_test_metric_read must be true",
    )
    _require(th.get("uses_test_scores") is False, "threshold_selection.uses_test_scores must be false")
    _require(th.get("uses_test_labels") is False, "threshold_selection.uses_test_labels must be false")
    _require(
        th.get("frozen_before_test_metric_read") is True,
        "threshold_selection.frozen_before_test_metric_read must be true",
    )
    _require(float(th.get("clean_false_fire_budget", -1.0)) == 0.010, "clean false-fire budget must be 0.010")


def _resolve_exec_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    exec_cfg = protocol.get("execution", {})
    keys = [
        "family_d_protocol_yaml",
        "family_d_pipeline_yaml",
        "family_d_cell_runner",
        "family_d_inference_runner",
    ]
    resolved: dict[str, Path] = {}
    for key in keys:
        rel = exec_cfg.get(key)
        _require(bool(rel), f"missing execution.{key}")
        path = ROOT / str(rel)
        _require(path.exists(), f"execution path not found: {path}")
        resolved[key] = path
    return resolved


def _write_manifest(
    manifest_path: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    resolved: dict[str, Path],
    full_run: bool,
    seeds: int,
    seed_start: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": protocol.get("id"),
        "title": protocol.get("title"),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_mode": "full_run" if full_run else "dry_run",
        "seed_count": seeds,
        "seed_start": seed_start,
        "protocol_yaml": str(protocol_path.relative_to(ROOT)),
        "protocol_yaml_sha256": _sha256(protocol_path),
        "family_d_protocol_yaml_sha256": _sha256(resolved["family_d_protocol_yaml"]),
        "family_d_pipeline_yaml_sha256": _sha256(resolved["family_d_pipeline_yaml"]),
        "calibration_provenance": protocol.get("calibration_provenance", {}),
        "clean_false_fire": protocol.get("clean_false_fire", {}),
        "stress_endpoints": protocol.get("stress_endpoints", {}),
        "selection_used_test_metrics_required": False,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_cmd(cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print(f"[RUN] {printable}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _load_phase2_v3_manifest() -> dict[str, Any]:
    if not PHASE2_V3_MANIFEST.exists():
        return {}
    try:
        return json.loads(PHASE2_V3_MANIFEST.read_text())
    except Exception:
        return {}


def _cell_executed_in_manifest(manifest: dict[str, Any], cell_id: str) -> bool:
    key = f"cell_{cell_id.lower().replace('-', '_')}_executed"
    return bool(manifest.get(key, False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="configs/phase3/calibration_transfer_closure_protocol.yaml",
        help="Phase-3 closure protocol YAML",
    )
    parser.add_argument("--seeds", type=int, default=30, help="Seed count")
    parser.add_argument("--seed-start", type=int, default=42, help="Start seed")
    parser.add_argument("--full-run", action="store_true", help="Run held-out test evaluation")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference after cell runs")
    args = parser.parse_args()

    protocol_path = ROOT / args.protocol
    protocol = _load_protocol(protocol_path)
    _validate_freeze(protocol)
    resolved = _resolve_exec_paths(protocol)

    manifest_rel = protocol.get("audit_outputs", {}).get(
        "manifest_json",
        "docs/research/phase3/CALIBRATION_TRANSFER_CLOSURE_MANIFEST.json",
    )
    manifest_path = ROOT / manifest_rel
    _write_manifest(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        protocol=protocol,
        resolved=resolved,
        full_run=args.full_run,
        seeds=int(args.seeds),
        seed_start=int(args.seed_start),
    )
    print(f"[OK] Wrote manifest: {manifest_path}", flush=True)

    python_bin = sys.executable
    dry_flag = [] if args.full_run else ["--dry-run"]
    cell_ids = list(protocol.get("execution", {}).get("cell_ids_primary", ["D-EYE-1", "D-EYE-2"]))
    phase2_manifest = _load_phase2_v3_manifest()
    skipped_cells: list[str] = []
    executed_cells: list[str] = []

    for cell in cell_ids:
        if _cell_executed_in_manifest(phase2_manifest, cell):
            print(
                f"[SKIP] {cell} already marked executed in phase2 v3 manifest; "
                "not re-running one-time cell.",
                flush=True,
            )
            skipped_cells.append(cell)
            continue
        cmd = [
            python_bin,
            str(resolved["family_d_cell_runner"]),
            "--cell",
            cell,
            "--seeds",
            str(int(args.seeds)),
            "--seed-start",
            str(int(args.seed_start)),
            "--protocol",
            str(resolved["family_d_protocol_yaml"].relative_to(ROOT)),
            "--pipeline-spec",
            str(resolved["family_d_pipeline_yaml"].relative_to(ROOT)),
        ] + dry_flag
        _run_cmd(cmd)
        executed_cells.append(cell)

    if not args.skip_inference and args.full_run:
        cmd = [python_bin, str(resolved["family_d_inference_runner"])]
        _run_cmd(cmd)

    if skipped_cells and not executed_cells:
        print("[INFO] No cell runs executed in this invocation (all already completed).", flush=True)

    print("[DONE] Phase-3 calibration-transfer closure orchestration completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())