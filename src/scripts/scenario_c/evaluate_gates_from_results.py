#!/usr/bin/env python3
"""Evaluate Gates B/D from frozen baselines + master_c result JSONs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "research_lock").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _mean_roc(payload: dict, method: str) -> float | None:
    rows = payload.get("clean_metric_summary") or payload.get("table_1_clean_performance")
    if isinstance(rows, dict):
        m = rows.get(method) or {}
        if isinstance(m, dict) and "mean" in m.get("roc_auc", {}):
            return float(m["roc_auc"]["mean"])
        return float(m.get("roc_auc", float("nan"))) if isinstance(m.get("roc_auc"), (int, float)) else None
    if isinstance(rows, list):
        vals = []
        for r in rows:
            block = r.get(method) or {}
            v = block.get("roc_auc")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--protocol", type=str, required=True)
    args = parser.parse_args()

    root = _repo_root()
    frozen = json.loads((root / "research_lock/strongest_baseline_frozen_v1.json").read_text(encoding="utf-8"))
    comp = None
    for cell in frozen.get("cells", []):
        if cell["benchmark"] == args.benchmark and cell["protocol"] == args.protocol:
            comp = cell["strongest_baseline"]
            comp_val = float(cell["validation_roc_auc"])
            break
    if comp is None:
        print("No frozen comparator for cell")
        return 2

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rga = _mean_roc(payload, "rga_boosted_fusion")
    static = _mean_roc(payload, "static_attention")
    base = _mean_roc(payload, comp)
    report = {
        "benchmark": args.benchmark,
        "protocol": args.protocol,
        "frozen_comparator": comp,
        "frozen_comparator_validation_auc": comp_val,
        "test_roc_auc": {
            "rga_boosted_fusion": rga,
            "static_attention": static,
            comp: base,
        },
        "gate_b_fusion_trained": static is not None and rga is not None,
        "gate_d_rga_beats_frozen_test": (rga is not None and base is not None and rga > base),
        "gate_d_rga_beats_static_test": (rga is not None and static is not None and rga > static),
    }
    out = root / "elara_master_c/audits/gate_bd_evaluation.json"
    existing = {}
    if out.is_file():
        existing = json.loads(out.read_text(encoding="utf-8"))
    existing[f"{args.benchmark}|{args.protocol}"] = report
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["gate_d_rga_beats_frozen_test"] else 1


if __name__ == "__main__":
    sys.exit(main())
