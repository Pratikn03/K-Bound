#!/usr/bin/env python3
"""Evaluate and save the fail-closed physical-study publication gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE = HERE.parent
SRC = EDGE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import _common as C
from kbound_edge.publication import evaluate_publication_gate


def read(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="edge_real_phone_v1.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    clean_cfg = C.clean_config(cfg)
    results = Path(C.resolve(cfg["paths"]["results_dir"]))
    report = evaluate_publication_gate(
        clean_cfg,
        model_card=read(results / "model_card.json"),
        split_audit=read(results / "split_audit.json"),
        inventory=read(results / "recording_inventory.json"),
        heldout=read(results / "heldout_metrics.json"),
        replication=read(results / "replication_metrics.json"),
        anti_leakage=read(results / "anti_leakage_audit.json"),
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    out = results / "publication_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status}] {row['check']}: {row['observed']}")
    print(f"Publication gate: {'PASS' if report['passed'] else 'FAIL'} -> {out}")
    return 0 if report["passed"] or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())

