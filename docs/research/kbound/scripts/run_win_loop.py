#!/usr/bin/env python3
"""End-to-end software loop for finding and validating K-Bound wins.

The loop is intentionally two-stage:
  1. dev/val finder: rank candidate/configs without touching held-out files;
  2. transfer scorer: apply the selected configs to known held-out files.

This script integrates the loop across datasets that already have separate
dev/val and held-out record artifacts.  It writes one machine-readable summary
and one compact markdown report for paper triage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import score_kbound_holdout as sh  # noqa: E402


@dataclass(frozen=True)
class TransferPlan:
    name: str
    cal_record: str
    test_record: str
    label_contains: str
    paper_ready: bool
    note: str


PLANS = [
    TransferPlan(
        name="officehome_protocol_m",
        cal_record="experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json",
        test_record="experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json",
        label_contains="officehome_full_targetval/result_target_val_361a1e8c.json",
        paper_ready=True,
        note="OfficeHome target-val selected config -> target-test score.",
    ),
    TransferPlan(
        name="iwildcam_full_idval_to_test",
        cal_record="experiments/kbound/results/iwildcam_full_idval/result_489da28f.json",
        test_record="experiments/kbound/results/iwildcam_full_test/result_e40faf29.json",
        label_contains="iwildcam_full_idval/result_489da28f.json",
        paper_ready=False,
        note="Diagnostic replication route; Protocol H remains the locked iWildCam headline.",
    ),
    TransferPlan(
        name="fmow_val_to_test",
        cal_record="experiments/kbound/results/fmow_protocol_L_val/result_d8278ebb.json",
        test_record="experiments/kbound/results/fmow_protocol_L_test/result_2c3e265a.json",
        label_contains="fmow_protocol_L_val/result_d8278ebb.json",
        paper_ready=False,
        note="Checks whether Protocol L has a transferable config; previous locked result was null.",
    ),
]


def run_finder(output_dir: Path, top: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "docs/research/kbound/scripts/find_kbound_wins.py"),
        "--top",
        str(top),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return output_dir / "finder_results.json"


def load_finder_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [r for r in data.get("rows", []) if r.get("status") == "ok"]


def candidate_rows_for_plan(rows: list[dict], plan: TransferPlan, limit: int) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        if plan.label_contains not in str(row.get("label", "")):
            continue
        key = (row.get("label"), row.get("candidate"), row.get("estimator"), row.get("conformal"))
        # Labels are path:candidate. Preserve the final candidate token.
        cand = str(row["label"]).rsplit(":", 1)[-1]
        key = (cand, row.get("estimator"), row.get("conformal"))
        if key in seen:
            continue
        seen.add(key)
        rr = dict(row)
        rr["candidate"] = cand
        selected.append(rr)
        if len(selected) >= limit:
            break
    return selected


def score_plan(plan: TransferPlan, rows: list[dict], limit: int, out_root: Path) -> list[dict]:
    scored = []
    for row in candidate_rows_for_plan(rows, plan, limit):
        out_dir = out_root / plan.name / f"{row['candidate']}__{row['estimator']}__{row['conformal']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cal = sh.load_records([plan.cal_record], row["candidate"])
        test = sh.load_records([plan.test_record], row["candidate"])
        if len(cal) < 5 or not test:
            scored.append({
                "dataset": plan.name,
                "status": "not_enough_records",
                "candidate": row["candidate"],
                "estimator": row["estimator"],
                "conformal": row["conformal"],
                "n_cal": len(cal),
                "n_test_records": len(test),
                "paper_ready": plan.paper_ready,
                "note": plan.note,
            })
            continue
        metrics = sh.score_transfer(cal, test, row["estimator"], row["conformal"])
        result = {
            "dataset": plan.name,
            "status": "ok",
            "paper_ready": plan.paper_ready,
            "note": plan.note,
            "cal_record": plan.cal_record,
            "test_record": plan.test_record,
            "candidate": row["candidate"],
            "estimator": row["estimator"],
            "conformal": row["conformal"],
            "dev_candidate_win": bool(row.get("candidate_win")),
            "dev_beats_both": bool(row.get("beats_both")),
            "dev_margin": row.get("margin"),
            "dev_false_adapt": row.get("false_adapt"),
            "dev_coverage": row.get("coverage"),
            "n_cal": len(cal),
            "n_test_records": len(test),
            "cal_seeds": sorted(set(r["seed"] for r in cal)),
            "test_seeds": sorted(set(r["seed"] for r in test)),
            "test_locked": metrics,
        }
        (out_dir / "holdout_score.json").write_text(json.dumps(result, indent=2))
        scored.append(result)
    if not scored:
        scored.append({
            "dataset": plan.name,
            "status": "no_dev_candidate_win",
            "paper_ready": plan.paper_ready,
            "note": plan.note,
            "cal_record": plan.cal_record,
            "test_record": plan.test_record,
        })
    return scored


def row_sort_key(row: dict):
    m = row.get("test_locked") or {}
    return (
        0 if m.get("verdict_win") else 1,
        -float(m.get("margin", -999.0)),
        float(m.get("false_adapt", 999.0)),
        -float(m.get("coverage", 0.0)),
    )


def write_report(results: list[dict], out_md: Path) -> None:
    lines = [
        "# K-Bound Cross-Dataset Win Loop",
        "",
        "Two-stage report: dev/val-selected configs scored on separate held-out artifacts where available.",
        "",
        "| dataset | paper-ready? | dev win? | heldout win? | candidate | est | conf | margin | KGA | adapt | freeze | FA | cov | n |",
        "|---|:---:|:---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=row_sort_key):
        m = r.get("test_locked") or {}
        lines.append(
            f"| {r.get('dataset')} | {'yes' if r.get('paper_ready') else 'diag'} | "
            f"{'YES' if r.get('dev_candidate_win') else ''} | "
            f"{'YES' if m.get('verdict_win') else ''} | `{r.get('candidate', '')}` | "
            f"{r.get('estimator', '')} | {r.get('conformal', '')} | "
            f"{m.get('margin', 0):.6g} | {m.get('regret_kga', 0):.6g} | "
            f"{m.get('regret_adapt', 0):.6g} | {m.get('regret_freeze', 0):.6g} | "
            f"{m.get('false_adapt', 0):.3g} | {m.get('coverage', 0):.3g} | {m.get('n_test', 0)} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="experiments/kbound/results/win_loop_v1")
    p.add_argument("--finder-dir", default="experiments/kbound/results/win_finder_v1")
    p.add_argument("--refresh-finder", action="store_true")
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--top-per-dataset", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    finder_dir = Path(args.finder_dir)
    finder_json = finder_dir / "finder_results.json"
    if args.refresh_finder or not finder_json.exists():
        finder_json = run_finder(finder_dir, args.top)

    rows = load_finder_rows(finder_json)
    results = []
    for plan in PLANS:
        results.extend(score_plan(plan, rows, args.top_per_dataset, out_dir))

    payload = {
        "finder_results": str(finder_json),
        "plans": [plan.__dict__ for plan in PLANS],
        "n_results": len(results),
        "n_wins": sum(1 for r in results if (r.get("test_locked") or {}).get("verdict_win")),
        "results": sorted(results, key=row_sort_key),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "win_loop_results.json").write_text(json.dumps(payload, indent=2))
    write_report(results, out_dir / "win_loop_report.md")

    print(f"finder={finder_json}")
    print(f"wrote {out_dir / 'win_loop_results.json'}")
    print(f"wrote {out_dir / 'win_loop_report.md'}")
    for r in payload["results"]:
        m = r.get("test_locked") or {}
        print(
            f"{r.get('dataset')} win={m.get('verdict_win')} cand={r.get('candidate')} "
            f"{r.get('estimator')}/{r.get('conformal')} margin={m.get('margin')} "
            f"FA={m.get('false_adapt')} cov={m.get('coverage')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
