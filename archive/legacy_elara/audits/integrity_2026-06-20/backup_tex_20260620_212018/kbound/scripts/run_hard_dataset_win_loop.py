#!/usr/bin/env python3
"""Seed-heldout win loop for ImageNet-R and RxRx1.

This is a CPU-only audit loop over existing record artifacts.  It does not train
models; it reruns the locked analyze_F decision evaluator on pre-specified
seed/modelseed splits and writes a reproducible report.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # noqa: E402


@dataclass(frozen=True)
class SplitPlan:
    name: str
    dataset: str
    records: tuple[str, ...]
    candidate: str
    estimator: str
    conformal: str
    dev_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    stability_group: str
    note: str


DEFAULT_PLANS = [
    SplitPlan(
        name="imagenetr_light_split01_23",
        dataset="imagenet-r",
        records=("experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json",),
        candidate="tent_online",
        estimator="gbr",
        conformal="cqr",
        dev_seeds=(0, 1),
        test_seeds=(2, 3),
        stability_group="imagenetr_light_tent_gbr_cqr",
        note="Original light ImageNet-R split lead.",
    ),
    SplitPlan(
        name="imagenetr_light_split02_13",
        dataset="imagenet-r",
        records=("experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json",),
        candidate="tent_online",
        estimator="gbr",
        conformal="cqr",
        dev_seeds=(0, 2),
        test_seeds=(1, 3),
        stability_group="imagenetr_light_tent_gbr_cqr",
        note="Alternate split stability check.",
    ),
    SplitPlan(
        name="imagenetr_light_split03_12",
        dataset="imagenet-r",
        records=("experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json",),
        candidate="tent_online",
        estimator="gbr",
        conformal="cqr",
        dev_seeds=(0, 3),
        test_seeds=(1, 2),
        stability_group="imagenetr_light_tent_gbr_cqr",
        note="Alternate split stability check.",
    ),
    SplitPlan(
        name="imagenetr_light_split12_03",
        dataset="imagenet-r",
        records=("experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json",),
        candidate="tent_online",
        estimator="gbr",
        conformal="cqr",
        dev_seeds=(1, 2),
        test_seeds=(0, 3),
        stability_group="imagenetr_light_tent_gbr_cqr",
        note="Alternate split stability check.",
    ),
    SplitPlan(
        name="rxrx1_modelseed0_tent_mondrian",
        dataset="rxrx1",
        records=("experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json",),
        candidate="tent_online",
        estimator="gbr",
        conformal="mondrian",
        dev_seeds=(0, 1, 2, 3, 4),
        test_seeds=(5, 6, 7, 8, 9),
        stability_group="rxrx1_tent_gbr_mondrian",
        note="Modelseed-0 RxRx1 lead.",
    ),
    SplitPlan(
        name="rxrx1_modelseed1_tent_mondrian",
        dataset="rxrx1",
        records=("experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed1/result_eef46aea.json",),
        candidate="tent_online",
        estimator="gbr",
        conformal="mondrian",
        dev_seeds=(0, 1, 2, 3, 4),
        test_seeds=(5, 6, 7, 8, 9),
        stability_group="rxrx1_tent_gbr_mondrian",
        note="Modelseed-1 replication check.",
    ),
    SplitPlan(
        name="rxrx1_modelseed2_tent_mondrian",
        dataset="rxrx1",
        records=("experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json",),
        candidate="tent_online",
        estimator="gbr",
        conformal="mondrian",
        dev_seeds=(0, 1, 2, 3, 4),
        test_seeds=(5, 6, 7, 8, 9),
        stability_group="rxrx1_tent_gbr_mondrian",
        note="Modelseed-2 replication check.",
    ),
    SplitPlan(
        name="rxrx1_pooled_modelseeds0_2_tent_mondrian",
        dataset="rxrx1",
        records=(
            "experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json",
            "experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed1/result_eef46aea.json",
            "experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json",
        ),
        candidate="tent_online",
        estimator="gbr",
        conformal="mondrian",
        dev_seeds=(0, 1, 2, 3, 4),
        test_seeds=(5, 6, 7, 8, 9),
        stability_group="rxrx1_tent_gbr_mondrian",
        note="Pooled modelseed check; this is the stricter replication summary.",
    ),
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def verdict_from_metrics(metrics: dict | None) -> dict:
    if not metrics:
        return {
            "beats_both": False,
            "fa_ok": False,
            "verdict_win": False,
            "margin": None,
        }
    regret_kga = _safe_float(metrics.get("regret_kga"))
    regret_adapt = _safe_float(metrics.get("regret_adapt"))
    regret_freeze = _safe_float(metrics.get("regret_freeze"))
    false_adapt = _safe_float(metrics.get("false_adapt"), default=1.0)
    beats_both = regret_kga < regret_adapt and regret_kga < regret_freeze
    fa_ok = false_adapt <= af.ALPHA
    return {
        "beats_both": bool(beats_both),
        "fa_ok": bool(fa_ok),
        "verdict_win": bool(beats_both and fa_ok),
        "margin": float(min(regret_adapt, regret_freeze) - regret_kga),
    }


def load_records(paths: Iterable[str], candidate: str) -> tuple[list[dict], str]:
    records: list[dict] = []
    panel = "unknown"
    for raw_path in paths:
        path = ROOT / raw_path
        part, panel = af.load_records(str(path), candidate=candidate)
        records.extend(part)
    return records, panel


def run_plan(plan: SplitPlan, out_root: Path) -> dict:
    missing = [p for p in plan.records if not (ROOT / p).is_file()]
    run_dir = out_root / plan.name
    run_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        result = {
            "name": plan.name,
            "dataset": plan.dataset,
            "status": "missing_records",
            "missing_records": missing,
            "candidate": plan.candidate,
            "estimator": plan.estimator,
            "conformal": plan.conformal,
            "dev_seeds": list(plan.dev_seeds),
            "test_seeds": list(plan.test_seeds),
            "stability_group": plan.stability_group,
            "note": plan.note,
        }
        (run_dir / "analyze_F_results.json").write_text(json.dumps(result, indent=2))
        return result

    records, panel = load_records(plan.records, plan.candidate)
    metrics = af.run_split(
        records,
        list(plan.dev_seeds),
        list(plan.test_seeds),
        estimator=plan.estimator,
        conformal=plan.conformal,
    )
    baseline = af.run_split(
        records,
        list(plan.dev_seeds),
        list(plan.test_seeds),
        estimator="gbr",
        conformal="global",
    )
    verdict = verdict_from_metrics(metrics)
    result = {
        "name": plan.name,
        "dataset": plan.dataset,
        "status": "ok" if metrics else "no_metrics",
        "alpha": af.ALPHA,
        "evidence_panel": panel,
        "records": list(plan.records),
        "candidate": plan.candidate,
        "estimator": plan.estimator,
        "conformal": plan.conformal,
        "dev_seeds": list(plan.dev_seeds),
        "test_seeds": list(plan.test_seeds),
        "stability_group": plan.stability_group,
        "note": plan.note,
        "n_records": len(records),
        "seeds_present": sorted(set(int(r["seed"]) for r in records)),
        "Z_dim": len(records[0]["Z"]) if records else None,
        "test_locked": metrics,
        "test_baseline_gbr_global": baseline,
        **verdict,
    }
    (run_dir / "analyze_F_results.json").write_text(json.dumps(result, indent=2))
    return result


def summarize_stability(results: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(str(result.get("stability_group", "ungrouped")), []).append(result)

    summary: dict[str, dict] = {}
    for group, rows in grouped.items():
        wins = 0
        ok_rows = 0
        margins = []
        for row in rows:
            if row.get("status", "ok") not in {"ok", None}:
                continue
            ok_rows += 1
            metrics = row.get("test_locked")
            verdict = {
                "verdict_win": bool(row.get("verdict_win")),
                "margin": row.get("margin"),
            }
            if "verdict_win" not in row and metrics:
                verdict = verdict_from_metrics(metrics)
            if verdict.get("verdict_win"):
                wins += 1
            if verdict.get("margin") is not None:
                margins.append(float(verdict["margin"]))
        total = len(rows)
        summary[group] = {
            "total": total,
            "ok": ok_rows,
            "wins": wins,
            "win_rate": float(wins / ok_rows) if ok_rows else 0.0,
            "replicated_win": bool(ok_rows > 1 and wins == ok_rows),
            "any_win": bool(wins > 0),
            "min_margin": min(margins) if margins else None,
            "max_margin": max(margins) if margins else None,
        }
    return summary


def sort_results(row: dict) -> tuple:
    return (
        str(row.get("dataset", "")),
        str(row.get("stability_group", "")),
        0 if row.get("verdict_win") else 1,
        -_safe_float(row.get("margin"), -999.0),
        str(row.get("name", "")),
    )


def write_report(results: list[dict], stability: dict[str, dict], out_md: Path) -> None:
    replicated = [name for name, item in stability.items() if item.get("replicated_win")]
    any_wins = [name for name, item in stability.items() if item.get("any_win")]
    lines = [
        "# ImageNet-R / RxRx1 Hard-Dataset Win Loop",
        "",
        "CPU-only audit over existing record artifacts. These rows rerun the locked K-Bound decision evaluator on fixed seed splits; no model training is performed.",
        "",
        f"- Stability groups with replicated wins: {len(replicated)}",
        f"- Stability groups with at least one split/modelseed win: {len(any_wins)}",
        "- Interpretation rule: split-only/modelseed-only wins are leads, not paper headline claims, until they replicate across the planned stability checks.",
        "",
        "## Stability Summary",
        "",
        "| group | rows | wins | win rate | replicated? | margin range |",
        "|---|---:|---:|---:|:---:|---:|",
    ]
    for group, item in sorted(stability.items()):
        mn = item.get("min_margin")
        mx = item.get("max_margin")
        margin_range = "" if mn is None else f"{mn:.6g}..{mx:.6g}"
        lines.append(
            f"| `{group}` | {item.get('ok', 0)}/{item.get('total', 0)} | "
            f"{item.get('wins', 0)} | {item.get('win_rate', 0):.3g} | "
            f"{'YES' if item.get('replicated_win') else ''} | {margin_range} |"
        )
    lines.extend([
        "",
        "## Per-Run Results",
        "",
        "| dataset | run | win? | candidate | est | conf | split | margin | KGA | adapt | freeze | FA | cov | n |",
        "|---|---|:---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(results, key=sort_results):
        metrics = row.get("test_locked") or {}
        split = f"{row.get('dev_seeds')}->{row.get('test_seeds')}"
        lines.append(
            f"| {row.get('dataset')} | `{row.get('name')}` | "
            f"{'YES' if row.get('verdict_win') else ''} | `{row.get('candidate', '')}` | "
            f"{row.get('estimator', '')} | {row.get('conformal', '')} | `{split}` | "
            f"{_safe_float(row.get('margin'), 0):.6g} | "
            f"{_safe_float(metrics.get('regret_kga'), 0):.6g} | "
            f"{_safe_float(metrics.get('regret_adapt'), 0):.6g} | "
            f"{_safe_float(metrics.get('regret_freeze'), 0):.6g} | "
            f"{_safe_float(metrics.get('false_adapt'), 0):.3g} | "
            f"{_safe_float(metrics.get('coverage'), 0):.3g} | "
            f"{int(metrics.get('n_test', 0) or 0)} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def filter_plans(only: list[str] | None) -> list[SplitPlan]:
    plans = list(DEFAULT_PLANS)
    if not only:
        return plans
    wanted = set(only)
    return [
        plan
        for plan in plans
        if plan.name in wanted or plan.dataset in wanted or plan.stability_group in wanted
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/kbound/results/hard_dataset_win_loop_v1")
    parser.add_argument("--only", nargs="*", default=None, help="Optional plan, dataset, or stability-group filters")
    parser.add_argument("--list-plans", action="store_true")
    args = parser.parse_args()

    plans = filter_plans(args.only)
    if args.list_plans:
        for plan in plans:
            print(f"{plan.name}\t{plan.dataset}\t{plan.candidate}\t{plan.estimator}/{plan.conformal}")
        return 0
    if not plans:
        print("ERROR: no plans matched --only", file=sys.stderr)
        return 2

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_plan(plan, out_dir) for plan in plans]
    stability = summarize_stability(results)
    payload = {
        "alpha": af.ALPHA,
        "mode": "seed_heldout_hard_dataset_loop",
        "plans": [plan.__dict__ for plan in plans],
        "n_runs": len(results),
        "n_wins": sum(1 for row in results if row.get("verdict_win")),
        "stability": stability,
        "results": sorted(results, key=sort_results),
    }
    out_json = out_dir / "hard_dataset_loop_results.json"
    out_md = out_dir / "hard_dataset_loop_report.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_report(results, stability, out_md)

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    for group, item in sorted(stability.items()):
        print(
            f"group={group} wins={item['wins']}/{item['ok']} "
            f"replicated={item['replicated_win']} margin={item['min_margin']}..{item['max_margin']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
