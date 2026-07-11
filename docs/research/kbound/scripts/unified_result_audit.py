#!/usr/bin/env python3
"""Uniform K-Bound result audit.

Dataset runners can remain dataset-specific, but the final result verdict must
not be dataset-specific.  This module defines the common result semantics used
by paper tables, raw run summaries, and reviewer audits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]


DIRECT_REGRET_KEYS = {
    "kga": ("regret_kga", "kga_mean_regret", "regret_KGA"),
    "adapt": ("regret_adapt", "adapt_mean_regret", "regret_always_adapt"),
    "freeze": ("regret_freeze", "freeze_mean_regret", "regret_always_freeze"),
}

NESTED_REGRET_KEYS = {
    "kga": ("K_Bound", "KGA", "kga", "kga_routed"),
    "adapt": ("always_adapt", "adapt", "always-adapt"),
    "freeze": ("always_freeze", "freeze", "always-freeze"),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def extract_regrets(row: dict[str, Any]) -> tuple[float, float, float]:
    """Return `(regret_kga, regret_adapt, regret_freeze)` from known result schemas."""
    out: dict[str, float] = {}
    for target, keys in DIRECT_REGRET_KEYS.items():
        for key in keys:
            if _is_number(row.get(key)):
                out[target] = float(row[key])
                break

    for nested_key in ("regret_vs_oracle", "regret"):
        nested = row.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for target, keys in NESTED_REGRET_KEYS.items():
            if target in out:
                continue
            for key in keys:
                if _is_number(nested.get(key)):
                    out[target] = float(nested[key])
                    break

    missing = {"kga", "adapt", "freeze"} - set(out)
    if missing:
        raise KeyError(f"missing regret fields: {sorted(missing)}")
    return out["kga"], out["adapt"], out["freeze"]


def point_beats_both(regret_kga: float, regret_adapt: float, regret_freeze: float) -> bool:
    """Strict run-only beats-both definition. Lower regret is better."""
    return regret_kga < regret_adapt and regret_kga < regret_freeze


def ci_robust_beats_both_from_comparisons(comparisons: list[dict[str, Any]], candidate: str) -> bool:
    """Return true only if both always-adapt and always-freeze comparisons survive."""
    needed = {"always-adapt", "always-freeze"}
    passed: set[str] = set()
    for comp in comparisons:
        if comp.get("candidate") != candidate:
            continue
        trivial = str(comp.get("trivial", ""))
        if trivial in needed and bool(comp.get("survives_holm")):
            passed.add(trivial)
    return passed == needed


def ci_robust_from_row(row: dict[str, Any]) -> str:
    """Return `True`, `False`, or `unknown` as a string for stable JSON/Markdown output."""
    if "ci_robust_beats_both" in row:
        value = row["ci_robust_beats_both"]
        return "unknown" if value is None else str(bool(value))
    verdict = str(row.get("verdict", "")).lower()
    if "ci-robust" in verdict or "ci robust" in verdict:
        return "True"
    if "beats_both_robust" in row:
        return str(bool(row["beats_both_robust"]))
    for key in ("beats_both_CI", "beats_both_ci"):
        if key in row:
            return str(bool(row[key]))

    # Paper-source gap convention is baseline regret minus KGA regret.
    ci_adapt = row.get("ci_vs_adapt")
    ci_freeze = row.get("ci_vs_freeze")
    if isinstance(ci_adapt, list) and isinstance(ci_freeze, list) and ci_adapt and ci_freeze:
        if _is_number(ci_adapt[0]) and _is_number(ci_freeze[0]):
            return str(float(ci_adapt[0]) > 0.0 and float(ci_freeze[0]) > 0.0)
    return "unknown"


def false_adapt_value(row: dict[str, Any]) -> str:
    for key in ("FA_u", "false_adapt", "false_adapt_rate_pooled", "false_adapt_rate_B<0"):
        if key in row:
            return str(row[key])
    return ""


def explicit_beats_both_flags(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "beats_both",
        "KGA_beats_both",
        "verdict_win",
        "heldout_verdict_win",
        "beats_both_robust",
        "beats_both_CI",
        "beats_both_ci",
        "point_beats_both",
        "ci_robust_beats_both",
    )
    return {key: row[key] for key in keys if key in row}


def summarize_result_row(
    *,
    dataset: str,
    artifact: str,
    location: str,
    row: dict[str, Any],
    candidate: str = "",
    source_level: str = "run",
    ci_robust: str | None = None,
) -> dict[str, Any]:
    kga, adapt, freeze = extract_regrets(row)
    explicit = explicit_beats_both_flags(row)
    point = point_beats_both(kga, adapt, freeze)
    explicit_point_flags = {
        key: value
        for key, value in explicit.items()
        if key in {"beats_both", "KGA_beats_both", "point_beats_both"}
    }
    return {
        "dataset": dataset,
        "source_level": source_level,
        "candidate": candidate or str(row.get("candidate") or row.get("method") or ""),
        "artifact": artifact,
        "location": location,
        "n": row.get("n_test") or row.get("n_conditions") or row.get("n_records") or "",
        "regret_kga": kga,
        "regret_adapt": adapt,
        "regret_freeze": freeze,
        "point_beats_both": point,
        "ci_robust_beats_both": ci_robust if ci_robust is not None else ci_robust_from_row(row),
        "false_adapt": false_adapt_value(row),
        "explicit_flags": explicit,
        "explicit_point_consistent": (
            all(bool(value) == point for value in explicit_point_flags.values())
            if explicit_point_flags else "not_present"
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return data


def _lookup(data: dict[str, Any], location: str) -> dict[str, Any]:
    if location == "$":
        return data
    cur: Any = data
    for part in location.strip("$.").split("."):
        if not isinstance(cur, dict):
            raise KeyError(location)
        cur = cur[part]
    if not isinstance(cur, dict):
        raise TypeError(f"{location} did not resolve to an object")
    return cur


def known_artifact_rows(repo: Path = REPO) -> list[dict[str, Any]]:
    """Collect canonical rows from current saved artifacts without mutating them."""
    rows: list[dict[str, Any]] = []

    paper_source = repo / "docs/research/kbound/results_source.json"
    if paper_source.exists():
        data = _load_json(paper_source)
        for group_name in ("natural_shifts", "corruption_grids"):
            group = data.get(group_name, {})
            if not isinstance(group, dict):
                continue
            for dataset, row in group.items():
                if not isinstance(row, dict):
                    continue
                summary = summarize_result_row(
                    dataset=str(dataset),
                    source_level="paper_source",
                    artifact="docs/research/kbound/results_source.json",
                    location=f"$.{group_name}.{dataset}",
                    row=row,
                )
                summary["paper_verdict"] = row.get("verdict", "")
                rows.append(summary)

    specs = [
        ("officehome_M_v2", "run_test_locked", "experiments/kbound/results/officehome_protocol_M_v2/protocol_result.json", "$.test_locked"),
        ("iwildcam_H_v2", "run_test_locked", "experiments/kbound/results/iwildcam_protocol_H_v2/protocol_result.json", "$.test_locked"),
        ("camelyon17_protocol_F", "run_test_locked", "experiments/kbound/results/camelyon17_richZ_F_v1/analyze_F_results.json", "$.test_locked"),
        ("camelyon17_protocol_G", "run_test_locked", "experiments/kbound/results/camelyon17_protocol_G_v1/analyze_F_results.json", "$.test_locked"),
        ("rxrx1_protocol_J", "run_test_locked", "experiments/kbound/results/rxrx1_protocol_J_v1/analyze_F_results.json", "$.test_locked"),
        ("mixed_oof_stream", "run_oof_summary", "experiments/kbound/results/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json", "$"),
    ]
    for dataset, level, rel, loc in specs:
        path = repo / rel
        if not path.exists():
            continue
        data = _load_json(path)
        row = _lookup(data, loc)
        rows.append(
            summarize_result_row(
                dataset=dataset,
                source_level=level,
                artifact=rel,
                location=loc,
                row=row,
            )
        )

    # Multi-seed analyzers expose candidate rows plus comparison-level CI flags.
    multiseed_specs = [
        ("cifar10c_stress", "run_multiseed_locked", "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json"),
        ("imagenet-r_protocol_d", "run_multiseed", "experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json"),
    ]
    for dataset, level, rel in multiseed_specs:
        path = repo / rel
        if not path.exists():
            continue
        data = _load_json(path)
        comparisons = data.get("comparisons", [])
        if not isinstance(comparisons, list):
            comparisons = []
        candidates = data.get("candidates", {})
        if not isinstance(candidates, dict):
            continue
        for candidate, row in candidates.items():
            if not isinstance(row, dict):
                continue
            rows.append(
                summarize_result_row(
                    dataset=dataset,
                    source_level=level,
                    candidate=str(candidate),
                    artifact=rel,
                    location=f"$.candidates.{candidate}",
                    row=row,
                    ci_robust=str(ci_robust_beats_both_from_comparisons(comparisons, str(candidate))),
                )
            )

    decisive_specs = [
        ("imagenetc_noise", "experiments/kbound/results/imagenetc_noise/decisive_tta_results.json"),
        ("cifar10c_decisive", "experiments/kbound/results/decisive_tta_results.json"),
    ]
    for dataset, rel in decisive_specs:
        path = repo / rel
        if not path.exists():
            continue
        data = _load_json(path)
        benches = data.get("benchmarks", {})
        if not isinstance(benches, dict):
            continue
        for bench, bench_doc in benches.items():
            methods = bench_doc.get("methods", {}) if isinstance(bench_doc, dict) else {}
            if not isinstance(methods, dict):
                continue
            for candidate, method_doc in methods.items():
                row = method_doc.get("metrics", method_doc) if isinstance(method_doc, dict) else {}
                if not isinstance(row, dict):
                    continue
                rows.append(
                    summarize_result_row(
                        dataset=f"{dataset}/{bench}",
                        source_level="run_decisive_tta",
                        candidate=str(candidate),
                        artifact=rel,
                        location=f"$.benchmarks.{bench}.methods.{candidate}.metrics",
                        row=row,
                    )
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit saved K-Bound results under one verdict logic.")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of TSV")
    parser.add_argument("--strict-explicit", action="store_true", help="fail if explicit beats_both disagrees with point rule")
    args = parser.parse_args(argv)

    rows = known_artifact_rows(args.repo)
    rows.sort(key=lambda r: (r["dataset"], r["source_level"], r["candidate"]))
    if args.json:
        print(json.dumps({"rows": rows}, indent=2))
    else:
        print("dataset\tlevel\tcandidate\tpoint_bb\tci_bb\tkga\tadapt\tfreeze\tartifact")
        for row in rows:
            print(
                f"{row['dataset']}\t{row['source_level']}\t{row['candidate'] or '-'}\t"
                f"{row['point_beats_both']}\t{row['ci_robust_beats_both']}\t"
                f"{row['regret_kga']:.8g}\t{row['regret_adapt']:.8g}\t{row['regret_freeze']:.8g}\t"
                f"{row['artifact']}"
            )

    inconsistent = [
        row for row in rows
        if row["explicit_point_consistent"] not in (True, "not_present")
    ]
    if args.strict_explicit and inconsistent:
        print(f"ERROR: {len(inconsistent)} explicit beats_both flags disagree with the unified point rule")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
