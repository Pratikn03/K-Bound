#!/usr/bin/env python3
"""Dev-side K-Bound win finder.

This is an exploratory search tool, not a paper-claim generator.  It scans
existing record JSONs, tries a small locked-family estimator/conformal grid on
seed-heldout splits, and reports candidates that are worth a fresh locked test.

Default behavior excludes paths that look like final test artifacts.  Use
``--include-test-sources`` only for diagnosis; results from that mode are not
clean held-out evidence.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # noqa: E402


DEFAULT_ROOTS = [
    "experiments/kbound/results/*val*/result_*.json",
    "experiments/kbound/results/*idval*/result_*.json",
    "experiments/kbound/results/*dev*/result_*.json",
    "experiments/kbound/results/*source*/result_*.json",
    "experiments/kbound/results/officehome_full_targetval/result_*.json",
    "experiments/kbound/results/poverty_protocol_L_dev/result_*.json",
    "experiments/kbound/results/camelyon17_richZ_F_v1/result_*.json",
    "experiments/kbound/results/camelyon17_fullscale_B_v2/_partial.json",
    "experiments/kbound/results/iwildcam_full_val/result_*.json",
    "experiments/kbound/results/iwildcam_full_idval/result_*.json",
    "experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_*.json",
    "experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed*/result_*.json",
]

DEFAULT_TEST_ROOTS = [
    "experiments/kbound/results/*test*/result_*.json",
    "experiments/kbound/results/officehome_full_targettest/result_*.json",
    "experiments/kbound/results/iwildcam_full_test/result_*.json",
    "experiments/kbound/results/fmow_protocol_L_test/result_*.json",
]

KNOWN_MULTI_FILE_PANELS = [
    (
        "cifar10c_stress",
        "experiments/kbound/results/stress_grid_multiseed_v1/seed*/per_condition_cifar10c_*_seed*.json",
    ),
    (
        "cifar101_multiseed",
        "experiments/kbound/results/cifar101_multiseed_v1/seed*/per_condition_cifar101_*_seed*.json",
    ),
]


def looks_test_path(path: Path) -> bool:
    s = str(path).lower()
    return "targettest" in s or "/test" in s or "_test" in s


def has_records(path: Path) -> bool:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(d, dict) and isinstance(d.get("records"), list) and bool(d["records"])


def expand_patterns(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(Path(p) for p in glob.glob(pat))
    return sorted(set(p for p in out if p.is_file() and has_records(p)))


def load_records_from_file(path: Path) -> list[dict]:
    """Load one record JSON while preserving a useful candidate label."""
    data = json.loads(path.read_text())
    top_method = data.get("method")
    if top_method and data.get("records"):
        recs, _ = af.load_records(str(path), candidate=str(top_method))
    else:
        recs, _ = af.load_records(str(path), candidate=None)
    return clean_records(recs)


def load_multi_file_panel(panel_name: str, pattern: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path(p) for p in glob.glob(pattern)):
        if not path.is_file() or not has_records(path):
            continue
        data = json.loads(path.read_text())
        cand = data.get("method")
        if not cand:
            stem = path.stem
            parts = stem.split("_")
            cand = parts[2] if len(parts) >= 3 else "unknown"
        recs, _ = af.load_records(str(path), candidate=str(cand))
        groups[f"{panel_name}:{cand}"].extend(clean_records(recs))
    return dict(groups)


def clean_records(recs: list[dict]) -> list[dict]:
    cleaned = []
    zdim = None
    for r in recs:
        try:
            z = np.asarray(r["Z"], dtype=float)
            vals = [float(r["B"]), float(r["a0"]), float(r["aa"])]
            if not np.isfinite(z).all() or not np.isfinite(vals).all():
                continue
            if zdim is None:
                zdim = len(z)
            if len(z) != zdim:
                continue
            rr = dict(r)
            rr["Z"] = [float(v) for v in z]
            rr["B"], rr["a0"], rr["aa"] = vals
            rr["seed"] = int(rr["seed"])
            rr["candidate"] = str(rr.get("candidate", rr.get("method", "unknown")))
            cleaned.append(rr)
        except Exception:
            continue
    return cleaned


def split_seeds(seeds: list[int], test_frac: float = 0.4) -> tuple[list[int], list[int]]:
    seeds = sorted(set(int(s) for s in seeds))
    if len(seeds) < 2:
        return seeds, []
    n_test = max(1, int(math.ceil(len(seeds) * test_frac)))
    n_test = min(n_test, len(seeds) - 1)
    return seeds[:-n_test], seeds[-n_test:]


def group_by_candidate(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[str(r.get("candidate", "unknown"))].append(r)
    return dict(out)


def evaluate_one(
    label: str,
    records: list[dict],
    estimators: list[str],
    conformals: list[str],
    min_records: int,
    min_coverage: float,
) -> list[dict]:
    rows = []
    if len(records) < min_records:
        return rows
    seeds = sorted(set(int(r["seed"]) for r in records))
    cal_seeds, score_seeds = split_seeds(seeds)
    if not score_seeds:
        return rows
    B = np.asarray([r["B"] for r in records], dtype=float)
    harmful_rate = float(np.mean(B < 0)) if len(B) else None
    for est in estimators:
        for conf in conformals:
            try:
                m = af.run_split(records, cal_seeds, score_seeds, estimator=est, conformal=conf)
            except Exception as e:
                rows.append({
                    "label": label,
                    "estimator": est,
                    "conformal": conf,
                    "status": "error",
                    "error": repr(e),
                    "n_records": len(records),
                    "seeds": seeds,
                })
                continue
            if not m:
                continue
            k = float(m["regret_kga"])
            a = float(m["regret_adapt"])
            f = float(m["regret_freeze"])
            margin = min(a, f) - k
            beats_both = k < a and k < f
            fa_ok = float(m["false_adapt"]) <= af.ALPHA
            cov_ok = float(m["coverage"]) >= min_coverage
            rows.append({
                "label": label,
                "estimator": est,
                "conformal": conf,
                "status": "ok",
                "candidate_win": bool(beats_both and fa_ok and cov_ok),
                "beats_both": bool(beats_both),
                "fa_ok": bool(fa_ok),
                "coverage_ok": bool(cov_ok),
                "margin": margin,
                "harmful_rate": harmful_rate,
                "n_records": len(records),
                "cal_seeds": cal_seeds,
                "score_seeds": score_seeds,
                **m,
            })
    return rows


def sort_key(row: dict) -> tuple:
    return (
        0 if row.get("candidate_win") else 1,
        -float(row.get("margin", -999.0)),
        float(row.get("false_adapt", 999.0)),
        -float(row.get("coverage", 0.0)),
    )


def write_markdown(rows: list[dict], out_md: Path, top: int) -> None:
    shown = sorted([r for r in rows if r.get("status") == "ok"], key=sort_key)[:top]
    lines = [
        "# K-Bound Dev Win Finder",
        "",
        "Exploratory/dev-screen report. A row is paper-eligible only after a fresh locked held-out evaluation.",
        "",
        "| rank | win? | label | est | conf | margin | KGA | adapt | freeze | FA | cov | seeds |",
        "|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(shown, 1):
        seeds = f"{r.get('cal_seeds')}->{r.get('score_seeds')}"
        lines.append(
            f"| {i} | {'YES' if r.get('candidate_win') else ''} | "
            f"`{r.get('label')}` | {r.get('estimator')} | {r.get('conformal')} | "
            f"{r.get('margin', 0):.6g} | {r.get('regret_kga', 0):.6g} | "
            f"{r.get('regret_adapt', 0):.6g} | {r.get('regret_freeze', 0):.6g} | "
            f"{r.get('false_adapt', 0):.3g} | {r.get('coverage', 0):.3g} | `{seeds}` |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", nargs="*", default=None, help="record JSON globs; default scans known dev/val artifacts")
    p.add_argument("--include-test-sources", action="store_true", help="diagnostic only; do not use as clean held-out evidence")
    p.add_argument("--include-cqr", action="store_true", help="also try CQR; slower")
    p.add_argument("--no-known-panels", action="store_true", help="only scan --records/default roots; do not append built-in multi-file panels")
    p.add_argument("--min-records", type=int, default=24)
    p.add_argument("--min-coverage", type=float, default=0.20)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--output-dir", default="experiments/kbound/results/win_finder_v1")
    args = p.parse_args()

    patterns = args.records if args.records else list(DEFAULT_ROOTS)
    if args.include_test_sources:
        patterns = list(patterns) + DEFAULT_TEST_ROOTS
    paths = expand_patterns(patterns)
    if not args.include_test_sources:
        paths = [p for p in paths if not looks_test_path(p)]

    sources: dict[str, list[dict]] = {}
    for path in paths:
        recs = load_records_from_file(path)
        for cand, cand_recs in group_by_candidate(recs).items():
            sources[f"{path}:{cand}"] = cand_recs
    if not args.no_known_panels:
        for panel_name, pattern in KNOWN_MULTI_FILE_PANELS:
            sources.update(load_multi_file_panel(panel_name, pattern))

    estimators = ["gbr", "ppi_debias"]
    conformals = ["global", "mondrian"] + (["cqr"] if args.include_cqr else [])

    rows = []
    for label, recs in sorted(sources.items()):
        rows.extend(evaluate_one(label, recs, estimators, conformals, args.min_records, args.min_coverage))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "finder_results.json"
    out_md = out_dir / "finder_report.md"
    payload = {
        "mode": "diagnostic_includes_test_sources" if args.include_test_sources else "dev_screen_excludes_test_sources",
        "alpha": af.ALPHA,
        "min_coverage": args.min_coverage,
        "known_panels_included": not args.no_known_panels,
        "n_sources": len(sources),
        "n_rows": len(rows),
        "n_candidate_wins": sum(1 for r in rows if r.get("candidate_win")),
        "rows": sorted(rows, key=sort_key),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(rows, out_md, args.top)

    ok_rows = [r for r in payload["rows"] if r.get("status") == "ok"]
    print(f"mode={payload['mode']} sources={len(sources)} rows={len(rows)} candidate_wins={payload['n_candidate_wins']}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print("\nTop rows:")
    for i, r in enumerate(ok_rows[: min(args.top, 12)], 1):
        print(
            f"{i:02d} win={r.get('candidate_win')} margin={r.get('margin', 0):.6g} "
            f"kga={r.get('regret_kga', 0):.6g} adapt={r.get('regret_adapt', 0):.6g} "
            f"freeze={r.get('regret_freeze', 0):.6g} FA={r.get('false_adapt', 0):.3g} "
            f"cov={r.get('coverage', 0):.3g} {r.get('estimator')}/{r.get('conformal')} "
            f"{r.get('label')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
