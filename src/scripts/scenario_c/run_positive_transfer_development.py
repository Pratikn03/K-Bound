#!/usr/bin/env python3
"""Development-only natural positive-transfer audit on opened datasets."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.scenario_c.run_positive_transfer_confirmatory import _pivot  # noqa: E402
from uais.fusion.attention.positive_transfer import (  # noqa: E402
    paired_auc_bootstrap,
    score_selected_candidate,
    select_candidate_on_validation,
)


def _archive_score(
    *,
    archive: Path,
    benchmark: str,
    protocol: str,
    method: str,
    ids: list[str],
) -> list[float] | None:
    idx_path = archive / "PREDICTION_ARCHIVE_INDEX.csv"
    if idx_path.is_file():
        idx = pd.read_csv(idx_path)
        rows = idx[
            (idx["benchmark"] == benchmark)
            & (idx["protocol"] == protocol)
            & (idx["method"] == method)
            & (idx["split"] == "test")
        ]
        if not rows.empty:
            rows = rows[~rows["artifact_path"].astype(str).str.contains("__rerun")]
            if rows.empty:
                rows = idx[
                    (idx["benchmark"] == benchmark)
                    & (idx["protocol"] == protocol)
                    & (idx["method"] == method)
                    & (idx["split"] == "test")
                ]
            p = ROOT / rows.iloc[0]["artifact_path"]
            if p.is_file():
                df = pd.read_parquet(p).set_index("sample_id")
                return df.loc[ids, "raw_score"].to_numpy(dtype=float).tolist()
    fs = [
        f
        for f in glob.glob(str(archive / f"*/*/{method}/test/seed_42*.parquet"))
        if not Path(f).name.startswith("._")
    ]
    if fs:
        df = pd.read_parquet(fs[0]).set_index("sample_id")
        return df.loc[ids, "raw_score"].to_numpy(dtype=float).tolist()
    return None


def _eval_dataset(spec: dict[str, Any], bootstrap_iter: int) -> dict[str, Any]:
    csv = ROOT / spec["csv"]
    val = _pivot(csv, "validation")
    test = _pivot(csv, "test")
    selection = select_candidate_on_validation(
        val["y"],
        val["a"],
        val["b"],
        val_rgb_confidence=val["a_conf"],
        val_depth_confidence=val["b_conf"],
    )
    candidate = score_selected_candidate(
        selection,
        test["a"],
        test["b"],
        rgb_confidence=test["a_conf"],
        depth_confidence=test["b_conf"],
    )
    archive = ROOT / spec["archive"]
    sar = _archive_score(
        archive=archive,
        benchmark=spec["benchmark"],
        protocol=spec["protocol"],
        method="sar_score_adapter",
        ids=test["ids"],
    )
    cw = _archive_score(
        archive=archive,
        benchmark=spec["benchmark"],
        protocol=spec["protocol"],
        method="confidence_weighted_mean",
        ids=test["ids"],
    )
    if cw is None:
        cw = score_selected_candidate(
            {"selected_rule": "cw", "params": {}},
            test["a"],
            test["b"],
            rgb_confidence=test["a_conf"],
            depth_confidence=test["b_conf"],
        ).tolist()
    stats: dict[str, Any] = {
        "vs_cw": paired_auc_bootstrap(test["y"], candidate, cw, n_iter=bootstrap_iter, seed=1),
    }
    if sar is not None:
        stats["vs_sar"] = paired_auc_bootstrap(test["y"], candidate, sar, n_iter=bootstrap_iter, seed=0)
    else:
        stats["vs_sar"] = {"valid": False, "reason": "missing_sar_scores", "delta": 0.0, "ci95": [0.0, 0.0]}
    return {
        "dataset_id": spec["dataset_id"],
        "benchmark": spec["benchmark"],
        "protocol": spec["protocol"],
        "csv": spec["csv"],
        "holdout_status": "OPENED_DEVELOPMENT_ONLY",
        "selection": selection.as_dict(),
        "stats": stats,
        "can_set_gate_e": False,
    }


def _fmt_signed(value: float | int | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.4f}"


def _fmt_effect(stats: dict[str, Any]) -> str:
    if not stats.get("valid", False):
        return "--"
    ci = stats.get("ci95") or [None, None]
    return f"{_fmt_signed(stats.get('delta'))} $[{_fmt_signed(ci[0])},{_fmt_signed(ci[1])}]$"


def _tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def _short_dataset(row: dict[str, Any]) -> str:
    dataset_id = str(row.get("dataset_id", ""))
    if dataset_id == "3d_adam_v3":
        return "3D-ADAM"
    if dataset_id == "mulsen_v2":
        return "MulSen"
    benchmark = str(row.get("benchmark", ""))
    return benchmark.split()[0] if benchmark else dataset_id


def _development_status(row: dict[str, Any]) -> str:
    sar = row.get("stats", {}).get("vs_sar", {})
    cw = row.get("stats", {}).get("vs_cw", {})
    sar_ci = sar.get("ci95") or [0.0]
    cw_ci = cw.get("ci95") or [0.0]
    sar_positive = bool(sar.get("valid")) and float(sar.get("delta", 0.0)) >= 0.010 and float(sar_ci[0]) > 0.0
    cw_positive = bool(cw.get("valid")) and float(cw.get("delta", 0.0)) >= 0.005 and float(cw_ci[0]) > 0.0
    if sar_positive and cw_positive:
        return "opened-dev positive"
    return "opened-dev unresolved"


def render_development_table(report: dict[str, Any]) -> str:
    """Render the D13 development report as a manuscript-safe LaTeX table."""
    lines = [
        "% Auto-generated by run_positive_transfer_development.py",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Dataset & Rule & $\Delta$ vs SAR (95\% CI) & $\Delta$ vs CW (95\% CI) & Status \\",
        r"\midrule",
    ]
    for row in report.get("datasets", []):
        dataset = _tex_escape(_short_dataset(row))
        rule = _tex_escape(str(row.get("selection", {}).get("selected_rule", "unknown")).replace("_", " "))
        sar = _fmt_effect(row.get("stats", {}).get("vs_sar", {}))
        cw = _fmt_effect(row.get("stats", {}).get("vs_cw", {}))
        status = _tex_escape(_development_status(row))
        lines.append(f"{dataset} & {rule} & {sar} & {cw} & {status} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            "% Opened development evidence cannot set Gate E; official D13 requires a fresh/unopened natural holdout.",
        ]
    )
    return "\n".join(lines)


def build_report(bootstrap_iter: int) -> dict[str, Any]:
    specs = [
        {
            "dataset_id": "3d_adam_v3",
            "benchmark": "3D-ADAM category-held-out",
            "protocol": "M2_external_one_shot_audit",
            "csv": "experiments/fusion/m2_external_3d_adam_v3_inputs.csv",
            "archive": "elara_master_c/predictions/v3_transfer",
        },
        {
            "dataset_id": "mulsen_v2",
            "benchmark": "MulSen-AD category-held-out",
            "protocol": "M2_external_v2_one_shot_audit",
            "csv": "experiments/fusion/m2_external_mulsen_sealed_inputs.csv",
            "archive": "elara_master_c/predictions/confirmation",
        },
    ]
    rows = []
    for spec in specs:
        if (ROOT / spec["csv"]).is_file():
            rows.append(_eval_dataset(spec, bootstrap_iter))
    return {
        "protocol": "POSITIVE_TRANSFER_PROTOCOL_v1",
        "status": "DEVELOPMENT_ONLY",
        "cannot_set_gate_e": True,
        "natural_clean_transfer": True,
        "synthetic_or_corrupted": False,
        "target": "beat both SAR and CW on fresh natural clean transfer",
        "datasets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iter", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "elara_master_c/audits/positive_transfer_development_report.json")
    parser.add_argument("--table-output", type=Path, default=None)
    args = parser.parse_args()
    report = build_report(args.bootstrap_iter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.table_output is not None:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(render_development_table(report) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
