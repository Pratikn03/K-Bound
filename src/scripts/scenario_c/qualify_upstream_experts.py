#!/usr/bin/env python3
"""Gate A — qualify upstream domain experts from fusion input CSVs (T1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _qualify_domain(
    df: pd.DataFrame,
    *,
    sample_col: str,
    domain_col: str,
    label_col: str,
    split_col: str,
    score_col: str,
    test_splits: tuple[str, ...],
    domain: str,
) -> dict:
    sub = df[(df[domain_col] == domain) & (df[split_col].isin(test_splits))].copy()
    if sub.empty:
        return {"domain": domain, "passed": False, "reason": "no test rows"}
    # one row per sample
    agg = sub.groupby(sample_col, as_index=False).agg({label_col: "max", score_col: "mean"})
    y = agg[label_col].astype(int).values
    s = agg[score_col].astype(float).values
    if len(np.unique(y)) < 2:
        return {"domain": domain, "passed": False, "reason": "single class in test", "n": int(len(y))}
    roc = float(roc_auc_score(y, s))
    pr = float(average_precision_score(y, s))
    passed = roc > 0.5
    return {
        "domain": domain,
        "passed": passed,
        "roc_auc": roc,
        "pr_auc": pr,
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
    }


def qualify_mvtec(
    csv_path: Path,
    metadata_path: Path | None,
    *,
    min_complement_auc: float = 0.52,
) -> dict:
    df = pd.read_csv(csv_path)
    meta = {}
    if metadata_path and metadata_path.is_file():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    domains = meta.get("domain_order") or sorted(df["domain"].unique().tolist())
    split_col = "split" if "split" in df.columns else "fusion_split"
    test_splits = ("test",)
    per_domain = [
        _qualify_domain(
            df,
            sample_col="sample_id",
            domain_col="domain",
            label_col="label",
            split_col=split_col,
            score_col="score",
            test_splits=test_splits,
            domain=d,
        )
        for d in domains
    ]
    rgb = next((d for d in per_domain if "rgb" in d["domain"]), per_domain[0])
    depth = next((d for d in per_domain if d["domain"] != rgb["domain"]), per_domain[-1])
    complement = bool(rgb.get("roc_auc", 0) > min_complement_auc and depth.get("roc_auc", 0) > min_complement_auc)
    rgb_auc = rgb.get("roc_auc", 0.5)
    depth_auc = depth.get("roc_auc", 0.5)
    depth_adds = abs(depth_auc - rgb_auc) >= 0.02 or (depth_auc > rgb_auc and depth_auc > 0.55)
    gate_pass = all(d.get("passed") for d in per_domain) and complement and depth_adds
    return {
        "dataset": "mvtec_3d_ad",
        "inputs_csv": str(csv_path),
        "feature_mode": meta.get("feature_mode", "unknown"),
        "natural_pairing": meta.get("natural_pairing", True),
        "per_domain": per_domain,
        "depth_adds_information": depth_adds,
        "gate_a_passed": gate_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-parquet", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="Fusion inputs CSV to qualify")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    root = _repo_root()
    canonical = args.csv or (root / "experiments/fusion/mvtec3d_patchcore_v2_inputs.csv")
    meta = args.metadata or (root / "experiments/fusion/mvtec3d_patchcore_v2_metadata.json")
    report: dict = {"datasets": [], "gate_a_overall": False}

    if canonical.is_file():
        report["datasets"].append(qualify_mvtec(canonical, meta if meta.is_file() else None))
    else:
        report["datasets"].append(
            {"dataset": "mvtec_3d_ad", "gate_a_passed": False, "reason": f"missing {canonical}"}
        )

    report["gate_a_overall"] = all(d.get("gate_a_passed") for d in report["datasets"])

    out = args.json_out or (root / "elara_master_c/audits/gate_a_expert_qualification_v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.export_parquet and canonical.is_file():
        df = pd.read_csv(canonical)
        export = root / "elara_master_c/data/processed/mvtec3d_expert_scores.parquet"
        export.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in ("sample_id", "domain", "label", "score", "confidence", "split", "category") if c in df.columns]
        df[cols].to_parquet(export, index=False)
        report["expert_export"] = str(export)

    print(f"Gate A: {'PASS' if report['gate_a_overall'] else 'FAIL'}")
    print(f"Report: {out}")
    for ds in report["datasets"]:
        for d in ds.get("per_domain", []):
            print(f"  {d.get('domain')}: ROC-AUC={d.get('roc_auc', 'n/a')} passed={d.get('passed')}")
    return 0 if report["gate_a_overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
