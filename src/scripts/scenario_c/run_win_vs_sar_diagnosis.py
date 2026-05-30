#!/usr/bin/env python3
"""Phase-1 WIN vs SAR diagnosis: per-category AUC, GDR rates, SAR TTA ablation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.scripts.scenario_c.win_vs_sar_harness import _repo_root, evaluate_seed

logger = logging.getLogger(__name__)

PRESET_CONFIGS = {
    "m1": "configs/elara_deploy_m1_validation_v1.yaml",
    "m2": "configs/elara_deploy_m2_external_validation_v1.yaml",
}


def _load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _summarize_per_category(rows: list[dict]) -> dict:
    """Mean per-category AUROC across seeds for each method."""
    cats: set[str] = set()
    methods: set[str] = set()
    for row in rows:
        for cat, metrics in (row.get("per_category") or {}).items():
            if metrics.get("_skipped"):
                continue
            cats.add(cat)
            for method, val in metrics.items():
                if method in {"n_samples", "n_positive", "_skipped"}:
                    continue
                if isinstance(val, (int, float)) and np.isfinite(val):
                    methods.add(method)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for cat in sorted(cats):
        out[cat] = {}
        for method in sorted(methods):
            values = []
            for row in rows:
                block = (row.get("per_category") or {}).get(cat, {})
                v = block.get(method)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    values.append(float(v))
            if values:
                out[cat][method] = {
                    "roc_auc_mean": float(np.mean(values)),
                    "roc_auc_std": float(np.std(values)),
                    "n_seeds": len(values),
                }
    return out


def _archive_diagnosis(
    root: Path,
    *,
    index_path: Path,
    experiment_id: str,
    split: str,
    seeds: list[int],
) -> dict:
    from src.scripts.scenario_c.run_m2_external_paired_inference import _load_method_scores, _pick_parquet_paths

    index_df = pd.read_csv(index_path)
    methods = ["static_attention", "craf_attention", "rga_boosted_fusion", "sar_score_adapter"]
    per_method_seed_scores: dict[str, dict[int, np.ndarray]] = {}
    labels = None
    sample_ids = None
    for method in methods:
        paths = _pick_parquet_paths(index_df, experiment_id=experiment_id, method=method, split=split)
        paths = {s: paths[s] for s in seeds if s in paths}
        if not paths:
            continue
        sid, lab, per_seed = _load_method_scores(paths)
        sample_ids = sid
        labels = lab
        per_method_seed_scores[method] = per_seed

    if labels is None:
        return {"error": "no archived methods found", "experiment_id": experiment_id}

    per_seed_rows = []
    for seed in seeds:
        if seed not in next(iter(per_method_seed_scores.values()), {}):
            continue
        method_aucs = {}
        for method, by_seed in per_method_seed_scores.items():
            if seed in by_seed:
                try:
                    method_aucs[method] = float(
                        __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels, by_seed[seed])
                    )
                except ValueError:
                    method_aucs[method] = 0.5
        sar = method_aucs.get("sar_score_adapter", 0.5)
        per_seed_rows.append(
            {
                "seed": seed,
                "method_roc_auc": method_aucs,
                "delta_vs_sar": {k: float(v) - sar for k, v in method_aucs.items()},
            }
        )

    return {
        "mode": "archive",
        "experiment_id": experiment_id,
        "split": split,
        "n_samples": int(len(labels)),
        "per_seed": per_seed_rows,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="WIN vs SAR diagnosis")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS))
    parser.add_argument("--config", type=str)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42])
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Default: elara_master_c/audits/WIN_VS_SAR_DIAGNOSIS_<preset>.json",
    )
    parser.add_argument(
        "--archive-index",
        type=str,
        default="",
        help="Optional prediction archive index CSV for read-only diagnosis",
    )
    parser.add_argument("--archive-experiment-id", type=str, default="M2-EXTERNAL-3D-ADAM")
    parser.add_argument("--archive-split", type=str, default="test")
    args = parser.parse_args()

    root = _repo_root()
    if args.config:
        cfg_path = root / args.config
    elif args.preset:
        cfg_path = root / PRESET_CONFIGS[args.preset]
    else:
        parser.error("Provide --preset or --config")
        return 2

    preset = args.preset or cfg_path.stem
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(cfg_path.relative_to(root)),
    }

    if args.archive_index:
        report["archive"] = _archive_diagnosis(
            root,
            index_path=root / args.archive_index,
            experiment_id=args.archive_experiment_id,
            split=args.archive_split,
            seeds=list(args.seeds),
        )
    else:
        cfg = _load_cfg(cfg_path)
        cfg.setdefault("elara_deploy", {})
        cfg["elara_deploy"]["policy_path"] = str(root / "research_lock/ELARA_DEPLOY_v1.yaml")

        fresh_rows = []
        ablation_rows = []
        for seed in args.seeds:
            logger.info("Diagnosis seed=%s (SAR steps=25)", seed)
            fresh_rows.append(evaluate_seed(cfg, seed=seed, eval_split="validation"))
            logger.info("Diagnosis seed=%s (SAR TTA ablation steps=0)", seed)
            ablation_rows.append(
                evaluate_seed(cfg, seed=seed, eval_split="validation", sar_adaptation_steps=0)
            )

        report["fresh_validation"] = {
            "per_seed": fresh_rows,
            "per_category_summary": _summarize_per_category(fresh_rows),
            "deploy_stats": [r.get("deploy_stats") for r in fresh_rows],
            "gdr_suppress_rate": float(
                np.mean([1.0 - float(r["deploy_stats"].get("switch_allowed_batches", 0)) / max(r["deploy_stats"].get("n_batches", 1), 1) for r in fresh_rows])
            ),
        }
        report["sar_tta_ablation"] = {
            "per_seed": ablation_rows,
            "delta_roc_auc_sar_with_vs_without_tta": [
                float(a["methods"]["sar_score_adapter"]["roc_auc"])
                - float(b["methods"]["sar_score_adapter"]["roc_auc"])
                for a, b in zip(fresh_rows, ablation_rows, strict=True)
            ],
        }

    out_path = Path(args.output) if args.output else root / f"elara_master_c/audits/WIN_VS_SAR_DIAGNOSIS_{preset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
