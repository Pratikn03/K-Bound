#!/usr/bin/env python3
"""Per-sample DeLong + paired bootstrap on M2 external (3D-ADAM) archived parquets.

Uses seed-averaged test prediction vectors (5 seeds) and the locked
Phase-2 ensemble inference rule in ``elara.evaluation.ensemble_inference``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from elara.evaluation.ensemble_inference import audited_analysis


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _pick_parquet_paths(
    index_df: pd.DataFrame,
    *,
    experiment_id: str,
    method: str,
    split: str = "test",
) -> dict[int, Path]:
    sub = index_df[
        (index_df["experiment_id"] == experiment_id)
        & (index_df["method"] == method)
        & (index_df["split"] == split)
        & (index_df["usable_for_inference"].astype(str).str.lower().isin({"true", "1"}))
    ]
    by_seed: dict[int, tuple[int, Path]] = {}
    for _, row in sub.iterrows():
        seed = int(row["seed"])
        path = Path(row["artifact_path"])
        if not path.is_file():
            root = _repo_root()
            alt = root / str(row["artifact_path"])
            path = alt if alt.is_file() else path
        rerun = 0
        if "__rerun_" in path.stem:
            try:
                rerun = int(path.stem.split("__rerun_")[-1])
            except ValueError:
                rerun = 0
        prev = by_seed.get(seed)
        if prev is None or rerun > prev[0]:
            by_seed[seed] = (rerun, path)
    return {s: p for s, (_, p) in by_seed.items()}


def _load_method_scores(paths: dict[int, Path], score_col: str = "raw_score") -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (sample_ids, labels, per_seed_scores dict)."""
    if not paths:
        raise FileNotFoundError("no parquet paths for method")
    seeds = sorted(paths)
    ref = pd.read_parquet(paths[seeds[0]])
    sample_ids = ref["sample_id"].astype(str).to_numpy()
    labels = ref["label"].astype(int).to_numpy()
    per_seed: dict[int, np.ndarray] = {}
    for s in seeds:
        df = pd.read_parquet(paths[s])
        df = df.set_index("sample_id").loc[sample_ids]
        scores = df[score_col].astype(float).to_numpy()
        per_seed[s] = scores
    return sample_ids, labels, per_seed


def _analysis_row(
    result,
    *,
    comparison_id: str,
    holm_p: float | None = None,
) -> dict:
    ci_excludes_zero = bool(result.bootstrap_ci_low > 0)
    return {
        "comparison_id": comparison_id,
        "cell_id": result.cell_id,
        "benchmark": result.benchmark,
        "protocol": result.protocol,
        "rga_method": result.rga_method,
        "comparator_method": result.comparator_method,
        "n_seeds": result.n_seeds,
        "n_test_samples": result.n_test_samples,
        "per_seed_rga_auc": list(result.per_seed_rga_aucs),
        "per_seed_comparator_auc": list(result.per_seed_comp_aucs),
        "per_seed_delta_auc": list(result.per_seed_deltas),
        "sign_consistent_seeds": result.sign_consistent_seeds,
        "ensemble_rga_auc": result.ensemble_rga_auc,
        "ensemble_comparator_auc": result.ensemble_comparator_auc,
        "ensemble_delta_auc": result.ensemble_delta_auc,
        "delong_p_raw": result.delong_p_value,
        "delong_p_holm": holm_p if holm_p is not None else result.delong_p_holm,
        "bootstrap_95_ci_low": result.bootstrap_ci_low,
        "bootstrap_95_ci_high": result.bootstrap_ci_high,
        "bootstrap_n_iter": result.bootstrap_n_iter,
        "bootstrap_ci_excludes_zero": ci_excludes_zero,
        "practical_effect_band": result.practical_effect_band,
        "inference_label": result.inference_label,
        "transfer_confirmed": bool(ci_excludes_zero and result.ensemble_delta_auc > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        default="elara_master_c/predictions/confirmation/PREDICTION_ARCHIVE_INDEX.csv",
    )
    parser.add_argument(
        "--experiment-id",
        default="M2-EXTERNAL-3D-ADAM",
    )
    parser.add_argument("--rga-method", default="rga_boosted_fusion")
    parser.add_argument(
        "--comparator",
        default="sar_score_adapter",
        help="Frozen strongest comparator (M1 rule: SAR for MVTec supervised-paired)",
    )
    parser.add_argument(
        "--also-static",
        action="store_true",
        default=True,
        help="Also run RGA+ vs static_attention (secondary)",
    )
    parser.add_argument("--bootstrap-n-iter", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--out-json",
        default="experiments/fusion/m2_external_3d_adam_paired_inference.json",
    )
    parser.add_argument(
        "--out-csv",
        default="experiments/fusion/m2_external_3d_adam_paired_inference.csv",
    )
    args = parser.parse_args()

    root = _repo_root()
    index_path = root / args.index
    if not index_path.is_file():
        print(f"ERROR: missing index {index_path}", file=sys.stderr)
        return 1

    index_df = pd.read_csv(index_path)
    rga_paths = _pick_parquet_paths(index_df, experiment_id=args.experiment_id, method=args.rga_method)
    comp_paths = _pick_parquet_paths(
        index_df, experiment_id=args.experiment_id, method=args.comparator
    )

    _, labels, per_seed_rga = _load_method_scores(rga_paths)
    _, _, per_seed_comp = _load_method_scores(comp_paths)

    benchmark = "3D-ADAM category-held-out"
    protocol = "M2_external_one_shot_audit"

    comparisons: list[dict] = []
    holm_input: dict[str, float] = {}

    primary = audited_analysis(
        cell_id="M2-EXTERNAL-vs-SAR",
        benchmark=benchmark,
        protocol=protocol,
        rga_method=args.rga_method,
        comparator_method=args.comparator,
        sample_ids=np.arange(len(labels)),  # unused internally
        labels=labels,
        per_seed_rga_scores=per_seed_rga,
        per_seed_comp_scores=per_seed_comp,
        holm_input=None,
        bootstrap_n_iter=args.bootstrap_n_iter,
        bootstrap_seed=args.bootstrap_seed,
    )
    holm_input["M2-EXTERNAL-vs-SAR"] = primary.delong_p_value
    comparisons.append(_analysis_row(primary, comparison_id="M2-EXTERNAL-vs-SAR"))

    if args.also_static:
        static_paths = _pick_parquet_paths(
            index_df, experiment_id=args.experiment_id, method="static_attention"
        )
        _, _, per_seed_static = _load_method_scores(static_paths)
        secondary = audited_analysis(
            cell_id="M2-EXTERNAL-vs-STATIC",
            benchmark=benchmark,
            protocol=protocol,
            rga_method=args.rga_method,
            comparator_method="static_attention",
            sample_ids=np.arange(len(labels)),
            labels=labels,
            per_seed_rga_scores=per_seed_rga,
            per_seed_comp_scores=per_seed_static,
            holm_input=holm_input,
            holm_K=2,
            bootstrap_n_iter=args.bootstrap_n_iter,
            bootstrap_seed=args.bootstrap_seed,
        )
        comparisons.append(
            _analysis_row(
                secondary,
                comparison_id="M2-EXTERNAL-vs-STATIC",
                holm_p=secondary.delong_p_holm,
            )
        )
        # Re-run primary with Holm K=2
        primary_holm = audited_analysis(
            cell_id="M2-EXTERNAL-vs-SAR",
            benchmark=benchmark,
            protocol=protocol,
            rga_method=args.rga_method,
            comparator_method=args.comparator,
            sample_ids=np.arange(len(labels)),
            labels=labels,
            per_seed_rga_scores=per_seed_rga,
            per_seed_comp_scores=per_seed_comp,
            holm_input={
                "M2-EXTERNAL-vs-SAR": primary.delong_p_value,
                "M2-EXTERNAL-vs-STATIC": secondary.delong_p_value,
            },
            holm_K=2,
            bootstrap_n_iter=args.bootstrap_n_iter,
            bootstrap_seed=args.bootstrap_seed,
        )
        comparisons[0] = _analysis_row(
            primary_holm,
            comparison_id="M2-EXTERNAL-vs-SAR",
            holm_p=primary_holm.delong_p_holm,
        )

    primary_row = comparisons[0]
    report = {
        "dataset_id": "m2_3d_adam_anomalib_external",
        "experiment_id": args.experiment_id,
        "archive_index": str(index_path.relative_to(root)),
        "n_test_samples": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "seeds": sorted(per_seed_rga.keys()),
        "primary_comparison": "M2-EXTERNAL-vs-SAR",
        "frozen_comparator": args.comparator,
        "comparisons": comparisons,
        "gate_e_transfer_confirmed": bool(primary_row["transfer_confirmed"]),
        "gate_d_beat_comparator": bool(primary_row["ensemble_delta_auc"] > 0),
        "cell_valid": True,
        "validity_note": (
            "Per-sample paired DeLong on seed-averaged ensemble predictions; "
            "95% bootstrap CI on paired AUROC delta (10k resamples, seed 0)."
        ),
    }

    out_json = root / args.out_json
    out_csv = root / args.out_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fieldnames = list(comparisons[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in comparisons:
            flat = row.copy()
            for k in ("per_seed_rga_auc", "per_seed_comparator_auc", "per_seed_delta_auc"):
                flat[k] = json.dumps(flat[k])
            w.writerow(flat)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}\nWrote {out_csv}")

    # Refresh confirmatory statistics report
    py = sys.executable
    import subprocess

    rc = subprocess.call(
        [py, "src/scripts/scenario_c/confirmatory_statistics.py"],
        cwd=root,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
