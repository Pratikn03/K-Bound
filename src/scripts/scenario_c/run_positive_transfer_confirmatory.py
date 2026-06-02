#!/usr/bin/env python3
"""Official natural positive-transfer confirmatory runner.

Opened 3D-ADAM and MulSen outputs can be diagnosed elsewhere, but this script
marks a result official only when the holdout is fresh/unopened and both clean
natural endpoints beat SAR and CW.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uais.fusion.attention.positive_transfer import (  # noqa: E402
    coendpoint_pass,
    paired_auc_bootstrap,
    score_selected_candidate,
    select_candidate_on_validation,
)

SAR_MIN_DELTA = 0.010
CW_MIN_DELTA = 0.005
CI_LOW_GT = 0.0


def _pivot(csv_path: Path, split: str) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    s = df[df["split"] == split].copy()
    if s.empty:
        raise ValueError(f"split not found in {csv_path}: {split}")
    domains = list(s["domain"].drop_duplicates())
    if "rgb" in domains:
        first = "rgb"
        second = next(d for d in domains if d != "rgb")
    else:
        first, second = domains[:2]
    a = s[s["domain"] == first].set_index("sample_id")
    b = s[s["domain"] == second].set_index("sample_id")
    ids = a.index.intersection(b.index)
    return {
        "ids": ids.astype(str).tolist(),
        "domain_a": first,
        "domain_b": second,
        "a": a.loc[ids, "score"].to_numpy(dtype=float),
        "b": b.loc[ids, "score"].to_numpy(dtype=float),
        "a_conf": a.loc[ids, "confidence"].to_numpy(dtype=float) if "confidence" in a else None,
        "b_conf": b.loc[ids, "confidence"].to_numpy(dtype=float) if "confidence" in b else None,
        "y": a.loc[ids, "label"].to_numpy(dtype=int),
    }


def _read_score_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _align_scores(ids: list[str], score_file: Path, *, column: str = "raw_score") -> list[float]:
    df = _read_score_file(score_file).set_index("sample_id")
    missing = [sid for sid in ids if sid not in df.index]
    if missing:
        raise ValueError(f"{score_file} missing {len(missing)} sample ids; first={missing[0]}")
    return df.loc[ids, column].to_numpy(dtype=float).tolist()


def is_official_confirmation(result: str | Path | dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        result = json.loads(Path(result).read_text(encoding="utf-8"))
    stats = result.get("stats") or {}
    return bool(
        result.get("protocol") == "POSITIVE_TRANSFER_PROTOCOL_v1"
        and result.get("holdout_status") == "FRESH_OR_UNOPENED"
        and result.get("natural_clean_transfer") is True
        and result.get("synthetic_or_corrupted") is False
        and coendpoint_pass(
            stats.get("vs_sar") or {},
            minimum_practical_delta=SAR_MIN_DELTA,
            ci_low_must_be_gt=CI_LOW_GT,
        )
        and coendpoint_pass(
            stats.get("vs_cw") or {},
            minimum_practical_delta=CW_MIN_DELTA,
            ci_low_must_be_gt=CI_LOW_GT,
        )
    )


def _endpoint_passes(stats: dict[str, Any]) -> dict[str, bool]:
    return {
        "vs_sar": coendpoint_pass(
            stats.get("vs_sar") or {},
            minimum_practical_delta=SAR_MIN_DELTA,
            ci_low_must_be_gt=CI_LOW_GT,
        ),
        "vs_cw": coendpoint_pass(
            stats.get("vs_cw") or {},
            minimum_practical_delta=CW_MIN_DELTA,
            ci_low_must_be_gt=CI_LOW_GT,
        ),
    }


def _official_attempt(
    *,
    holdout_status: str,
    synthetic_or_corrupted: bool,
) -> bool:
    return bool(holdout_status == "FRESH_OR_UNOPENED" and not synthetic_or_corrupted)


def _confirmation_note(
    *,
    confirmed: bool,
    official_attempt: bool,
    endpoint_pass: dict[str, bool],
    synthetic_or_corrupted: bool,
) -> tuple[str, str]:
    if confirmed:
        return "OFFICIAL_PASS", "official positive transfer confirmed"
    if synthetic_or_corrupted:
        return "REJECTED", "not official: synthetic/corrupted inputs are forbidden for D13"
    if not official_attempt:
        return "DEVELOPMENT_ONLY", "not official: holdout is opened development evidence"
    failed = [
        label
        for key, label in (("vs_sar", "SAR"), ("vs_cw", "CW"))
        if not endpoint_pass.get(key, False)
    ]
    if not failed:
        failed = ["unknown"]
    return "OFFICIAL_FAIL", "official attempt not confirmed: " + ", ".join(failed) + " endpoint failed"


def run_confirmatory(
    *,
    csv_path: Path,
    dataset_id: str,
    benchmark: str,
    holdout_status: str,
    sar_scores: Path,
    cw_scores: Path | None,
    output: Path,
    bootstrap_iter: int,
    synthetic_or_corrupted: bool,
) -> dict[str, Any]:
    val = _pivot(csv_path, "validation")
    test = _pivot(csv_path, "test")
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
    sar = _align_scores(test["ids"], sar_scores)
    if cw_scores is not None:
        cw = _align_scores(test["ids"], cw_scores)
    else:
        cw = score_selected_candidate(
            {"selected_rule": "cw", "params": {}},
            test["a"],
            test["b"],
            rgb_confidence=test["a_conf"],
            depth_confidence=test["b_conf"],
        )
    stats = {
        "vs_sar": paired_auc_bootstrap(test["y"], candidate, sar, n_iter=bootstrap_iter, seed=0),
        "vs_cw": paired_auc_bootstrap(test["y"], candidate, cw, n_iter=bootstrap_iter, seed=1),
    }
    endpoint_pass = _endpoint_passes(stats)
    result = {
        "protocol": "POSITIVE_TRANSFER_PROTOCOL_v1",
        "dataset_id": dataset_id,
        "benchmark": benchmark,
        "status": "PENDING_EVALUATION",
        "holdout_status": holdout_status,
        "natural_clean_transfer": True,
        "synthetic_or_corrupted": bool(synthetic_or_corrupted),
        "selection": selection.as_dict(),
        "stats": stats,
        "endpoint_pass": endpoint_pass,
        "gate_e_positive_transfer_confirmed": False,
        "official_note": "",
    }
    result["gate_e_positive_transfer_confirmed"] = is_official_confirmation(result)
    result["status"], result["official_note"] = _confirmation_note(
        confirmed=bool(result["gate_e_positive_transfer_confirmed"]),
        official_attempt=_official_attempt(
            holdout_status=holdout_status,
            synthetic_or_corrupted=synthetic_or_corrupted,
        ),
        endpoint_pass=endpoint_pass,
        synthetic_or_corrupted=synthetic_or_corrupted,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--holdout-status", choices=["FRESH_OR_UNOPENED", "OPENED_DEVELOPMENT_ONLY"], required=True)
    parser.add_argument("--sar-scores", type=Path, required=True)
    parser.add_argument("--cw-scores", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/fusion/positive_transfer_confirmatory_result.json")
    parser.add_argument("--bootstrap-iter", type=int, default=10000)
    parser.add_argument("--synthetic-or-corrupted", action="store_true")
    args = parser.parse_args()
    result = run_confirmatory(
        csv_path=args.csv,
        dataset_id=args.dataset_id,
        benchmark=args.benchmark,
        holdout_status=args.holdout_status,
        sar_scores=args.sar_scores,
        cw_scores=args.cw_scores,
        output=args.output,
        bootstrap_iter=args.bootstrap_iter,
        synthetic_or_corrupted=args.synthetic_or_corrupted,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
