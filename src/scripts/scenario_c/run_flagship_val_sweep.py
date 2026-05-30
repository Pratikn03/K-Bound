#!/usr/bin/env python3
"""Flagship development: RGA+ variants + deploy v3 on validation only (stop rule)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.scripts.scenario_c.flagship_harness import evaluate_flagship_seed
from src.scripts.scenario_c.win_vs_sar_harness import _repo_root

logger = logging.getLogger(__name__)

PRESETS = {
    "m1": "configs/elara_deploy_m1_validation_v1.yaml",
    "m2": "configs/elara_deploy_m2_external_validation_v1.yaml",
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Flagship val-only method sweep")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="m2")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--min-delta-vs-sar", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    root = _repo_root()
    cfg_path = root / PRESETS[args.preset]
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if args.allow_test:
        cfg.setdefault("elara_deploy", {})["allow_test_eval"] = True

    rows = []
    for seed in args.seeds:
        logger.info("Flagship seed=%s split=%s", seed, args.eval_split)
        rows.append(evaluate_flagship_seed(cfg, seed=int(seed), eval_split=args.eval_split))

    # Leaderboard: mean delta vs SAR per variant + deploy v3
    leaderboard: list[dict] = []
    variant_names = sorted(rows[0]["variants"].keys()) if rows else []
    for name in variant_names:
        deltas = [r["variants"][name]["delta_vs_sar"] for r in rows]
        leaderboard.append(
            {
                "name": name,
                "delta_vs_sar_mean": float(sum(deltas) / len(deltas)),
                "win_rate": float(sum(1 for d in deltas if d > 0) / len(deltas)),
                "n_seeds": len(deltas),
            }
        )
    for key in ("elara_deploy_v3", "elara_chf_v1"):
        deltas = [r[key]["delta_vs_sar"] for r in rows if r.get(key)]
        if deltas:
            leaderboard.append(
                {
                    "name": key,
                    "delta_vs_sar_mean": float(sum(deltas) / len(deltas)),
                    "win_rate": float(sum(1 for d in deltas if d > 0) / len(deltas)),
                    "n_seeds": len(deltas),
                }
            )
    leaderboard.sort(key=lambda r: r["delta_vs_sar_mean"], reverse=True)
    winner = leaderboard[0] if leaderboard else None
    stop_passed = bool(winner and winner["delta_vs_sar_mean"] >= args.min_delta_vs_sar)

    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preset": args.preset,
        "eval_split": args.eval_split,
        "seeds": list(args.seeds),
        "min_delta_vs_sar": args.min_delta_vs_sar,
        "per_seed": rows,
        "leaderboard": leaderboard,
        "winner": winner,
        "stop_rule_passed": stop_passed,
        "recommend_confirmatory_test": stop_passed and args.eval_split == "validation",
    }

    out = (
        Path(args.output)
        if args.output
        else root / f"elara_master_c/audits/flagship_val_sweep_{args.preset}_{args.eval_split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    if winner:
        logger.info(
            "Winner %s mean Δ vs SAR=%.4f stop_passed=%s",
            winner["name"],
            winner["delta_vs_sar_mean"],
            stop_passed,
        )
    return 0 if stop_passed else 2


if __name__ == "__main__":
    sys.exit(main())
