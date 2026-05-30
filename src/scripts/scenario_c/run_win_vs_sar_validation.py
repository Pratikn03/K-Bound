#!/usr/bin/env python3
"""Validation-only WIN vs SAR sweep (M1 / M2 external dev configs).

Scores only the validation split unless ``--allow-test`` is passed for a
locked one-shot confirmatory rerun after a positive validation stop rule.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.scripts.scenario_c.win_vs_sar_harness import _repo_root, aggregate_seed_rows, evaluate_seed
from uais.fusion.attention.elara_deploy_policy import load_deploy_policy_spec

logger = logging.getLogger(__name__)

PRESET_CONFIGS = {
    "m1": "configs/elara_deploy_m1_validation_v1.yaml",
    "m2": "configs/elara_deploy_m2_external_validation_v1.yaml",
}


def _load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="WIN vs SAR validation-only evaluation")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS),
        help="Use bundled M1 or M2 external validation config",
    )
    parser.add_argument("--config", type=str, help="Override config YAML path")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="JSON output path (default: elara_master_c/audits/win_vs_sar_validation_<preset>.json)",
    )
    parser.add_argument("--seeds", type=int, nargs="*", help="Override evaluation seeds")
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Permit scoring the locked test split (confirmatory one-shot only)",
    )
    parser.add_argument(
        "--deploy-lock",
        type=str,
        default="research_lock/ELARA_DEPLOY_v2.yaml",
        help="Deploy policy lock YAML",
    )
    args = parser.parse_args()

    root = _repo_root()
    if args.config:
        cfg_path = root / args.config
    elif args.preset:
        cfg_path = root / PRESET_CONFIGS[args.preset]
    else:
        parser.error("Provide --preset m1|m2 or --config PATH")
        return 2

    if not cfg_path.is_file():
        logger.error("Config not found: %s", cfg_path)
        return 1

    cfg = _load_cfg(cfg_path)
    deploy = load_deploy_policy_spec(root / args.deploy_lock)
    cfg.setdefault("elara_deploy", {})
    cfg["elara_deploy"]["policy_path"] = str(root / args.deploy_lock)
    if args.allow_test:
        cfg["elara_deploy"]["allow_test_eval"] = True

    eval_cfg = cfg.get("evaluation", {})
    seeds = [int(s) for s in args.seeds] if args.seeds else [int(s) for s in eval_cfg.get("seeds", [42])]
    preset = args.preset or cfg_path.stem

    rows = []
    for seed in seeds:
        logger.info("WIN vs SAR seed=%s eval_split=%s", seed, args.eval_split)
        rows.append(
            evaluate_seed(
                cfg,
                seed=seed,
                eval_split=args.eval_split,
                deploy_policy=deploy,
            )
        )

    summary = aggregate_seed_rows(rows)
    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(cfg_path.relative_to(root)),
        "deploy_policy": deploy.policy_id,
        "eval_split": args.eval_split,
        "seeds": seeds,
        "per_seed": rows,
        "summary": summary,
        "stop_rule": {
            "validation_must_beat_sar_mean_delta": 0.0,
            "passed": bool(summary.get("recommend_confirmatory_test")),
        },
    }

    out_path = Path(args.output) if args.output else root / f"elara_master_c/audits/win_vs_sar_validation_{preset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info(
        "%s Δ vs SAR (mean): %.4f | recommend confirmatory test: %s",
        summary.get("deploy_method", "elara_deploy"),
        summary.get("delta_roc_auc_vs_sar", {}).get(summary.get("deploy_method", ""), {}).get("mean", float("nan")),
        summary.get("recommend_confirmatory_test"),
    )
    return 0 if summary.get("recommend_confirmatory_test") else 2


if __name__ == "__main__":
    sys.exit(main())
