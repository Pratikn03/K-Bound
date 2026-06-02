#!/usr/bin/env python3
"""Phase 2: validation-only deploy policy sweep (no re-test until winner beats SAR)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import yaml

from src.scripts.scenario_c.win_vs_sar_harness import (
    _repo_root,
    _safe_auc,
    train_seed_bundle,
)
from uais.fusion.attention.elara_deploy_policy import DeployPolicySpec, predict_elara_deploy

logger = logging.getLogger(__name__)

PRESET_CONFIGS = {
    "m1": "configs/elara_deploy_m1_validation_v1.yaml",
    "m2": "configs/elara_deploy_m2_external_validation_v1.yaml",
}


def _load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sweep_variants(
    coherence_values: list[float],
    tau_values: list[float],
    routing_modes: list[str],
) -> list[DeployPolicySpec]:
    variants: list[DeployPolicySpec] = []
    for coherence_min, tau, mode in product(coherence_values, tau_values, routing_modes):
        variants.append(
            DeployPolicySpec(
                policy_id=f"SWEEP_c{coherence_min}_t{tau}_{mode}",
                routing_mode=mode,
                fallback_method="sar_score_adapter"
                if mode == "gdr_sar_fallback"
                else "val_selected_sar_or_rga",
                gate_decision_rule={
                    "enabled": True,
                    "coherence_min": float(coherence_min),
                    "tau": float(tau),
                    "margin_epsilon": 0.0,
                },
            )
        )
    return variants


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Phase 2 WIN vs SAR policy sweep")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--coherence", type=float, nargs="*", default=[0.35, 0.45, 0.5])
    parser.add_argument("--tau", type=float, nargs="*", default=[0.55, 0.66, 0.75])
    parser.add_argument(
        "--routing",
        nargs="*",
        default=["gdr_sar_fallback", "gdr_val_router_fallback"],
    )
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    root = _repo_root()
    cfg = _load_cfg(root / PRESET_CONFIGS[args.preset])
    variants = _sweep_variants(args.coherence, args.tau, args.routing)

    per_variant_rows: dict[str, list[dict]] = {v.policy_id: [] for v in variants}
    seed_bundles = []

    for seed in args.seeds:
        logger.info("Training seed bundle seed=%s", seed)
        bundle = train_seed_bundle(cfg, seed=int(seed))
        seed_bundles.append({"seed": seed, "val_fallback": bundle.val_fallback_choice})

        for policy in variants:
            gdr = policy.gate_decision_rule
            cal = bundle.gate_calibrations.get(float(gdr["tau"]))
            artifacts = bundle.make_deploy_artifacts(gdr, cal)
            if policy.routing_mode == "gdr_val_router_fallback":
                fb = bundle.rga_eval if bundle.val_fallback_choice == "rga_boosted_fusion" else bundle.sar_eval
                policy = DeployPolicySpec(
                    policy_id=policy.policy_id,
                    routing_mode=policy.routing_mode,
                    fallback_method=bundle.val_fallback_choice,
                    gate_decision_rule=policy.gate_decision_rule,
                )
            else:
                fb = None
            eval_probs, stats = predict_elara_deploy(
                artifacts,
                bundle.eval_feat,
                bundle.eval_mask,
                policy=policy,
                fallback_probs=fb,
            )
            val_probs, _ = predict_elara_deploy(
                artifacts,
                bundle.val_feat,
                bundle.val_mask,
                policy=policy,
                fallback_probs=bundle.rga_val
                if bundle.val_fallback_choice == "rga_boosted_fusion"
                else None,
            )
            eval_auc = _safe_auc(bundle.eval_labels, eval_probs)
            sar_auc = _safe_auc(bundle.eval_labels, bundle.sar_eval)
            per_variant_rows[policy.policy_id].append(
                {
                    "seed": int(seed),
                    "eval_roc_auc": eval_auc,
                    "sar_roc_auc": sar_auc,
                    "delta_vs_sar": float(eval_auc - sar_auc),
                    "deploy_stats": stats,
                    "val_fallback": bundle.val_fallback_choice,
                    "policy": {
                        "routing_mode": policy.routing_mode,
                        "coherence_min": gdr["coherence_min"],
                        "tau": gdr["tau"],
                    },
                }
            )

    leaderboard = []
    for policy_id, rows in per_variant_rows.items():
        if not rows:
            continue
        deltas = [r["delta_vs_sar"] for r in rows]
        leaderboard.append(
            {
                "policy_id": policy_id,
                "delta_vs_sar_mean": float(sum(deltas) / len(deltas)),
                "delta_vs_sar_std": float(
                    (sum((d - sum(deltas) / len(deltas)) ** 2 for d in deltas) / max(len(deltas) - 1, 1)) ** 0.5
                ),
                "win_rate": float(sum(1 for d in deltas if d > 0) / len(deltas)),
                "n_seeds": len(rows),
                "routing_mode": rows[0]["policy"]["routing_mode"],
                "coherence_min": rows[0]["policy"]["coherence_min"],
                "tau": rows[0]["policy"]["tau"],
            }
        )
    leaderboard.sort(key=lambda r: r["delta_vs_sar_mean"], reverse=True)
    winner = leaderboard[0] if leaderboard else None

    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": 2,
        "preset": args.preset,
        "seeds": list(args.seeds),
        "n_variants": len(variants),
        "seed_bundles": seed_bundles,
        "leaderboard": leaderboard,
        "winner": winner,
        "per_variant": per_variant_rows,
        "stop_rule_passed": bool(winner and winner["delta_vs_sar_mean"] > 0.0),
    }

    out = (
        Path(args.output)
        if args.output
        else root / f"elara_master_c/audits/win_vs_sar_policy_sweep_{args.preset}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    if winner:
        logger.info(
            "Winner %s Δ vs SAR mean=%.4f routing=%s coherence=%s tau=%s",
            winner["policy_id"],
            winner["delta_vs_sar_mean"],
            winner["routing_mode"],
            winner["coherence_min"],
            winner["tau"],
        )
    return 0 if report["stop_rule_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
