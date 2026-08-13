#!/usr/bin/env python3
"""Build the canonical promoted-result manifest from the claim ledger.

The ledger controls wording/status.  This manifest records one existing authoritative
artifact for every promoted empirical claim and adds machine-readable headline metrics
for the completed PACS and ImageNet-R diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LEDGER = ROOT / "docs/research/kbound/claim_ledger.json"
OUT = ROOT / "docs/research/kbound/RESULT_MANIFEST.json"
RECONCILED = ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
SOURCE_MANIFEST = ROOT / "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def special_metrics(claim_id: str) -> dict:
    if not RECONCILED.is_file():
        return {}
    panels = json.loads(RECONCILED.read_text())["panels"]
    if claim_id == "KB-CLAIM-011":
        score = panels["imagenetc"]["panel"]["candidates"]["sar"]
        return {
            "n_seeds": 5,
            "n_decisions": score["n"],
            "regret_kga": score["regret"]["kga"],
            "regret_adapt": score["regret"]["always_adapt"],
            "regret_freeze": score["regret"]["always_freeze"],
            "false_adapt_num": score["false_adapt_count"],
            "false_adapt_den": score["n"],
            "ci_robust_beats_both": score["seed_inference"]["ci_robust_beats_both"],
        }
    if claim_id in {"KB-CLAIM-020", "KB-CLAIM-021"}:
        key = "officehome" if claim_id == "KB-CLAIM-020" else "iwildcam"
        primary = panels[key]["primary"]
        score = primary["exact_rank_transfer_score"]
        return {
            "n_decisions": score["n"],
            "regret_kga": score["regret"]["kga"],
            "regret_adapt": score["regret"]["always_adapt"],
            "regret_freeze": score["regret"]["always_freeze"],
            "false_adapt_num": score["false_adapt_count"],
            "false_adapt_den": score["n"],
            "decision_counts": {
                "ADAPT": score["adapt_count"],
                "FREEZE": score["freeze_count"],
                "ABSTAIN": score["abstain_count"],
            },
            "ci_robust_beats_both": score["seed_inference"]["ci_robust_beats_both"],
            "a7_status": primary["calibration"]["a7_status"],
        }
    if claim_id == "KB-CLAIM-041":
        pacs = panels["pacs"]
        score = pacs["pooled_domain_seed_mean"]
        return {
            "n_seeds": len(pacs["seeds"]),
            "n_domain_seed_units": score["n_domain_seed_units"],
            "regret_kga_mean_across_domains": score["regret"]["kga"],
            "regret_adapt_mean_across_domains": score["regret"]["always_adapt"],
            "regret_freeze_mean_across_domains": score["regret"]["always_freeze"],
            "false_adapt_reported_rate": score["fa_u"],
            "decision_replay_available": pacs["decision_replay_available"],
            "beats_both_promoted": False,
        }
    if claim_id == "KB-CLAIM-042":
        grid = panels["imagenet_r"]["panel"]
        score = grid["architecture_panel_aggregate"]
        vals = list(grid["candidates"].values())
        return {
            "n_seeds": len(grid["seeds"]),
            "n_backbones": len(vals),
            "conditions_per_backbone_seed": 12,
            "regret_kga_mean_across_backbones": score["regret"]["kga"],
            "regret_adapt_mean_across_backbones": score["regret"]["always_adapt"],
            "regret_freeze_mean_across_backbones": score["regret"]["always_freeze"],
            "false_adapt_num": score["false_adapt_count"],
            "false_adapt_den": score["n"],
            "beats_both_candidates": sum(row["point_beats_both"] for row in vals),
            "worse_than_always_adapt_candidates": sum(
                row["regret"]["kga"] > row["regret"]["always_adapt"] for row in vals
            ),
        }
    return {}


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    results = []
    missing = []
    for claim in ledger["claims"]:
        if claim.get("claim_type") != "empirical" or claim.get("status") not in {
            "supported", "no-harm", "descriptive", "diagnostic"
        }:
            continue
        existing = [ROOT / rel for rel in claim.get("supporting_artifacts", []) if (ROOT / rel).is_file()]
        if not existing:
            missing.append(claim["claim_id"])
            continue
        source = existing[0]
        rel = source.relative_to(ROOT).as_posix()
        metrics = special_metrics(claim["claim_id"])
        metrics["artifact_sha256"] = digest(source)
        metrics["artifact_bytes"] = source.stat().st_size
        results.append({
            "claim_id": claim["claim_id"], "dataset": claim.get("dataset", "n/a"),
            "protocol": claim.get("protocol", "n/a"), "status": claim["status"],
            "source_artifact": rel, "config_hash": None,
            "quantile_rule": claim.get("calibration_method"), "metrics": metrics,
        })
    if missing:
        raise FileNotFoundError("promoted empirical claims without an artifact: " + ", ".join(missing))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip() or None
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True
        ).stdout.strip()
    )
    payload = {
        "schema_version": "kbound-result-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
        "runtime": {
            "builder": "docs/research/kbound/scripts/build_result_manifest.py",
            "worktree_dirty": dirty,
        },
        "results": results,
        "reconciliation_source": {
            "canonical_panel": RECONCILED.relative_to(ROOT).as_posix(),
            "canonical_panel_sha256": digest(RECONCILED) if RECONCILED.is_file() else None,
            "source_manifest": SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": digest(SOURCE_MANIFEST) if SOURCE_MANIFEST.is_file() else None,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} ({len(results)} promoted empirical claims)")


if __name__ == "__main__":
    main()
