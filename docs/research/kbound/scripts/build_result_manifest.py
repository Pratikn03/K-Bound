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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def special_metrics(claim_id: str) -> dict:
    if claim_id == "KB-CLAIM-041":
        p = ROOT / "experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json"
        d = json.loads(p.read_text())
        domains = list(d["per_domain"].values())
        n = sum(sum(x["n_test_cells_per_seed"]) for x in domains)
        return {
            "n_seeds": d["n_seeds"], "n_decisions": n,
            "regret_kga_mean_across_domains": sum(x["regret_K_Bound"]["mean"] for x in domains) / len(domains),
            "regret_adapt_mean_across_domains": sum(x["regret_always_adapt"]["mean"] for x in domains) / len(domains),
            "regret_freeze_mean_across_domains": sum(x["regret_always_freeze"]["mean"] for x in domains) / len(domains),
            "false_adapt_reported_rate": sum(x["FA_u"]["mean"] for x in domains) / len(domains),
            "false_adapt_count_status": "not_retained", "beats_both_promoted": False,
        }
    if claim_id == "KB-CLAIM-042":
        p = ROOT / "experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json"
        d = json.loads(p.read_text())
        vals = list(d["candidates"].values())
        return {
            "n_seeds": d["n_seeds"], "n_backbones": len(vals),
            "conditions_per_backbone_seed": d["n_conditions"],
            "regret_kga_mean_across_backbones": sum(x["kga_mean_regret"] for x in vals) / len(vals),
            "regret_adapt_mean_across_backbones": sum(x["adapt_mean_regret"] for x in vals) / len(vals),
            "regret_freeze_mean_across_backbones": sum(x["freeze_mean_regret"] for x in vals) / len(vals),
            "false_adapt_num": sum(x["false_adapt_num"] for x in vals),
            "false_adapt_den": sum(x["false_adapt_den"] for x in vals),
            "beats_both_candidates": sum(bool(v) for v in d["beats_both_by_candidate"].values()),
        }
    return {}


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    results = []
    missing = []
    for claim in ledger["claims"]:
        if claim.get("claim_type") != "empirical" or claim.get("status") not in {"supported", "no-harm"}:
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
    payload = {
        "schema_version": "kbound-result-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
        "runtime": {"builder": "docs/research/kbound/scripts/build_result_manifest.py"},
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} ({len(results)} promoted empirical claims)")


if __name__ == "__main__":
    main()
