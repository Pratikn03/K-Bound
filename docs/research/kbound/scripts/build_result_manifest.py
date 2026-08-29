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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LEDGER = ROOT / "docs/research/kbound/claim_ledger.json"
OUT = ROOT / "docs/research/kbound/RESULT_MANIFEST.json"
RECONCILED = ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
SOURCE_MANIFEST = ROOT / "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
CURRENT_CLUSTER = (
    ROOT
    / "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json"
)
HEADTOHEAD = (
    ROOT
    / "experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json"
)
CI_CONVENTION = "baseline_regret_minus_kga_regret; positive values favor KGA"
PHASE1_PROVENANCE = (
    "docs/research/kbound/audits/phase1_provenance_2026_08_27/provenance_seal.json"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_metrics(score: dict) -> dict:
    bootstrap = score.get("seed_inference", {}).get("descriptive_seed_bootstrap")
    comparisons = {}
    for baseline in ("always_adapt", "always_freeze"):
        comparisons[baseline] = {
            "point": score["regret"][baseline] - score["regret"]["kga"],
            "ci95": bootstrap["gaps"][baseline]["ci95"] if bootstrap else None,
        }
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
        "point_beats_both": score["point_beats_both"],
        "ci_robust_beats_both": score.get("seed_inference", {}).get(
            "ci_robust_beats_both", False
        ),
        "comparison_inference": {
            "convention": CI_CONVENTION,
            "comparisons": comparisons,
            "unit": bootstrap.get("unit") if bootstrap else None,
            "current_policy_authority": True,
        },
    }


def current_cluster_metrics() -> dict:
    raw = json.loads(CURRENT_CLUSTER.read_text())
    if raw.get("schema") != "kbound-current-policy-cluster-inference-v2":
        raise ValueError("current-policy family sensitivity must use the v2 schema")
    if raw.get("contrast_convention") != CI_CONVENTION:
        raise ValueError("current-policy family sensitivity uses the wrong contrast convention")
    analysis_path = ROOT / raw["analysis_script"]
    if not analysis_path.is_file() or digest(analysis_path) != raw["analysis_script_sha256"]:
        raise ValueError("current-policy family sensitivity analysis-script binding is stale")
    family = raw.get("preregistered_six_comparison_holm", {})
    if family.get("family_size") != 6 or family.get("alpha") != 0.05:
        raise ValueError("current-policy family sensitivity lacks the preregistered six-way Holm family")

    for name, binding in raw.get("live_code_bindings", {}).items():
        path = ROOT / binding["path"]
        if not path.is_file() or digest(path) != binding["sha256"]:
            raise ValueError(f"current-policy family sensitivity {name} binding is stale")

    candidates = {}
    for candidate in ("tent", "eata", "sar"):
        row = raw["candidates"][candidate]
        if row["gate"]["preregistered_six_comparison_cluster_sensitivity_pass"] is not False:
            raise ValueError(f"unexpected preregistered family-sensitivity pass for {candidate}")
        candidates[candidate] = {
            "inference_unit": row["grain"]["inference_unit"],
            "n_inference_units": row["grain"]["n_inference_units"],
            "nested_repetitions": row["grain"]["nested_repetitions"],
            "pointwise_family_intervals_positive_vs_both": row["gate"][
                "both_pointwise_95pct_cluster_bootstrap_intervals_positive"
            ],
            "within_candidate_posthoc_holm_rejects_both": row["gate"][
                "both_one_sided_sign_flip_tests_survive_within_candidate_posthoc_holm_0.05"
            ],
            "preregistered_six_comparison_holm_rejects_both": row["gate"][
                "both_sign_flip_tests_survive_preregistered_six_comparison_holm_0.05"
            ],
            "comparisons": {
                baseline: {
                    "point": comparison["point"],
                    "ci95_unadjusted_family_bootstrap": comparison["ci"],
                    "family_effects": comparison["family_effects"],
                    "p_value_one_sided_exact_sign_flip": comparison[
                        "p_value_one_sided_exact_sign_flip"
                    ],
                    "p_value_holm_within_candidate_posthoc": comparison[
                        "p_value_holm_within_candidate_posthoc"
                    ],
                    "p_value_holm_preregistered_six_comparison_family": comparison[
                        "p_value_holm_preregistered_six_comparison_family"
                    ],
                }
                for baseline, comparison in row["comparisons"].items()
            },
        }

    return {
        "status": "retrospective_current_policy_family_sensitivity",
        "confirmatory": False,
        "artifact_path": CURRENT_CLUSTER.relative_to(ROOT).as_posix(),
        "artifact_sha256": digest(CURRENT_CLUSTER),
        "artifact_bytes": CURRENT_CLUSTER.stat().st_size,
        "schema": raw["schema"],
        "generated_utc": raw["generated_utc"],
        "git_head": raw.get("git_head"),
        "analysis_script": raw["analysis_script"],
        "analysis_script_sha256": raw["analysis_script_sha256"],
        "runtime": raw["runtime"],
        "live_code_bindings": raw["live_code_bindings"],
        "convention": CI_CONVENTION,
        "inference": raw["inference"],
        "claim_boundary": raw["claim_boundary"],
        "preregistered_six_comparison_holm": family,
        "candidates": candidates,
        "release_interpretation": (
            "Tent has positive ordinary family-bootstrap intervals against both fixed policies. "
            "Its within-candidate two-contrast Holm result is post hoc; the preregistered six-way "
            "Holm p-values are 0.09375 against both baselines, so no cluster-robust or "
            "confirmatory win is promoted. EATA and SAR also fail the preregistered gate."
        ),
    }


def special_metrics(claim_id: str) -> dict:
    if not RECONCILED.is_file():
        return {}
    panels = json.loads(RECONCILED.read_text())["panels"]
    if claim_id == "KB-CLAIM-010":
        cifar = panels["cifar10c"]["panel"]
        return {
            "candidates": {
                candidate: score_metrics(cifar["candidates"][candidate])
                for candidate in ("tent", "eata", "sar")
            },
            "current_policy_family_sensitivity": current_cluster_metrics(),
            "independent_checkpoint_inference": "not_available",
        }
    if claim_id == "KB-CLAIM-011":
        score = panels["imagenetc"]["panel"]["candidates"]["sar"]
        return {**score_metrics(score), "n_seeds": 5}
    if claim_id == "KB-CLAIM-021":
        return {
            "numeric_release_eligible": False,
            "release_disposition": "WITHHELD_INVALID_METRIC_CONTRACT_DIAGNOSTIC_ONLY",
            "withheld_reason": (
                "The archived scorer does not implement the official WILDS label-present "
                "macro-F1 contract; a population-sealed official-metric rerun is required."
            ),
        }
    if claim_id == "KB-CLAIM-020":
        primary = panels["officehome"]["primary"]
        score = primary["exact_rank_transfer_score"]
        return {**score_metrics(score), "a7_status": primary["calibration"]["a7_status"]}
    if claim_id == "KB-CLAIM-026":
        raw = json.loads(HEADTOHEAD.read_text())
        comparisons = []
        for row in raw["headtohead"]["comparisons"]:
            comparisons.append(
                {
                    "competitor": row["competitor"],
                    "mean_gap_baseline_minus_kga": -row["mean_diff_kga_minus_competitor"],
                    "ci95_baseline_minus_kga": [-row["ci95_hi"], -row["ci95_lo"]],
                    "p_raw": row["p_raw"],
                    "p_holm": row["p_holm"],
                    "holm_applies_to": "p_value_only",
                    "ci_adjustment": "unadjusted paired percentile interval",
                }
            )
        current = panels["cifar10c"]["panel"]["candidates"]["tent"]
        return {
            "numeric_release_eligible": False,
            "release_eligible_win": False,
            "policy_synchronized": False,
            "current_policy_authority": False,
            "archived_policy_recomputed_kga": bool(raw.get("recompute_kga")),
            "comparison_ci_convention": CI_CONVENTION,
            "archived_comparisons": comparisons,
            "current_exact_rank_reference": score_metrics(current),
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
    if claim_id == "KB-CLAIM-047":
        return {
            **score_metrics(panels["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"]),
            "headline_promotion_eligible": False,
            "opened_target": True,
        }
    if claim_id == "KB-CLAIM-048":
        return {
            **score_metrics(panels["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"]),
            "headline_promotion_eligible": False,
            "model_seed_robustness": panels["rxrx1"]["model_seed_robustness"]["aggregate"],
        }
    if claim_id == "KB-CLAIM-049":
        return {
            **score_metrics(panels["cifar101"]["replay"]["exact_rank_transfer_score"]),
            "headline_promotion_eligible": False,
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
            "config_hash_status": (
                "No single execution-time configuration hash can truthfully represent this "
                "aggregate or historical claim. Recoverable serialized configuration hashes, "
                "protocol-file hashes, and unresolved gaps are recorded in the Phase-1 "
                "provenance seal."
            ),
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
        "created_at": f"{ledger.get('generated_at', 'unknown')}T00:00:00Z", "code_commit": sha,
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
            "comparison_ci_convention": CI_CONVENTION,
            "current_policy_family_sensitivity": {
                "artifact": CURRENT_CLUSTER.relative_to(ROOT).as_posix(),
                "artifact_sha256": digest(CURRENT_CLUSTER),
                "artifact_bytes": CURRENT_CLUSTER.stat().st_size,
                "status": "retrospective_not_preregistered_significant",
            },
            "phase1_provenance_seal": PHASE1_PROVENANCE,
            "identity_scope": (
                "Retrospective hashes bind current or archived bytes only; they are not "
                "silently treated as missing historical execution identities."
            ),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} ({len(results)} promoted empirical claims)")


if __name__ == "__main__":
    main()
