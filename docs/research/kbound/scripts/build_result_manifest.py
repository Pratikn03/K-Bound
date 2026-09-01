#!/usr/bin/env python3
"""Build the canonical empirical-result manifest from the claim ledger.

The ledger controls wording/status. This manifest records one existing authoritative
artifact for every included empirical claim and adds bounded, machine-readable
metrics for canonical panels plus the separately sealed CCT-20 and So2Sat authorities.
Historical FMoW/PovertyMap diagnostics are inventoried separately and are never
treated as canonical-panel or headline evidence.
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
CCT20_RELEASE = ROOT / "docs/research/kbound/paper/generated/cct20_release_manifest.json"
SO2SAT_SELECTION = (
    ROOT
    / "experiments/kbound/results/so2sat_lcz42_prospective_v1/"
    "development_mps_bn_fix_v1/so2sat_candidate_selection.json"
)
FMOW_FINDINGS = (
    ROOT / "experiments/kbound/results/fmow_protocol_L_v1/VERIFIED_FINDINGS.json"
)
POVERTY_FINDINGS = (
    ROOT / "experiments/kbound/results/poverty_protocol_L_dev/VERIFIED_FINDINGS.json"
)
CI_CONVENTION = "baseline_regret_minus_kga_regret; positive values favor KGA"
CURRENT_CLUSTER_SCHEMA = "kbound-current-policy-cluster-inference-v3"
CURRENT_POLICY_BINDING_PATHS = {
    "policy": "kga/policy.py",
    "certificate": "kga/certificate.py",
    "numeric_validation": "kga/_validation.py",
    "preregistered_protocol": "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml",
}
FAMILY_FIELD = "retrospective_holm_over_six_prospectively_named_contrasts"
COMPARISON_P_FIELD = "p_value_retrospective_holm_six_prospectively_named_contrasts"
GATE_REJECTS_BOTH_FIELD = "both_sign_flip_tests_survive_retrospective_six_contrast_holm_0_05"
GATE_PASS_FIELD = "retrospective_six_contrast_cluster_sensitivity_pass"
PHASE1_PROVENANCE = (
    "docs/research/kbound/audits/phase1_provenance_2026_08_27/provenance_seal.json"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validated_separate_authorities(ledger: dict) -> dict:
    """Validate and return the non-panel CCT-20 and So2Sat authorities."""

    reconciliation = ledger.get("reconciliation_source", {})
    authorities = reconciliation.get("separate_receipt_linked_authorities", {})
    expected = {
        "cct20": (
            "KB-CLAIM-051",
            CCT20_RELEASE,
            "SAFE_UTILITY_ONLY",
        ),
        "so2sat_development": (
            "KB-CLAIM-052",
            SO2SAT_SELECTION,
            "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL",
        ),
    }
    if set(authorities) != set(expected):
        raise ValueError("release ledger must name exactly the CCT-20 and So2Sat authorities")
    for name, (claim_id, path, verdict) in expected.items():
        row = authorities[name]
        if (
            row.get("claim_id") != claim_id
            or row.get("artifact") != path.relative_to(ROOT).as_posix()
            or row.get("artifact_sha256") != digest(path)
            or row.get("verdict") != verdict
        ):
            raise ValueError(f"release ledger has a stale or malformed {name} authority")
    return authorities


def validated_historical_diagnostic_authorities(ledger: dict) -> dict:
    """Validate non-panel FMoW/PovertyMap inventory authorities."""

    reconciliation = ledger.get("reconciliation_source", {})
    authorities = reconciliation.get("separate_historical_diagnostic_authorities", {})
    expected = {
        "fmow_protocol_l": (
            "KB-CLAIM-054",
            FMOW_FINDINGS,
            "not-cleared",
        ),
        "poverty_protocol_l_development": (
            "KB-CLAIM-055",
            POVERTY_FINDINGS,
            "dev-screen-stop",
        ),
    }
    if set(authorities) != set(expected):
        raise ValueError(
            "release ledger must name exactly the FMoW and PovertyMap historical diagnostics"
        )
    for name, (claim_id, path, verdict) in expected.items():
        row = authorities[name]
        if (
            row.get("claim_id") != claim_id
            or row.get("artifact") != path.relative_to(ROOT).as_posix()
            or row.get("artifact_sha256") != digest(path)
            or row.get("artifact_bytes") != path.stat().st_size
            or row.get("verdict") != verdict
            or row.get("canonical_panel_member") is not False
            or row.get("headline_promotion_eligible") is not False
        ):
            raise ValueError(f"release ledger has a stale or malformed {name} authority")
    poverty_authority = authorities["poverty_protocol_l_development"]
    if poverty_authority.get("held_out_evaluation_run") is not False:
        raise ValueError("PovertyMap development stop must not imply held-out evaluation")
    return authorities


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


def config_hash_status(claim_id: str) -> str:
    if claim_id == "KB-CLAIM-051":
        return (
            "Protocol, runtime, checkpoint, data, and artifact identities are recorded "
            "in the receipt-linked CCT-20 release manifest and "
            "KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml."
        )
    if claim_id == "KB-CLAIM-052":
        return (
            "Candidate configuration, environment, source-checkpoint, protocol, "
            "runtime-amendment, bundle, and receipt hashes are stored in the selection "
            "and gate-fit artifacts."
        )
    return (
        "No single execution-time configuration hash can truthfully represent this "
        "aggregate or historical claim. Recoverable serialized configuration hashes, "
        "protocol-file hashes, and unresolved gaps are recorded in the Phase-1 "
        "provenance seal."
    )


def _validated_current_policy_bindings(bindings: object) -> dict:
    """Require the complete canonical binding set, paths, and live file hashes."""
    if not isinstance(bindings, dict) or set(bindings) != set(CURRENT_POLICY_BINDING_PATHS):
        raise ValueError(
            "current-policy family sensitivity must bind exactly "
            + ", ".join(CURRENT_POLICY_BINDING_PATHS)
        )
    for name, relative_path in CURRENT_POLICY_BINDING_PATHS.items():
        binding = bindings[name]
        if (
            not isinstance(binding, dict)
            or binding.get("path") != relative_path
            or not isinstance(binding.get("sha256"), str)
        ):
            raise ValueError(f"current-policy family sensitivity {name} binding has invalid path/hash metadata")
        path = ROOT / relative_path
        if not path.is_file() or digest(path) != binding["sha256"]:
            raise ValueError(f"current-policy family sensitivity {name} binding is stale")
    return bindings


def current_cluster_metrics() -> dict:
    raw = json.loads(CURRENT_CLUSTER.read_text())
    if raw.get("schema") != CURRENT_CLUSTER_SCHEMA:
        raise ValueError("current-policy family sensitivity must use the v3 schema")
    if raw.get("contrast_convention") != CI_CONVENTION:
        raise ValueError("current-policy family sensitivity uses the wrong contrast convention")
    analysis_path = ROOT / raw["analysis_script"]
    if not analysis_path.is_file() or digest(analysis_path) != raw["analysis_script_sha256"]:
        raise ValueError("current-policy family sensitivity analysis-script binding is stale")
    family = raw.get(FAMILY_FIELD, {})
    if family.get("family_size") != 6 or family.get("alpha") != 0.05:
        raise ValueError(
            "current-policy family sensitivity lacks retrospective Holm over the six "
            "prospectively named contrasts"
        )

    bindings = _validated_current_policy_bindings(raw.get("live_code_bindings"))

    candidates = {}
    for candidate in ("tent", "eata", "sar"):
        row = raw["candidates"][candidate]
        if row["gate"][GATE_PASS_FIELD] is not False:
            raise ValueError(f"unexpected retrospective family-sensitivity pass for {candidate}")
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
            "retrospective_six_contrast_holm_rejects_both": row["gate"][
                GATE_REJECTS_BOTH_FIELD
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
                    COMPARISON_P_FIELD: comparison[COMPARISON_P_FIELD],
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
        "live_code_bindings": bindings,
        "convention": CI_CONVENTION,
        "inference": raw["inference"],
        "claim_boundary": raw["claim_boundary"],
        FAMILY_FIELD: family,
        "candidates": candidates,
        "release_interpretation": (
            "Tent has positive ordinary family-bootstrap intervals against both fixed policies. "
            "Its within-candidate two-contrast Holm result is post hoc; the retrospectively "
            "Holm-adjusted p-values over the six prospectively named contrasts are 0.09375 "
            "against both baselines. The exact-rank replay, sign-flip tests, and Holm analysis "
            "are retrospective and non-confirmatory; EATA and SAR also fail the gate."
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
        replication = panels["officehome"]["test_stream_seed_replication"]
        return {
            **score_metrics(score),
            "a7_status": primary["calibration"]["a7_status"],
            "test_stream_seed_replication": {
                **score_metrics(replication["exact_rank_transfer_score"]),
                "a7_status": replication["calibration"]["a7_status"],
                "independent_checkpoint_inference": "not_available",
                "headline_promotion_eligible": False,
                "claim_scope": (
                    "separate run-seed replication conditional on archived checkpoint "
                    "identities; subordinate to the KB-CLAIM-020 primary result"
                ),
            },
        }
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
    if claim_id == "KB-CLAIM-051":
        raw = json.loads(CCT20_RELEASE.read_text())
        verdict = raw.get("verdict", {})
        if (
            raw.get("schema") != "kbound_cct20_release_manifest_v1"
            or raw.get("status") != "RELEASE_COMPLETE"
            or verdict.get("code") != "SAFE_UTILITY_ONLY"
            or verdict.get("safe_utility_passes") is not True
            or verdict.get("protocol_strong_success") is not False
        ):
            raise ValueError("CCT-20 release authority is not the sealed safe-utility-only result")
        counts = raw.get("action_exposure", {}).get("counts", {})
        if counts != {"ABSTAIN": 1, "ADAPT": 0, "FREEZE": 44}:
            raise ValueError("CCT-20 release action counts drifted")
        return {
            "n_decisions": raw["design"]["cell_count"],
            "checkpoint_count": raw["design"]["checkpoint_count"],
            "location_cluster_count": raw["design"]["location_cluster_count"],
            "decision_counts": counts,
            "adaptation_effect_counts": {
                "helpful": raw["adaptation_effect_mix"]["helpful_cells_strictly_positive"],
                "harmful": raw["adaptation_effect_mix"]["harmful_cells_strictly_negative"],
                "zero": raw["adaptation_effect_mix"]["neutral_cells_exactly_zero"],
            },
            "point_beats_both": False,
            "ci_robust_beats_both": False,
            "verdict": verdict["code"],
            "safe_utility_passes": True,
            "comparison_inference": {
                "convention": CI_CONVENTION,
                "unit": (
                    "camera location, averaging the five independent checkpoints "
                    "within each location"
                ),
                "multiplicity": "separate locked two-comparison Holm family",
                "comparisons": {
                    baseline.removeprefix("versus_"): {
                        "point": row["point_estimate"],
                        "simultaneous_bonferroni_97_5_ci": row[
                            "simultaneous_bonferroni_97_5_ci"
                        ],
                        "holm_p": row["holm_adjusted_p"],
                    }
                    for baseline, row in raw["primary_comparisons"].items()
                },
                "current_policy_authority": True,
            },
            "headline_promotion_eligible": False,
            "prospective_disclosure": raw["prospective_disclosure"],
        }
    if claim_id == "KB-CLAIM-052":
        raw = json.loads(SO2SAT_SELECTION.read_text())
        candidate_names = {
            "tent_adam_bn_affine_probe_transfer_v1": "tent",
            "sar_sam_bn_affine_probe_transfer_v1": "sar",
        }
        candidate_ids = raw.get("candidate_ids", [])
        candidate_summaries = raw.get("candidate_summaries", {})
        gate_fit_cities = raw.get("study_binding", {}).get("gate_fit_cities", [])
        if (
            raw.get("schema") != "kbound_so2sat_adapter_candidate_selection_v1"
            or raw.get("status") != "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
            or raw.get("selected_candidate_id") is not None
            or set(candidate_ids) != set(candidate_names)
            or set(candidate_summaries) != set(candidate_names)
            or not gate_fit_cities
            or len(set(gate_fit_cities)) != len(gate_fit_cities)
            or raw.get("gate_cal_rows_read_before_selection") != 0
            or raw.get("target_pixels_read") != 0
            or raw.get("target_labels_read") != 0
            or raw.get("target_inputs") != []
        ):
            raise ValueError("So2Sat selection authority does not preserve the no-candidate stop")
        candidates = {}
        study_shapes = set()
        for candidate_id, summary in sorted(candidate_summaries.items()):
            feasibility = summary["feasibility"]
            if feasibility.get("feasible") is not False:
                raise ValueError(f"So2Sat candidate unexpectedly became feasible: {candidate_id}")
            short_name = candidate_names.get(candidate_id)
            if short_name is None:
                raise ValueError(f"unknown So2Sat candidate ID: {candidate_id}")
            city_count = feasibility.get("city_count")
            checkpoint_count = feasibility.get("checkpoint_count")
            cell_count = feasibility.get("cell_count")
            if (
                city_count != len(gate_fit_cities)
                or set(feasibility.get("city_mean_benefit", {})) != set(gate_fit_cities)
                or not isinstance(checkpoint_count, int)
                or checkpoint_count <= 0
                or cell_count != city_count * checkpoint_count
            ):
                raise ValueError(f"So2Sat study shape drifted for candidate: {candidate_id}")
            study_shapes.add((city_count, checkpoint_count, cell_count))
            candidates[short_name] = {
                "feasible": False,
                "helpful_cities": len(feasibility["helpful_cities"]),
                "harmful_cities": len(feasibility["harmful_cities"]),
                "always_adapt_accuracy": feasibility["always_adapt_accuracy"],
                "always_freeze_accuracy": feasibility["always_freeze_accuracy"],
                "loco_routed_accuracy": feasibility["loco_routed_accuracy"],
                "oracle_routing_gap": feasibility["oracle_routing_gap"],
                "loco_sign_accuracy": feasibility["loco_sign_accuracy"],
                "loco_gain_over_best_fixed": feasibility[
                    "loco_routed_gain_over_best_fixed"
                ],
            }
        if len(study_shapes) != 1:
            raise ValueError("So2Sat candidates do not share one development study shape")
        development_city_count, checkpoint_count, development_cell_count = study_shapes.pop()
        return {
            "verdict": raw["status"],
            "selected_candidate_id": None,
            "development_city_count": development_city_count,
            "checkpoint_count": checkpoint_count,
            "candidate_count": len(candidate_ids),
            "development_cell_count_per_candidate": development_cell_count,
            "target_score": None,
            "target_access": {
                "target_inputs": [],
                "target_pixels_read": 0,
                "target_labels_read": 0,
                "gate_cal_rows_read_before_selection": 0,
            },
            "candidate_summaries": candidates,
            "point_beats_both": False,
            "ci_robust_beats_both": False,
            "headline_promotion_eligible": False,
        }
    if claim_id == "KB-CLAIM-053":
        diagnostic = panels["camelyon17"]["b_v2_diagnostic"]
        score = diagnostic["panel"]["candidates"]["sar"]
        if (
            diagnostic["headline_promotion"].get("eligible") is not False
            or score.get("point_beats_both") is not True
            or score.get("seed_inference", {}).get("ci_robust_beats_both") is not False
        ):
            raise ValueError("Camelyon17 B-v2 SAR diagnostic scope drifted")
        return {
            **score_metrics(score),
            "n_run_seeds": len(diagnostic["panel"]["seeds"]),
            "within_seed_diagnostic": True,
            "untouched_target_domain_evaluation": False,
            "independent_checkpoint_identities_recorded": False,
            "headline_promotion_eligible": False,
            "claim_scope": diagnostic["claim_scope"],
        }
    if claim_id == "KB-CLAIM-054":
        raw = json.loads(FMOW_FINDINGS.read_text())
        score = raw.get("analyze_F", {})
        if (
            raw.get("protocol") != "FMOW_PROTOCOL_L_v1"
            or raw.get("dataset") != "wilds-fmow"
            or raw.get("verdict") != "not-cleared"
            or raw.get("headline") is not False
            or score.get("candidate") != "sar_online"
            or score.get("n_test") != 180
            or score.get("beats_both") is not False
        ):
            raise ValueError("FMoW historical diagnostic is stale or malformed")
        return {
            "candidate": score["candidate"],
            "n_decisions": score["n_test"],
            "regret_kga": score["regret_kga"],
            "regret_adapt": score["regret_adapt"],
            "regret_freeze": score["regret_freeze"],
            "false_adapt_conditional_rate": score["false_adapt"],
            "false_adapt_unconditional_rate": None,
            "point_beats_both": score["beats_both"],
            "ci_robust_beats_both": False,
            "verdict": raw["verdict"],
            "canonical_panel_member": False,
            "headline_promotion_eligible": False,
        }
    if claim_id == "KB-CLAIM-055":
        raw = json.loads(POVERTY_FINDINGS.read_text())
        screen = raw.get("dev_screen", {})
        if (
            raw.get("protocol") != "POVERTY_PROTOCOL_L_v1"
            or raw.get("dataset") != "wilds-poverty"
            or raw.get("verdict") != "dev-screen-stop"
            or raw.get("headline") is not False
            or screen.get("screen") != "STOP"
            or screen.get("reason") != "harm_AUC below 0.65 gate"
            or raw.get("held_out_val_test") != "not run per pre-registration"
        ):
            raise ValueError("PovertyMap development diagnostic is stale or malformed")
        return {
            "development_screen": screen["screen"],
            "development_harm_auc": screen["harm_AUC"],
            "development_harmful_rate": screen["harmful_rate"],
            "stop_reason": screen["reason"],
            "held_out_evaluation_run": False,
            "target_score": None,
            "point_beats_both": None,
            "ci_robust_beats_both": None,
            "verdict": raw["verdict"],
            "canonical_panel_member": False,
            "headline_promotion_eligible": False,
        }
    return {}


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    separate_authorities = validated_separate_authorities(ledger)
    historical_diagnostic_authorities = validated_historical_diagnostic_authorities(ledger)
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
            "config_hash_status": config_hash_status(claim["claim_id"]),
            "quantile_rule": claim.get("calibration_method"), "metrics": metrics,
        })
    if missing:
        raise FileNotFoundError(
            "release-manifest empirical claims without an artifact: " + ", ".join(missing)
        )
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
            "separate_receipt_linked_authorities": separate_authorities,
            "separate_historical_diagnostic_authorities": historical_diagnostic_authorities,
            "multiplicity_status": ledger["reconciliation_source"][
                "multiplicity_status"
            ],
            "phase1_provenance_seal": PHASE1_PROVENANCE,
            "identity_scope": (
                "Retrospective hashes bind current or archived bytes only; they are not "
                "silently treated as missing historical execution identities."
            ),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} ({len(results)} empirical evidence entries)")


if __name__ == "__main__":
    main()
