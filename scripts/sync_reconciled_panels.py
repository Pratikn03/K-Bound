#!/usr/bin/env python3
"""Propagate source-replayed panel results into the paper's structured manifests.

The numerical source of truth is
``experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json``.
This script intentionally updates structured JSON only; manuscript prose is audited
separately so a numeric refresh cannot silently change the claim scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
SOURCE_MANIFEST = ROOT / "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
TABLE_PATH = ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json"
LEDGER_PATH = ROOT / "docs/research/kbound/claim_ledger.json"
FRONTIER_PATH = ROOT / "experiments/kbound/frontier_sweep_v1/decision_value_results.json"
UNIFORM_VERDICTS_PATH = ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json"
DECISION_METRICS_PATH = (
    ROOT / "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json"
)
HISTORICAL_CLUSTER_PATH = ROOT / "research_lock/CIFAR10C_TENT_CORRUPTION_UNIT_CI_v1.json"
CURRENT_CLUSTER_PATH = (
    ROOT
    / "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json"
)
CURRENT_CLUSTER_TABLE_PATH = (
    ROOT / "docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex"
)
HISTORICAL_HEADTOHEAD_PATH = (
    ROOT
    / "experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json"
)

CI_CONVENTION = "baseline_regret_minus_kga_regret; positive values favor KGA"
CURRENT_POLICY_STATUS = "current_policy_exact_rank_replay"
HISTORICAL_POLICY_STATUS = "historical_policy_only"
CURRENT_CLUSTER_STATUS = "retrospective_current_policy_family_sensitivity"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regret(score: dict[str, Any]) -> list[float]:
    values = score["regret"]
    return [values["kga"], values["always_adapt"], values["always_freeze"]]


def _decision_counts(score: dict[str, Any]) -> dict[str, int]:
    return {
        "ADAPT": score["adapt_count"],
        "FREEZE": score["freeze_count"],
        "ABSTAIN": score["abstain_count"],
    }


def _ci(score: dict[str, Any], baseline: str) -> list[float] | None:
    inference = score.get("seed_inference", {})
    bootstrap = inference.get("descriptive_seed_bootstrap") or inference.get("paired_seed_bootstrap")
    return bootstrap["gaps"][baseline]["ci95"] if bootstrap else None


def _comparison_inference(score: dict[str, Any]) -> dict[str, Any]:
    """Expose one unambiguous comparison direction on every current score row."""

    inference = score.get("seed_inference", {})
    bootstrap = inference.get("descriptive_seed_bootstrap") or inference.get("paired_seed_bootstrap")
    comparisons: dict[str, Any] = {}
    for baseline in ("always_adapt", "always_freeze"):
        gap = score["regret"][baseline] - score["regret"]["kga"]
        comparisons[baseline] = {
            "point": gap,
            "ci95": bootstrap["gaps"][baseline]["ci95"] if bootstrap else None,
        }
    return {
        "convention": CI_CONVENTION,
        "comparisons": comparisons,
        "method": "paired percentile bootstrap of run-seed means" if bootstrap else None,
        "unit": bootstrap.get("unit") if bootstrap else None,
        "replicates": bootstrap.get("replicates") if bootstrap else None,
        "random_seed": bootstrap.get("random_seed") if bootstrap else None,
        "current_policy_authority": True,
        "supports_ci_robust_beats_both": bool(inference.get("ci_robust_beats_both", False)),
    }


def _drop_stale_ci_aliases(row: dict[str, Any]) -> None:
    for key in (
        "ci_vs_adapt",
        "ci_vs_freeze",
        "gap_vs_adapt_ci95",
        "gap_vs_freeze_ci95",
        "gap_baseline_minus_kga_ci95_seed",
        "gap_kga_minus_adapt_ci95_seedavg27",
        "gap_kga_minus_freeze_ci95_seedavg27",
        "gap_kga_minus_freeze_ci95_iid135",
        "gap_kga_minus_adapt_ci95",
        "gap_kga_minus_freeze_ci95",
        "gap_vs_adapt_ci95_seedavg27",
        "gap_vs_freeze_ci95_seedavg27",
        "gap_vs_adapt_ci95_iid135",
        "gap_vs_freeze_ci95_iid135",
    ):
        row.pop(key, None)


def _normalized_historical_cluster() -> dict[str, Any]:
    """Retain the old Tent cluster result without treating it as current-policy evidence."""

    raw = _load(HISTORICAL_CLUSTER_PATH)

    def normalized_arm(arm: dict[str, Any]) -> dict[str, Any]:
        comparisons: dict[str, Any] = {}
        for baseline, old_key in (
            ("always_adapt", "gap_vs_adapt"),
            ("always_freeze", "gap_vs_freeze"),
        ):
            old = arm[old_key]
            lo, hi = old["ci95"]
            comparisons[baseline] = {
                "point": -old["point"],
                "ci95": [-hi, -lo],
                "excludes_zero": bool(old["excludes_zero"]),
            }
        return {
            "epsilon_mean": arm["eps_mean"],
            "adapt_rate_mean": arm["adapt_rate_mean"],
            "false_adapts_total": arm["false_adapts_total"],
            "comparisons": comparisons,
        }

    return {
        "status": HISTORICAL_POLICY_STATUS,
        "policy_synchronized": False,
        "current_policy_authority": False,
        "release_eligible": False,
        "convention": CI_CONVENTION,
        "artifact_path": HISTORICAL_CLUSTER_PATH.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256(HISTORICAL_CLUSTER_PATH),
        "artifact_bytes": HISTORICAL_CLUSTER_PATH.stat().st_size,
        "historical_method_caveat": (
            "The archived artifact used an earlier KGA policy/radius implementation, including "
            "a clamped exact-rank rule. Its values are retained for audit and cannot support a "
            "current-policy Tent claim."
        ),
        "as_shipped_cell_out": normalized_arm(raw["arms"]["as_shipped_cell_out"]),
        "leave_one_corruption_out": normalized_arm(raw["arms"]["leave_one_corruption_out"]),
    }


def _normalized_current_cluster(raw: dict[str, Any], candidate: str) -> dict[str, Any]:
    """Expose the corrected family sensitivity without promoting a confirmatory win."""

    if raw.get("schema") != "kbound-current-policy-cluster-inference-v2":
        raise ValueError("current-policy cluster artifact must use the v2 schema")
    if raw.get("contrast_convention") != CI_CONVENTION:
        raise ValueError("current-policy cluster artifact uses the wrong contrast convention")

    family = raw.get("preregistered_six_comparison_holm", {})
    if family.get("family_size") != 6 or family.get("alpha") != 0.05:
        raise ValueError("current-policy cluster artifact must expose the preregistered six-way Holm family")

    bindings = raw.get("live_code_bindings", {})
    required_bindings = ("policy", "certificate", "preregistered_protocol")
    for name in required_bindings:
        binding = bindings.get(name, {})
        path = binding.get("path")
        expected_hash = binding.get("sha256")
        if not isinstance(path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"current-policy cluster artifact is missing the {name} binding")
        bound_path = ROOT / path
        if not bound_path.is_file() or _sha256(bound_path) != expected_hash:
            raise ValueError(f"current-policy cluster {name} binding does not match the live file")

    row = raw.get("candidates", {}).get(candidate)
    if not isinstance(row, dict):
        raise ValueError(f"current-policy cluster artifact is missing candidate {candidate!r}")
    grain = row.get("grain", {})
    if grain.get("inference_unit") != "corruption_family" or grain.get("n_inference_units") != 6:
        raise ValueError(f"current-policy cluster candidate {candidate!r} has the wrong inference unit")
    if row.get("gate", {}).get("preregistered_six_comparison_cluster_sensitivity_pass") is not False:
        raise ValueError(f"current-policy cluster candidate {candidate!r} must fail the preregistered gate")

    comparisons: dict[str, Any] = {}
    for baseline in ("always_adapt", "always_freeze"):
        comparison = row["comparisons"][baseline]
        comparisons[baseline] = {
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
            "preregistered_six_comparison_reject_at_0.05": comparison[
                "holm_preregistered_six_comparison_reject_at_0.05"
            ],
        }

    return {
        "status": CURRENT_CLUSTER_STATUS,
        "current_policy_authority": True,
        "retrospective": True,
        "confirmatory": False,
        "convention": CI_CONVENTION,
        "artifact_path": CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256(CURRENT_CLUSTER_PATH),
        "artifact_bytes": CURRENT_CLUSTER_PATH.stat().st_size,
        "artifact_schema": raw["schema"],
        "generated_utc": raw["generated_utc"],
        "git_head": raw.get("git_head"),
        "analysis_script": raw["analysis_script"],
        "analysis_script_sha256": raw["analysis_script_sha256"],
        "runtime": raw["runtime"],
        "live_code_bindings": bindings,
        "grain": grain,
        "comparisons": comparisons,
        "pointwise_family_intervals_positive_vs_both": row["gate"][
            "both_pointwise_95pct_cluster_bootstrap_intervals_positive"
        ],
        "within_candidate_posthoc_holm_rejects_both": row["gate"][
            "both_one_sided_sign_flip_tests_survive_within_candidate_posthoc_holm_0.05"
        ],
        "preregistered_six_comparison_holm_rejects_both": row["gate"][
            "both_sign_flip_tests_survive_preregistered_six_comparison_holm_0.05"
        ],
        "claim_boundary": raw["claim_boundary"],
        "claim_note": (
            "Retrospective sensitivity on six observed corruption families. Ordinary family-"
            "bootstrap intervals are unadjusted; the within-candidate two-contrast Holm values "
            "are post hoc, and the preregistered six-comparison Holm family fails for every "
            "candidate. This is not independent-checkpoint, prospective, natural-shift, or "
            "official-code POEM/AETTA evidence."
        ),
    }


def _write_current_cluster_table(raw: dict[str, Any]) -> None:
    """Write the compact paper table for the current-policy family sensitivity."""

    labels = {"tent": "Tent", "eata": "EATA", "sar": "SAR"}

    def interval(comparison: dict[str, Any]) -> str:
        lo, hi = comparison["ci"]
        return f"{comparison['point']:.5f} [{lo:.5f}, {hi:.5f}]"

    lines = [
        "% AUTO-GENERATED by scripts/sync_reconciled_panels.py. Do not edit by hand.",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Candidate & Adapt gap [95\% CI] & Freeze gap [95\% CI] & Six-way Holm $p$ (A/F) \\",
        r"\midrule",
    ]
    for candidate in ("tent", "eata", "sar"):
        row = raw["candidates"][candidate]
        adapt = row["comparisons"]["always_adapt"]
        freeze = row["comparisons"]["always_freeze"]
        lines.append(
            f"{labels[candidate]} & {interval(adapt)} & {interval(freeze)} & "
            f"{adapt['p_value_holm_preregistered_six_comparison_family']:.5f}/"
            f"{freeze['p_value_holm_preregistered_six_comparison_family']:.5f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    CURRENT_CLUSTER_TABLE_PATH.write_text("\n".join(lines) + "\n")


def _normalized_historical_headtohead(current_tent: dict[str, Any]) -> dict[str, Any]:
    """Expose archived port comparisons without reusing their stale KGA policy as current."""

    raw = _load(HISTORICAL_HEADTOHEAD_PATH)
    comparisons = []
    for old in raw["headtohead"]["comparisons"]:
        comparisons.append(
            {
                "competitor": old["competitor"],
                "mean_gap_baseline_minus_kga": -old["mean_diff_kga_minus_competitor"],
                "ci95_baseline_minus_kga": [-old["ci95_hi"], -old["ci95_lo"]],
                "p_raw": old["p_raw"],
                "p_holm": old["p_holm"],
                "holm_applies_to": "p_value_only",
                "ci_adjustment": "unadjusted paired percentile interval",
            }
        )
    return {
        "status": HISTORICAL_POLICY_STATUS,
        "verdict": "HISTORICAL_ONLY_CURRENT_POLICY_RECOMPUTATION_REQUIRED",
        "policy_synchronized": False,
        "current_policy_authority": False,
        "numeric_release_eligible": False,
        "release_eligible_win": False,
        "convention": CI_CONVENTION,
        "source": HISTORICAL_HEADTOHEAD_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(HISTORICAL_HEADTOHEAD_PATH),
        "source_bytes": HISTORICAL_HEADTOHEAD_PATH.stat().st_size,
        "archived_policy_recomputed_kga": bool(raw.get("recompute_kga")),
        "archived_policy_metrics": {
            "kga_regret": raw["policy_mean_regret"]["kga"],
            "always_adapt_regret": raw["policy_mean_regret"]["always_adapt"],
            "always_freeze_regret": raw["policy_mean_regret"]["always_freeze"],
            "poem_style_regret": raw["policy_mean_regret"]["poem"],
            "aetta_style_regret": raw["policy_mean_regret"]["aetta"],
            "kga_false_adapt": raw["policy_false_adapt_rate"]["kga"],
            "kga_decisive_rate": raw["policy_decisive_rate"]["kga"],
        },
        "archived_comparisons": comparisons,
        "current_exact_rank_reference": {
            "kga_regret": current_tent["regret"]["kga"],
            "always_adapt_regret": current_tent["regret"]["always_adapt"],
            "always_freeze_regret": current_tent["regret"]["always_freeze"],
            "decision_counts": _decision_counts(current_tent),
            "source": PANEL_PATH.relative_to(ROOT).as_posix(),
        },
        "caveat": (
            "POEM/AETTA-style values and their paired comparisons were computed against an earlier "
            "non-recomputed KGA policy. Holm adjustment applies to the archived p-values, not the "
            "confidence intervals. A current-policy head-to-head recomputation is required."
        ),
    }


def _zero_error_cp95(n: int) -> float | None:
    return 1.0 - math.pow(0.05, 1.0 / n) if n else None


def _cp95_upper(events: int, n: int) -> float | None:
    """One-sided 95% Clopper-Pearson upper bound for a binomial rate."""

    if n <= 0:
        return None
    if events <= 0:
        return _zero_error_cp95(n)
    if events >= n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(
            math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
            for i in range(events + 1)
        )

    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if cdf(mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _row_by_track(rows: list[dict[str, Any]], track: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("track") == track]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {track!r} row, found {len(matches)}")
    return matches[0]


def _withheld_iwildcam_fields() -> dict[str, Any]:
    return {
        "numeric_release_eligible": False,
        "release_disposition": "WITHHELD_INVALID_METRIC_CONTRACT_DIAGNOSTIC_ONLY",
        "withheld_reason": (
            "The archived scorer does not implement the official WILDS label-present macro-F1 "
            "contract, and no population-sealed official-metric rerun is available."
        ),
    }


def _claim(ledger: dict[str, Any], claim_id: str) -> dict[str, Any]:
    return next(row for row in ledger["claims"] if row["claim_id"] == claim_id)


def _sync_table(
    panel: dict[str, Any], table: dict[str, Any], current_cluster: dict[str, Any]
) -> None:
    panels = panel["panels"]
    tracks = table["tracks"]
    source = PANEL_PATH.relative_to(ROOT).as_posix()
    source_manifest = SOURCE_MANIFEST.relative_to(ROOT).as_posix()
    table["quantile_provenance"] = {
        "current_policy_rule": (
            "Unclamped exact empirical rank k=ceil((n+1)(1-alpha)); an infeasible rank "
            "forces abstention rather than clamping."
        ),
        "current_policy_scope": (
            "Current rows use the per-track calibration and quantile_rule recorded in the "
            "source-hashed canonical panel. Corruption panels use per-candidate/per-run-seed "
            "leave-one-condition-out residual calibration; natural panels use the recorded "
            "calibration-record leave-one-out transfer radius."
        ),
        "historical_artifacts": (
            "Earlier clamped/interpolated policies are not exposed as current rows. When kept "
            "for audit, they appear only inside explicitly historical, non-release-eligible "
            "blocks with artifact hashes and policy_synchronized=false."
        ),
    }

    imagenetc = panels["imagenetc"]["panel"]["candidates"]
    for candidate in ("tent", "eata", "sar"):
        score = imagenetc[candidate]
        key = f"imagenetc_{candidate}"
        row = tracks.setdefault(key, {})
        _drop_stale_ci_aliases(row)
        historical_seal = row.get("historical_audit_seal") or row.pop("seal", None)
        row.pop("seal", None)
        row.pop("in_pool_superseded", None)
        row.pop("source_correction", None)
        row.pop("reproduction", None)
        row.pop("regenerated", None)
        for stale in (
            "decisions_changed_by_leave_one_out_of_pool",
            "per_seed_beating_both",
            "per_seed_tying_freeze_bit_identically",
        ):
            row.pop(stale, None)
        row.update(
            {
                "regret": _regret(score),
                "false_adapt": score["fa_u"],
                "false_adapt_count": score["false_adapt_count"],
                "cp95_upper_fa_c": _cp95_upper(
                    score["false_adapt_count"], score["adapt_count"]
                ),
                "decision_counts": _decision_counts(score),
                "n_cells": score["n"],
                "seeds": panels["imagenetc"]["panel"]["seeds"],
                "quantile_rule": "per-candidate/per-seed exact-rank leave-one-condition-out residual calibration",
                "point_beats_both": score["point_beats_both"],
                "ci_robust_beats_both": score["seed_inference"]["ci_robust_beats_both"],
                "comparison_inference": _comparison_inference(score),
                "source": source,
                "source_manifest": source_manifest,
                "status": CURRENT_POLICY_STATUS,
                "current_policy_authority": True,
                "historical_audit_seal": historical_seal,
            }
        )
    tracks["imagenetc_sar"]["verdict"] = (
        "Pooled point estimate is below both fixed policies, with one false adaptation in 135 cells; "
        "the seed bootstrap touches zero on the freeze side, so CI-robust beats-both is not claimed."
    )

    cifar_panel = panels["cifar10c"]["panel"]
    for candidate in ("tent", "eata", "sar"):
        score = cifar_panel["candidates"][candidate]
        cluster_sensitivity = _normalized_current_cluster(current_cluster, candidate)
        if cluster_sensitivity["grain"]["n_records"] != score["n"]:
            raise ValueError(f"current-policy cluster record count disagrees for {candidate}")
        if current_cluster["candidates"][candidate]["decision_counts"] != _decision_counts(score):
            raise ValueError(f"current-policy cluster decision counts disagree for {candidate}")
        key = f"cifar10c_{candidate}"
        row = tracks.setdefault(key, {})
        _drop_stale_ci_aliases(row)
        historical_seal = row.get("historical_audit_seal") or row.pop("seal", None)
        row.pop("seal", None)
        for stale in (
            "cluster_robust_beats_both",
            "conditional_cluster_resampling",
            "regret_interpolated_superseded",
            "source_correction",
            "reproduction",
            "adverse_corruption_families",
            "decisions_changed_by_leave_one_out_of_pool",
        ):
            row.pop(stale, None)
        row.update(
            {
                "regret": _regret(score),
                "false_adapt": score["fa_u"],
                "false_adapt_count": score["false_adapt_count"],
                "cp95_upper_fa_c": _cp95_upper(score["false_adapt_count"], score["adapt_count"]),
                "decision_counts": _decision_counts(score),
                "n_cells": score["n"],
                "seeds": cifar_panel["seeds"],
                "conditions_per_seed": score["n"] // len(cifar_panel["seeds"]),
                "quantile_rule": (
                    "per-candidate/per-run-seed exact-rank leave-one-condition-out empirical residual calibration"
                ),
                "point_beats_both": score["point_beats_both"],
                "ci_robust_beats_both": score["seed_inference"]["ci_robust_beats_both"],
                "run_seed_inference": score["seed_inference"],
                "comparison_inference": _comparison_inference(score),
                "inference_scope": (
                    "run-seed summaries are conditional on one archived checkpoint/protocol; "
                    "they are not independent model-seed inference"
                ),
                "source": source,
                "source_manifest": source_manifest,
                "status": CURRENT_POLICY_STATUS,
                "current_policy_authority": True,
                "current_policy_family_sensitivity": cluster_sensitivity,
                "historical_audit_seal": historical_seal,
            }
        )
        if candidate == "tent":
            row["historical_cluster_resampling"] = _normalized_historical_cluster()
    tracks["cifar10c_tent"]["verdict"] = (
        "Current exact-rank point estimate beats both fixed policies. In a retrospective "
        "current-policy sensitivity over six corruption families, both ordinary family-bootstrap "
        "intervals are positive; however, both preregistered six-comparison Holm p-values are "
        "0.09375, so no cluster-robust, confirmatory, or independent-checkpoint win is claimed."
    )
    tracks["cifar10c_eata"]["verdict"] = (
        "Current exact-rank point estimate beats both fixed policies, but the ordinary adapt-side "
        "family-bootstrap interval crosses zero and the preregistered six-comparison Holm family "
        "fails; no cluster-robust or independent-checkpoint claim is made."
    )
    tracks["cifar10c_sar"]["verdict"] = (
        "Completed negative arm: zero observed false adaptation but higher regret than always-adapt."
    )

    office_panel = panels["officehome"]
    office = office_panel["primary"]["exact_rank_transfer_score"]
    office_row = tracks["officehome_M_v2"]
    _drop_stale_ci_aliases(office_row)
    office_row.pop("independent_seed_replication", None)
    historical_seal = office_row.get("historical_audit_seal") or office_row.pop("seal", None)
    office_row.pop("seal", None)
    office_row.update(
        {
            "regret": _regret(office),
            "false_adapt": office["fa_u"],
            "false_adapt_count": office["false_adapt_count"],
            "n_test": office["n"],
            "seeds": office_panel["primary"]["test_seeds"],
            "decision_counts": _decision_counts(office),
            "cp95_upper_fa_c": _cp95_upper(
                office["false_adapt_count"], office["adapt_count"]
            ),
            "point_beats_both": office["point_beats_both"],
            "ci_robust_beats_both": office["seed_inference"]["ci_robust_beats_both"],
            "comparison_inference": _comparison_inference(office),
            "source": source,
            "source_manifest": source_manifest,
            "quantile_rule": "held-out target test with leave-one-calibration-record-out residuals and exact-rank radius",
            "a7_status": office_panel["primary"]["calibration"]["a7_status"],
            "verdict": (
                "Descriptive no-harm tie with freeze and zero ADAPT decisions under the locked "
                "release runtime; the predeclared uniform A7 stability premise is absent."
            ),
            "source_caveat": (
                "The compact source records and original SHA-256 hashes are released. Decision "
                f"counts are {office['adapt_count']} ADAPT, {office['freeze_count']} FREEZE, and "
                f"{office['abstain_count']} ABSTAIN under the recorded runtime."
            ),
            "test_stream_seed_replication": {
                "regret": _regret(office_panel["test_stream_seed_replication"]["exact_rank_transfer_score"]),
                "n_test": office_panel["test_stream_seed_replication"]["n_test"],
                "seeds": office_panel["test_stream_seed_replication"]["test_seeds"],
                "decision_counts": _decision_counts(
                    office_panel["test_stream_seed_replication"]["exact_rank_transfer_score"]
                ),
                "ci_robust_beats_both": office_panel["test_stream_seed_replication"]["exact_rank_transfer_score"][
                    "seed_inference"
                ]["ci_robust_beats_both"],
                "a7_status": office_panel["test_stream_seed_replication"]["calibration"]["a7_status"],
                "comparison_inference": _comparison_inference(
                    office_panel["test_stream_seed_replication"]["exact_rank_transfer_score"]
                ),
            },
            "status": CURRENT_POLICY_STATUS,
            "current_policy_authority": True,
            "historical_audit_seal": historical_seal,
        }
    )

    iwild_panel = panels["iwildcam"]["primary"]
    iwild_row = tracks["iwildcam_H_v2"]
    historical_audit_seal = iwild_row.get("historical_audit_seal") or iwild_row.get("seal")
    _drop_stale_ci_aliases(iwild_row)
    iwild_row.update(
        {
            "regret": None,
            "false_adapt": None,
            "false_adapt_count": None,
            "n_test": None,
            "decision_counts": {"ADAPT": None, "FREEZE": None, "ABSTAIN": None},
            "cp95_upper_fa_c": None,
            "point_beats_both": None,
            "ci_robust_beats_both": None,
            "source": source,
            "source_manifest": source_manifest,
            "seal": None,
            "historical_audit_seal": historical_audit_seal,
            "quantile_rule": None,
            "a7_status": iwild_panel["calibration"]["a7_status"],
            "verdict": "Numerical and action claims withheld pending an official-metric, population-sealed rerun.",
            "guarantee_status": "not_evaluable_for_release",
            "source_caveat": (
                "The archived records remain available for audit, but their scores and actions "
                "are not release-eligible."
            ),
            **_withheld_iwildcam_fields(),
        }
    )

    camelyon_panel = panels["camelyon17"]["ood"]
    camelyon = camelyon_panel["replay"]["exact_rank_transfer_score"]
    camelyon_row = tracks["camelyon17_ood"]
    _drop_stale_ci_aliases(camelyon_row)
    camelyon_row.pop("source_correction", None)
    camelyon_seal = camelyon_row.get("historical_audit_seal") or camelyon_row.pop("seal", None)
    camelyon_row.pop("seal", None)
    camelyon_row.update(
        {
            "regret": _regret(camelyon),
            "false_adapt": camelyon["fa_u"],
            "false_adapt_count": camelyon["false_adapt_count"],
            "n_test": camelyon["n"],
            "decision_counts": _decision_counts(camelyon),
            "cp95_upper_fa_c": _cp95_upper(
                camelyon["false_adapt_count"], camelyon["adapt_count"]
            ),
            "guarantee_status": "exercised",
            "dev_seeds": camelyon_panel["replay"]["calibration_seeds"],
            "test_seeds": camelyon_panel["replay"]["test_seeds"],
            "point_beats_both": camelyon["point_beats_both"],
            "ci_robust_beats_both": camelyon["seed_inference"]["ci_robust_beats_both"],
            "comparison_inference": _comparison_inference(camelyon),
            "source": source,
            "source_manifest": source_manifest,
            "status": CURRENT_POLICY_STATUS,
            "current_policy_authority": True,
            "numeric_release_eligible": True,
            "headline_promotion_eligible": False,
            "historical_audit_seal": camelyon_seal,
            "verdict": (
                "Opened OOD diagnostic: KGA adapts on all 18 conditions and ties always-adapt; "
                "it is not prospective and not a beats-both result."
            ),
            "source_caveat": camelyon_panel["claim_scope"],
            "reproducibility_status": "REPRODUCIBLE_FROM_CANONICAL_COMPACT_SOURCE",
        }
    )

    rxrx_panel = panels["rxrx1"]
    rxrx = rxrx_panel["primary_model_seed0"]["exact_rank_transfer_score"]
    rxrx_row = tracks["rxrx1_J"]
    _drop_stale_ci_aliases(rxrx_row)
    rxrx_seal = rxrx_row.get("historical_audit_seal") or rxrx_row.pop("seal", None)
    rxrx_row.pop("seal", None)
    rxrx_row.update(
        {
            "regret": _regret(rxrx),
            "false_adapt": rxrx["fa_u"],
            "false_adapt_count": rxrx["false_adapt_count"],
            "n_test": rxrx["n"],
            "decision_counts": _decision_counts(rxrx),
            "cp95_upper_fa_c": None,
            "guarantee_status": "not_exercised_zero_adapt",
            "dev_seeds": rxrx_panel["primary_model_seed0"]["calibration_seeds"],
            "test_seeds": rxrx_panel["primary_model_seed0"]["test_seeds"],
            "point_beats_both": rxrx["point_beats_both"],
            "ci_robust_beats_both": rxrx["seed_inference"]["ci_robust_beats_both"],
            "comparison_inference": _comparison_inference(rxrx),
            "model_seed_robustness": rxrx_panel["model_seed_robustness"]["aggregate"],
            "source": source,
            "source_manifest": source_manifest,
            "status": CURRENT_POLICY_STATUS,
            "current_policy_authority": True,
            "numeric_release_eligible": True,
            "headline_promotion_eligible": False,
            "historical_audit_seal": rxrx_seal,
            "verdict": (
                "Harmful-dominated no-harm diagnostic: KGA freezes on every primary condition "
                "and ties always-freeze; three independent model seeds reproduce that tie."
            ),
            "source_note": rxrx_panel["claim_scope"],
        }
    )

    cifar101_panel = panels["cifar101"]
    cifar101 = cifar101_panel["replay"]["exact_rank_transfer_score"]
    cifar101_row = tracks["cifar10_1_K"]
    _drop_stale_ci_aliases(cifar101_row)
    cifar101_seal = cifar101_row.get("historical_audit_seal") or cifar101_row.pop("seal", None)
    cifar101_row.pop("seal", None)
    cifar101_row.update(
        {
            "regret": _regret(cifar101),
            "false_adapt_unconditional": cifar101["fa_u"],
            "false_adapt_conditional": cifar101["fa_c"],
            "false_adapt_count": cifar101["false_adapt_count"],
            "n_test": cifar101["n"],
            "decision_counts": _decision_counts(cifar101),
            "point_beats_both": cifar101["point_beats_both"],
            "ci_robust_beats_both": cifar101["seed_inference"]["ci_robust_beats_both"],
            "comparison_inference": _comparison_inference(cifar101),
            "source": source,
            "source_manifest": source_manifest,
            "status": CURRENT_POLICY_STATUS,
            "current_policy_authority": True,
            "numeric_release_eligible": True,
            "headline_promotion_eligible": False,
            "historical_audit_seal": cifar101_seal,
            "cp95_upper_fa_c": None,
            "verdict": (
                "Locked negative diagnostic: exact-rank KGA makes no ADAPT decisions and ties "
                "always-freeze, so it does not beat both fixed policies."
            ),
            "source_note": cifar101_panel["claim_scope"],
        }
    )

    table["headtohead"] = _normalized_historical_headtohead(cifar_panel["candidates"]["tent"])

    accounting = table.setdefault("decision_accounting_summary", {})
    accounting["note"] = (
        "Current rows below are refreshed from the source-hashed canonical panel. "
        "The controlled multimodal D33 row is a separately sealed auxiliary result; "
        "iWildCam remains withheld. This block is not an independent inference authority."
    )
    accounting_rows = accounting.setdefault("rows", [])
    # PACS has only aggregate seed summaries in the canonical authority; its archived
    # per-cell state cannot replay actions, so legacy action counts must not survive here.
    accounting_rows[:] = [row for row in accounting_rows if row.get("track") != "PACS (pooled)"]
    for candidate, track in (
        ("tent", "CIFAR-10-C Tent"),
        ("eata", "CIFAR-10-C EATA"),
        ("sar", "CIFAR-10-C SAR"),
    ):
        score = cifar_panel["candidates"][candidate]
        matches = [row for row in accounting_rows if row.get("track") == track]
        accounting_row = matches[0] if matches else {"track": track}
        if not matches:
            accounting_rows.append(accounting_row)
        accounting_row.update(
            {
                "n": score["n"],
                "ADAPT": score["adapt_count"],
                "FREEZE": score["freeze_count"],
                "ABSTAIN": score["abstain_count"],
                "false_adapts": score["false_adapt_count"],
                "FA_u": score["fa_u"],
                "cp95_upper_fa_c": _cp95_upper(
                    score["false_adapt_count"], score["adapt_count"]
                ),
                "guarantee": "exercised",
            }
        )

    for track, score, guarantee in (
        ("ImageNet-C SAR", imagenetc["sar"], "observed false-adapt count is 1/135"),
        ("Office-Home M v2", office, "not exercised: zero ADAPT decisions"),
        ("Camelyon17 OOD", camelyon, "vacuous on an all-helpful opened diagnostic"),
        ("RxRx1 J", rxrx, "not exercised: zero ADAPT decisions"),
        ("CIFAR-10.1 K", cifar101, "not exercised: zero ADAPT decisions"),
    ):
        accounting_row = _row_by_track(accounting_rows, track)
        accounting_row.update(
            {
                "n": score["n"],
                **_decision_counts(score),
                "false_adapts": score["false_adapt_count"],
                "FA_u": score["fa_u"],
                "cp95_upper_fa_c": _cp95_upper(score["false_adapt_count"], score["adapt_count"]),
                "guarantee": guarantee,
            }
        )
    _row_by_track(accounting_rows, "iWildCam H v2").update(
        {
            "n": None,
            "ADAPT": None,
            "FREEZE": None,
            "ABSTAIN": None,
            "false_adapts": None,
            "FA_u": None,
            "cp95_upper_fa_c": None,
            "guarantee": "withheld: invalid archived metric contract",
            **_withheld_iwildcam_fields(),
        }
    )

    pacs_panel = panels["pacs"]
    pacs = pacs_panel["pooled_domain_seed_mean"]
    pacs_row = tracks["pacs"]
    for stale in (
        "cp95_upper_fa_c",
        "decision_counts",
        "false_adapt_count",
        "false_adapt_count_status",
        "false_adapt_unconditional",
    ):
        pacs_row.pop(stale, None)
    pacs_row.update(
        {
            "completed_seeds": len(pacs_panel["seeds"]),
            "planned_seeds": len(pacs_panel["seeds"]),
            "mean_regret_kga_adapt_freeze": _regret(pacs),
            "reported_false_adapt_mean": pacs["fa_u"],
            "source": source,
            "source_manifest": source_manifest,
            "decision_replay_available": pacs_panel["decision_replay_available"],
            "decision_replay_blocker": pacs_panel["decision_replay_blocker"],
            "decision_counts_available": False,
            "false_adapt_count_available": False,
            "status": "canonical_aggregate_only_not_decision_replayable",
            "verdict": (
                "Completed three-seed null diagnostic. Seed summaries agree exactly, but archived "
                "per-cell files omit b_hat and calibration residuals, so gate decisions cannot be replayed."
            ),
        }
    )

    imagenetr_panel = panels["imagenet_r"]["panel"]
    imagenetr = imagenetr_panel["architecture_panel_aggregate"]
    candidates = imagenetr_panel["candidates"]
    worse = sum(row["regret"]["kga"] > row["regret"]["always_adapt"] for row in candidates.values())
    imagenetr_row = tracks["imagenet_r_D"]
    imagenetr_row.pop("mean_regret_interpolated_superseded", None)
    imagenetr_row.update(
        {
            "completed_seeds": imagenetr_panel["seeds"],
            "planned_seed_count": len(imagenetr_panel["seeds"]),
            "mean_regret_kga_adapt_freeze": _regret(imagenetr),
            "observed_false_adapt": f"{imagenetr['false_adapt_count']}/{imagenetr['n']}",
            "false_adapt_count": imagenetr["false_adapt_count"],
            "decision_counts": _decision_counts(imagenetr),
            "beats_both_backbones": "0/10",
            "worse_than_always_adapt_backbones": f"{worse}/10",
            "source": source,
            "source_manifest": source_manifest,
            "quantile_rule": "per-backbone/per-seed exact-rank leave-one-condition-out residual calibration",
            "verdict": (
                f"Negative four-seed, ten-backbone diagnostic: KGA is worse than always-adapt on "
                f"{worse}/10 backbones; no architecture has CI-robust beats-both."
            ),
            "per_backbone": {
                name: {
                    "regret": _regret(row),
                    "decision_counts": _decision_counts(row),
                    "false_adapt_count": row["false_adapt_count"],
                    "comparison_inference": _comparison_inference(row),
                }
                for name, row in candidates.items()
            },
            "comparison_inference": _comparison_inference(imagenetr),
            "status": CURRENT_POLICY_STATUS,
            "current_policy_authority": True,
        }
    )

    table["ci_convention"] = CI_CONVENTION
    normalized_cluster = {
        candidate: _normalized_current_cluster(current_cluster, candidate)
        for candidate in ("tent", "eata", "sar")
    }
    table["current_policy_family_sensitivity"] = {
        "status": CURRENT_CLUSTER_STATUS,
        "artifact_path": CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256(CURRENT_CLUSTER_PATH),
        "artifact_bytes": CURRENT_CLUSTER_PATH.stat().st_size,
        "artifact_schema": current_cluster["schema"],
        "convention": CI_CONVENTION,
        "runtime": current_cluster["runtime"],
        "live_code_bindings": current_cluster["live_code_bindings"],
        "inference": current_cluster["inference"],
        "preregistered_six_comparison_holm": current_cluster[
            "preregistered_six_comparison_holm"
        ],
        "claim_boundary": current_cluster["claim_boundary"],
        "candidates": normalized_cluster,
        "release_interpretation": (
            "Tent has positive ordinary family-bootstrap intervals against both fixed policies, "
            "but its two preregistered six-comparison Holm p-values are 0.09375. EATA and SAR "
            "also fail the preregistered gate. No cluster-robust or confirmatory win is promoted."
        ),
    }
    table["regenerated_utc"] = "2026-08-27"
    table["regeneration_provenance"] = (
        "All current-policy rows are regenerated from the source-hashed canonical panel. "
        "The separately code-bound current-policy family sensitivity is retrospective and fails "
        "the preregistered six-comparison Holm gate. Earlier-policy cluster and head-to-head "
        "artifacts are retained only in explicitly historical, non-release-eligible blocks."
    )
    table["nine_track_lock_seal"]["status"] = HISTORICAL_POLICY_STATUS
    table["nine_track_lock_seal"]["current_policy_authority"] = False
    table.setdefault("withheld_or_pending", {}).pop("cifar10c_sar", None)
    table["withheld_or_pending"].pop("current_policy_cluster_inference", None)
    table["withheld_or_pending"]["current_policy_headtohead"] = (
        "POEM/AETTA-style comparisons require Phase 2 recomputation against the current exact-rank KGA policy."
    )
    table["reconciliation_source"] = {
        "canonical_panel": source,
        "canonical_panel_sha256": _sha256(PANEL_PATH),
        "source_manifest": source_manifest,
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "generator": panel["generator"],
        "generator_sha256": panel["generator_sha256"],
        "runtime": panel.get("runtime"),
        "ci_convention": CI_CONVENTION,
        "current_policy_family_sensitivity": {
            "artifact": CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha256(CURRENT_CLUSTER_PATH),
            "artifact_bytes": CURRENT_CLUSTER_PATH.stat().st_size,
            "runtime": current_cluster["runtime"],
            "live_code_bindings": current_cluster["live_code_bindings"],
        },
    }


def _sync_uniform_verdicts(
    panel: dict[str, Any], uniform: dict[str, Any], current_cluster: dict[str, Any]
) -> None:
    source = PANEL_PATH.relative_to(ROOT).as_posix()
    source_manifest = SOURCE_MANIFEST.relative_to(ROOT).as_posix()
    panels = panel["panels"]
    cifar = panels["cifar10c"]["panel"]
    rows = uniform["wave"]

    def sync_score(
        track: str,
        score: dict[str, Any],
        *,
        unit: str,
        verdict: str,
        evidence_tier: str,
        note: str,
        decision_counts: bool = True,
    ) -> None:
        row = _row_by_track(rows, track)
        _drop_stale_ci_aliases(row)
        for stale in (
            "adverse_corruption_families",
            "cluster_robust_beats_both",
            "decisions_changed_by_leave_one_out_of_pool_radius",
            "leave_one_corruption_out_ablation",
            "per_seed_beating_both",
            "per_seed_tying_freeze_bit_identically",
            "quantile_rule",
            "regret_exact_rank",
            "regret_kga_interpolated_as_published",
            "sensitivity_in_pool_radius_SUPERSEDED",
        ):
            row.pop(stale, None)
        better = min(
            ("adapt", score["regret"]["always_adapt"]),
            ("freeze", score["regret"]["always_freeze"]),
            key=lambda item: item[1],
        )[0]
        row.update(
            {
                "unit": unit,
                "regret_kga": score["regret"]["kga"],
                "regret_adapt": score["regret"]["always_adapt"],
                "regret_freeze": score["regret"]["always_freeze"],
                "FA_u": score.get("fa_u"),
                "better_policy": better,
                "point_beats_both": score["point_beats_both"],
                "ci_robust_beats_both": score.get("seed_inference", {}).get(
                    "ci_robust_beats_both", False
                ),
                "survives_wave_holm": None,
                "verdict": verdict,
                "evidence_tier": evidence_tier,
                "source": source,
                "source_manifest": source_manifest,
                "numeric_release_eligible": True,
                "current_policy_authority": True,
                "comparison_inference": _comparison_inference(score),
                "note": note,
            }
        )
        if decision_counts:
            row["decision_counts"] = _decision_counts(score)
            row["cp95_upper_fa_c"] = _cp95_upper(
                score["false_adapt_count"], score["adapt_count"]
            )

    for candidate, track in (
        ("tent", "CIFAR-10-C Tent"),
        ("eata", "CIFAR-10-C EATA"),
        ("sar", "CIFAR-10-C SAR"),
    ):
        score = cifar["candidates"][candidate]
        cluster_sensitivity = _normalized_current_cluster(current_cluster, candidate)
        sync_score(
            track,
            score,
            unit=f"{len(cifar['seeds'])} run seeds x {score['n'] // len(cifar['seeds'])} cells",
            verdict=(
                "current-policy point beats both; retrospective six-family sensitivity only; "
                "preregistered six-comparison Holm fails"
                if candidate == "tent"
                else (
                    "current-policy point beats both; adapt-side family interval crosses zero; "
                    "preregistered six-comparison Holm fails"
                    if candidate == "eata"
                    else "negative arm: current-policy KGA has higher regret than always-adapt"
                )
            ),
            evidence_tier="source-hashed current exact-rank replay",
            note=(
                "Run seeds are nested under one archived checkpoint. Current family sensitivity "
                "is retrospective, has six corruption-family units, and is neither prospective "
                "nor independent-checkpoint inference."
            ),
        )
        uniform_row = _row_by_track(rows, track)
        uniform_row["current_policy_family_sensitivity"] = cluster_sensitivity
        uniform_row["survives_preregistered_six_comparison_holm"] = False

    imagenetc = panels["imagenetc"]["panel"]
    for candidate, track in (
        ("tent", "ImageNet-C Tent"),
        ("eata", "ImageNet-C EATA"),
        ("sar", "ImageNet-C SAR"),
    ):
        score = imagenetc["candidates"][candidate]
        sync_score(
            track,
            score,
            unit=f"{len(imagenetc['seeds'])} run seeds x {score['n'] // len(imagenetc['seeds'])} conditions",
            verdict=(
                "point estimate below both fixed policies; not CI-robust"
                if score["point_beats_both"]
                else "does not beat both fixed policies"
            ),
            evidence_tier="source-hashed current exact-rank replay",
            note="Current canonical per-candidate exact-rank row.",
        )

    office = panels["officehome"]["primary"]["exact_rank_transfer_score"]
    sync_score(
        "Office-Home M v2",
        office,
        unit=f"held-out target test n={office['n']}",
        verdict="descriptive no-harm tie with always-freeze; zero ADAPT decisions",
        evidence_tier="source-hashed current exact-rank replay",
        note="A7 full-fit-versus-LOO stability and independent model-seed replication are absent.",
    )

    camelyon = panels["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"]
    sync_score(
        "Camelyon17 OOD",
        camelyon,
        unit=f"opened OOD test diagnostic n={camelyon['n']}",
        verdict="ties always-adapt on an all-helpful opened diagnostic; not beats-both",
        evidence_tier="source-hashed opened diagnostic",
        note=panels["camelyon17"]["ood"]["claim_scope"],
    )

    rxrx = panels["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"]
    sync_score(
        "RxRx1 J",
        rxrx,
        unit=f"held-out harmful-dominated test n={rxrx['n']}",
        verdict="ties always-freeze; three independent model seeds reproduce the no-harm tie",
        evidence_tier="source-hashed current exact-rank replay",
        note=panels["rxrx1"]["claim_scope"],
    )

    cifar101 = panels["cifar101"]["replay"]["exact_rank_transfer_score"]
    sync_score(
        "CIFAR-10.1 K",
        cifar101,
        unit=f"cross-seed test n={cifar101['n']}",
        verdict="negative diagnostic: no ADAPT decisions and exact tie with always-freeze",
        evidence_tier="source-hashed current exact-rank replay",
        note=panels["cifar101"]["claim_scope"],
    )

    pacs = panels["pacs"]["pooled_domain_seed_mean"]
    pacs_row = _row_by_track(rows, "PACS (4 LODO)")
    _drop_stale_ci_aliases(pacs_row)
    pacs_row.update(
        {
            "unit": "3 seeds x 4 held-out domains",
            "regret_kga": pacs["regret"]["kga"],
            "regret_adapt": pacs["regret"]["always_adapt"],
            "regret_freeze": pacs["regret"]["always_freeze"],
            "FA_u": pacs["fa_u"],
            "better_policy": "adapt",
            "point_beats_both": False,
            "ci_robust_beats_both": False,
            "survives_wave_holm": None,
            "verdict": "three-seed null diagnostic; KGA is worse than always-adapt",
            "evidence_tier": "source-hashed aggregate; gate decisions not replayable",
            "source": source,
            "source_manifest": source_manifest,
            "numeric_release_eligible": True,
            "current_policy_authority": True,
            "note": panels["pacs"]["decision_replay_blocker"],
        }
    )

    imagenetr = panels["imagenet_r"]["panel"]["architecture_panel_aggregate"]
    sync_score(
        "ImageNet-R D",
        imagenetr,
        unit="4 run seeds x 10 backbones x 12 conditions",
        verdict="negative architecture-panel diagnostic; KGA is worse than always-adapt on 8/10 backbones",
        evidence_tier="source-hashed current exact-rank replay",
        note="No backbone has a CI-robust beats-both result.",
    )

    historical_h2h = _row_by_track(rows, "Mixed head-to-head (CIFAR-10-C Tent primary)")
    _drop_stale_ci_aliases(historical_h2h)
    historical_h2h.update(
        {
            "unit": "archived earlier-policy pooled stream",
            "regret_kga": None,
            "regret_adapt": None,
            "regret_freeze": None,
            "FA_u": None,
            "better_policy": None,
            "point_beats_both": None,
            "ci_robust_beats_both": None,
            "survives_wave_holm": None,
            "verdict": "historical only; current-policy head-to-head recomputation required",
            "evidence_tier": HISTORICAL_POLICY_STATUS,
            "source": HISTORICAL_HEADTOHEAD_PATH.relative_to(ROOT).as_posix(),
            "source_manifest": None,
            "numeric_release_eligible": False,
            "policy_synchronized": False,
            "current_policy_authority": False,
            "release_eligible_win": False,
            "historical_comparisons": _normalized_historical_headtohead(cifar["candidates"]["tent"])[
                "archived_comparisons"
            ],
            "note": "Holm-adjusted p-values are historical; the paired confidence intervals are unadjusted.",
        }
    )

    iwild = _row_by_track(rows, "iWildCam H v2")
    _drop_stale_ci_aliases(iwild)
    iwild.update(
        {
            "unit": "archived test row withheld; official-metric, population-sealed rerun required",
            "regret_kga": None,
            "regret_adapt": None,
            "regret_freeze": None,
            "FA_u": None,
            "better_policy": None,
            "point_beats_both": None,
            "ci_robust_beats_both": None,
            "survives_wave_holm": None,
            "verdict": "withheld pending an official-metric, population-sealed rerun",
            "evidence_tier": "audit-only; not release-eligible",
            "source": source,
            "source_manifest": source_manifest,
            "note": "Archived sklearn macro-F1 values and actions are not promoted.",
            **_withheld_iwildcam_fields(),
        }
    )
    meta = uniform["_meta"]
    meta["scored_utc"] = "2026-08-27"
    meta.pop("nboot", None)
    meta["current_seed_bootstrap_replicates"] = 20000
    meta["ci_convention"] = CI_CONVENTION
    meta["ci_method"] = (
        "Current comparison intervals are descriptive paired percentile bootstraps of run-seed "
        "means and use baseline regret minus KGA regret. The current CIFAR-10-C family-sensitivity "
        "intervals are ordinary, unadjusted family bootstraps. Historical head-to-head intervals "
        "are also unadjusted; Holm applies only to their archived p-values."
    )
    meta["rule"] = (
        "Point beats-both means current KGA regret is lower than both fixed policies. No current "
        "row is promoted as CI-robust unless its current-policy inference explicitly supports it."
    )
    meta["integrity"] = (
        "Every current row is regenerated from the source-hashed canonical panel; withheld, "
        "withdrawn, and earlier-policy evidence is explicitly segregated."
    )
    meta["radius_rule"] = (
        "Current rows use the unclamped exact empirical rank and per-track leave-one-out "
        "calibration recorded in the canonical panel. Earlier clamped/interpolated policies "
        "appear only in explicitly historical, non-release-eligible evidence."
    )
    meta["stale_rows_requiring_pipeline_regeneration"] = [
        "Mixed head-to-head: recompute POEM/AETTA-style comparisons against the current exact-rank KGA policy."
    ]
    meta["multiplicity_family_status"] = (
        "The current CIFAR-10-C sensitivity applies Holm to the preregistered six "
        "candidate-by-baseline sign-flip p-values; every candidate fails its two-baseline gate. "
        "No current-policy POEM/AETTA head-to-head family has been evaluated."
    )
    meta.pop("wave_holm_family", None)
    meta["corrections_applied"] = (
        "Current CIFAR-10-C family-bootstrap confidence intervals are unadjusted. Holm is applied "
        "to the preregistered six sign-flip p-values; within-candidate two-contrast Holm values "
        "are post hoc only. Archived POEM/AETTA-style p-values use a separate historical Holm "
        "correction; their archived paired percentile confidence intervals are unadjusted."
    )
    meta["reconciliation_source"] = {
        "canonical_panel": source,
        "canonical_panel_sha256": _sha256(PANEL_PATH),
        "source_manifest": source_manifest,
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "scope": "Current rows are canonical; explicitly named earlier-policy fields are historical only.",
        "current_policy_family_sensitivity_artifact": (
            CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix()
        ),
        "current_policy_family_sensitivity_sha256": _sha256(CURRENT_CLUSTER_PATH),
    }
    meta["current_policy_family_sensitivity"] = {
        "artifact_path": CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256(CURRENT_CLUSTER_PATH),
        "artifact_bytes": CURRENT_CLUSTER_PATH.stat().st_size,
        "runtime": current_cluster["runtime"],
        "live_code_bindings": current_cluster["live_code_bindings"],
        "preregistered_six_comparison_holm": current_cluster[
            "preregistered_six_comparison_holm"
        ],
        "claim_boundary": current_cluster["claim_boundary"],
    }
    # The prior Holm family and migration summary mixed earlier policies with
    # current rows.  Do not leave their numeric p-values or promotion labels on
    # a generated release surface: Phase 2 must rebuild the family from a
    # policy-synchronized arm inventory.
    uniform["wave_holm"] = []
    uniform["migration"] = {
        "status": "current_policy_reconciled_phase1",
        "current_policy_point_beats_both": [
            "CIFAR-10-C Tent",
            "CIFAR-10-C EATA",
            "ImageNet-C SAR",
        ],
        "retrospective_current_policy_family_sensitivity": {
            "positive_unadjusted_intervals_vs_both": ["CIFAR-10-C Tent"],
            "preregistered_six_comparison_holm_rejects_both": [],
            "note": (
                "Tent's within-candidate two-contrast Holm p-values are 0.03125, but that "
                "comparison family is post hoc. The preregistered six-comparison Holm p-values "
                "are 0.09375 for both Tent baselines."
            ),
        },
        "current_policy_negative_or_tie_diagnostics": [
            "CIFAR-10-C SAR",
            "Office-Home M v2",
            "Camelyon17 OOD",
            "RxRx1 J",
            "PACS (4 LODO)",
            "ImageNet-R D",
            "CIFAR-10.1 K",
        ],
        "historical_only": [
            "CIFAR-10-C Tent cluster resampling",
            "Mixed head-to-head (CIFAR-10-C Tent primary)",
        ],
        "withheld": ["iWildCam H v2"],
        "withdrawn": ["Camelyon17 G"],
        "note": (
            "A current-policy six-family sensitivity is now available. Tent's ordinary intervals "
            "are positive, but no candidate passes the preregistered six-comparison Holm gate; "
            "therefore no cluster-robust or confirmatory win is promoted. A policy-synchronized "
            "POEM/AETTA Phase 2 recomputation is still required."
        ),
    }


def _sync_decision_metrics(
    panel: dict[str, Any], metrics: dict[str, Any], current_cluster: dict[str, Any]
) -> None:
    source = PANEL_PATH.relative_to(ROOT).as_posix()
    source_manifest = SOURCE_MANIFEST.relative_to(ROOT).as_posix()
    panels = panel["panels"]
    cifar = panels["cifar10c"]["panel"]
    rows = metrics["tracks"]

    def sync_score(track: str, score: dict[str, Any], *, seeds: list[int], note: str) -> None:
        matches = [row for row in rows if row.get("track") == track]
        row = matches[0] if matches else {"track": track}
        if not matches:
            rows.append(row)
        row.setdefault("actions", {})
        for stale in ("superseded_aggregate", "reproduction_script", "supplemental_multiseed_stability"):
            row.pop(stale, None)
        for action, field in (("adapt", "adapt_count"), ("freeze", "freeze_count"), ("abstain", "abstain_count")):
            count = score[field]
            row["actions"][action] = {
                "count": count,
                "rate": count / score["n"],
                "ci95_wilson": _wilson(count, score["n"]),
            }
        row.update(
            {
                "n_decisions": score["n"],
                "seeds": seeds,
                "source": source,
                "source_manifest": source_manifest,
                "reproduction_command": (
                    "python scripts/reconcile_result_panels.py && python scripts/sync_reconciled_panels.py"
                ),
                "regret_kga_adapt_freeze": _regret(score),
                "numeric_release_eligible": True,
                "current_policy_authority": True,
                "comparison_inference": _comparison_inference(score),
                "status": CURRENT_POLICY_STATUS,
                "note": note,
            }
        )
        row["false_adapt_unconditional"] = {
            "count": score["false_adapt_count"],
            "n": score["n"],
            "rate": score["fa_u"],
            "ci95_wilson": _wilson(score["false_adapt_count"], score["n"]),
        }
        row["false_adapt_conditional"] = {
            "definition": "Pr[Delta<=0 | ADAPT]",
            "false_adapts": score["false_adapt_count"],
            "n_adapt_decisions": score["adapt_count"],
            "rate": score["fa_c"],
            "cp95_upper": _cp95_upper(
                score["false_adapt_count"], score["adapt_count"]
            ),
            "guarantee_status": "exercised" if score["adapt_count"] else "not_exercised_zero_adapt",
        }
        row["interval_coverage_observed"] = {
            "status": "not_promoted_from_canonical_score_summary"
        }
        row["theoretical_coverage"] = {
            "status": "conditional_on_declared_premise",
            "note": "No theorem premise is inferred from an empirical score summary.",
        }

    def scrub_historical(track: str, reason: str) -> None:
        row = _row_by_track(rows, track)
        old_source = row.get("historical_source") or row.get("source")
        row.update(
            {
                "n_decisions": None,
                "seeds": None,
                "source": None,
                "source_manifest": None,
                "historical_source": old_source,
                "reproduction_command": None,
                "actions": {
                    action: {
                        "count": None,
                        "rate": None,
                        "ci95_wilson": None,
                        "status": HISTORICAL_POLICY_STATUS,
                    }
                    for action in ("adapt", "freeze", "abstain")
                },
                "false_adapt_unconditional": {
                    "count": None,
                    "n": None,
                    "rate": None,
                    "ci95_wilson": None,
                    "status": HISTORICAL_POLICY_STATUS,
                },
                "false_adapt_conditional": {
                    "false_adapts": None,
                    "n_adapt_decisions": None,
                    "rate": None,
                    "cp95_upper": None,
                    "guarantee_status": HISTORICAL_POLICY_STATUS,
                },
                "regret_kga_adapt_freeze": None,
                "comparison_inference": None,
                "numeric_release_eligible": False,
                "current_policy_authority": False,
                "status": HISTORICAL_POLICY_STATUS,
                "note": reason,
            }
        )

    for candidate, track in (
        ("tent", "CIFAR-10-C TENT"),
        ("eata", "CIFAR-10-C EATA"),
        ("sar", "CIFAR-10-C SAR"),
    ):
        sync_score(
            track,
            cifar["candidates"][candidate],
            seeds=cifar["seeds"],
            note=(
                "Current per-candidate exact-rank replay. Run-seed inference is descriptive; "
                "the separate six-family sensitivity is retrospective and fails the "
                "preregistered six-comparison Holm gate."
            ),
        )
        _row_by_track(rows, track)["current_policy_family_sensitivity"] = (
            _normalized_current_cluster(current_cluster, candidate)
        )

    imagenetc = panels["imagenetc"]["panel"]
    for candidate, track in (
        ("tent", "ImageNet-C TENT"),
        ("eata", "ImageNet-C EATA"),
        ("sar", "ImageNet-C SAR"),
    ):
        sync_score(
            track,
            imagenetc["candidates"][candidate],
            seeds=imagenetc["seeds"],
            note="Current per-candidate exact-rank replay.",
        )

    cifar101 = panels["cifar101"]["replay"]["exact_rank_transfer_score"]
    sync_score(
        "CIFAR-10.1 TENT",
        cifar101,
        seeds=panels["cifar101"]["replay"]["test_seeds"],
        note="Current locked negative cross-seed diagnostic.",
    )
    scrub_historical("CIFAR-10.1 EATA", "No current-policy EATA row exists in the canonical panel.")
    scrub_historical("CIFAR-10.1 SAR", "No current-policy SAR row exists in the canonical panel.")

    office = panels["officehome"]["primary"]["exact_rank_transfer_score"]
    sync_score(
        "OfficeHome",
        office,
        seeds=panels["officehome"]["primary"]["test_seeds"],
        note="Current primary held-out target replay; A7 stability is not established.",
    )

    camelyon = panels["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"]
    sync_score(
        "Camelyon17",
        camelyon,
        seeds=panels["camelyon17"]["ood"]["replay"]["test_seeds"],
        note="Opened all-helpful OOD diagnostic; not prospective and not beats-both.",
    )

    rxrx = panels["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"]
    sync_score(
        "RxRx1 sar_online",
        rxrx,
        seeds=panels["rxrx1"]["primary_model_seed0"]["test_seeds"],
        note="Primary model-seed-0 score; three independent model seeds all tie always-freeze.",
    )
    scrub_historical("RxRx1 eata_online", "No current-policy EATA row exists in the canonical panel.")
    scrub_historical("RxRx1 tent_online", "No current-policy Tent row exists in the canonical panel.")

    imagenetr = panels["imagenet_r"]["panel"]
    for backbone, score in imagenetr["candidates"].items():
        sync_score(
            f"ImageNet-R {backbone}",
            score,
            seeds=imagenetr["seeds"],
            note="Current per-backbone exact-rank replay.",
        )

    for domain in ("art_painting", "cartoon", "photo", "sketch"):
        scrub_historical(
            f"PACS {domain}",
            "The canonical PACS evidence supports only the pooled three-seed summary; archived per-domain gate decisions are not replayable.",
        )

    iwild = _row_by_track(rows, "iWildCam")
    iwild.pop("supplemental_multiseed_stability", None)
    iwild.update(
        {
            "n_decisions": None,
            "seeds": None,
            "source": source,
            "source_manifest": source_manifest,
            "reproduction_command": None,
            "actions": {
                action: {"count": None, "rate": None, "ci95_wilson": None, "status": "withheld"}
                for action in ("adapt", "freeze", "abstain")
            },
            "false_adapt_unconditional": {
                "count": None,
                "n": None,
                "rate": None,
                "ci95_wilson": None,
                "status": "withheld",
            },
            "interval_coverage_observed": {
                "count": None,
                "n": None,
                "rate": None,
                "status": "withheld",
            },
            "theoretical_coverage": {
                "status": "not_evaluable_for_release",
                "note": "The archived metric contract is invalid for release-level numerical use.",
            },
            "false_adapt_conditional": {
                "false_adapts": None,
                "n_adapt_decisions": None,
                "rate": None,
                "cp95_upper": None,
                "guarantee_status": "withheld",
            },
            **_withheld_iwildcam_fields(),
        }
    )
    metrics["aggregate_provenance_rule"] = (
        "Every numeric row marked current_policy_authority=true comes from the source-hashed canonical panel. "
        "Rows absent from that panel are scrubbed and retained only as historical placeholders."
    )
    metrics["interval_method"] = (
        "Two-sided 95% Wilson intervals are used for action and unconditional false-adapt "
        "counts. Conditional false-adapt reports a one-sided 95% exact Clopper-Pearson upper "
        "bound from false-adapt events over ADAPT decisions; it is undefined when ADAPT=0."
    )
    metrics["fa_u_identity_note"] = (
        "No in-sample rank identity is promoted. Current false-adapt counts and rates are copied "
        "from the source-hashed canonical per-track replay; unreplayable PACS counts are omitted."
    )
    metrics["guarantee_untested_rule"] = (
        "Conditional false-adapt evidence is exercised only when at least one ADAPT decision "
        "occurs; zero-ADAPT rows are marked not_exercised_zero_adapt."
    )
    metrics["comparison_ci_convention"] = CI_CONVENTION
    metrics["current_policy_family_sensitivity"] = {
        "artifact_path": CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256(CURRENT_CLUSTER_PATH),
        "artifact_bytes": CURRENT_CLUSTER_PATH.stat().st_size,
        "status": CURRENT_CLUSTER_STATUS,
        "preregistered_six_comparison_holm": current_cluster[
            "preregistered_six_comparison_holm"
        ],
        "interpretation": (
            "Tent has positive ordinary intervals against both baselines. No candidate rejects "
            "both comparisons after the preregistered six-way Holm correction."
        ),
    }
    metrics["reconciliation_source"] = {
        "canonical_panel": source,
        "canonical_panel_sha256": _sha256(PANEL_PATH),
        "source_manifest": source_manifest,
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
    }


def _sync_ledger(ledger: dict[str, Any], current_cluster: dict[str, Any]) -> None:
    source = PANEL_PATH.relative_to(ROOT).as_posix()
    source_manifest = SOURCE_MANIFEST.relative_to(ROOT).as_posix()

    theorem = _claim(ledger, "KB-CLAIM-001")
    theorem["claim_text"] = (
        "A strict adapt or freeze commitment is uniformly supportable over the declared drift class "
        "iff |M| > beta; on |M| <= beta, abstention is the maximal sound three-way action."
    )
    theorem["allowed_wording"] = "strict-commitment frontier over the declared drift class"
    theorem["forbidden_wording"] = ["assumption-free", "universal", "benefit sign identifiable iff"]

    certificate = _claim(ledger, "KB-CLAIM-003")
    certificate["claim_text"] = (
        "If P(|Delta_hat-Delta| <= epsilon) >= 1-alpha, the KGA interval rule controls the "
        "unconditional false-adapt event FA_u at level alpha."
    )
    certificate["assumptions"] = [
        "valid marginal interval coverage",
        "exchangeability or another justified calibration argument is one route to coverage",
    ]
    certificate["allowed_wording"] = "FA_u <= alpha under interval coverage"

    cifar = _claim(ledger, "KB-CLAIM-010")
    cifar.update(
        {
            "claim_text": (
                "CIFAR-10-C stress grid: Tent and EATA beat both fixed policies in pooled point "
                "estimates under the current exact-rank replay, while SAR loses to always-adapt. "
                "A retrospective current-policy sensitivity over six corruption families gives "
                "Tent positive ordinary family-bootstrap intervals against both baselines, but "
                "the preregistered six-comparison Holm p-values are 0.09375 for both Tent "
                "contrasts; EATA and SAR also fail the two-baseline gate."
            ),
            "supporting_artifacts": [
                source,
                source_manifest,
                CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
            ],
            "assumptions": [
                "controlled cross-fitted grid",
                "run seeds share the archived checkpoint/protocol",
                "six observed corruption families are the retrospective sensitivity units",
            ],
            "allowed_wording": (
                "controlled current-policy point-estimate routing gains; retrospective six-family "
                "Tent sensitivity with positive unadjusted intervals; preregistered six-comparison "
                "Holm fails"
            ),
            "forbidden_wording": [
                "five independent model seeds",
                "current-policy cluster-robust win",
                "preregistered cluster pass",
                "confirmatory cluster win",
                "simultaneous confidence intervals",
                "universal TTA improvement",
                "natural-shift win",
                "official POEM or AETTA superiority",
            ],
            "current_policy_family_sensitivity": {
                "artifact_path": CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
                "artifact_sha256": _sha256(CURRENT_CLUSTER_PATH),
                "artifact_bytes": CURRENT_CLUSTER_PATH.stat().st_size,
                "status": CURRENT_CLUSTER_STATUS,
                "runtime": current_cluster["runtime"],
                "live_code_bindings": current_cluster["live_code_bindings"],
                "preregistered_six_comparison_holm": current_cluster[
                    "preregistered_six_comparison_holm"
                ],
                "candidates": {
                    candidate: {
                        "pointwise_family_intervals_positive_vs_both": current_cluster[
                            "candidates"
                        ][candidate]["gate"][
                            "both_pointwise_95pct_cluster_bootstrap_intervals_positive"
                        ],
                        "within_candidate_posthoc_holm_rejects_both": current_cluster[
                            "candidates"
                        ][candidate]["gate"][
                            "both_one_sided_sign_flip_tests_survive_within_candidate_posthoc_holm_0.05"
                        ],
                        "preregistered_six_comparison_holm_rejects_both": False,
                    }
                    for candidate in ("tent", "eata", "sar")
                },
            },
        }
    )

    head_to_head = _claim(ledger, "KB-CLAIM-026")
    head_to_head.update(
        {
            "claim_text": (
                "The archived CIFAR-10-C Tent head-to-head compared protocol-matched POEM- and "
                "AETTA-style ports against an earlier KGA policy that was not recomputed. Its "
                "Holm-adjusted p-values and unadjusted paired intervals are historical only."
            ),
            "status": "diagnostic",
            "supporting_artifacts": [
                HISTORICAL_HEADTOHEAD_PATH.relative_to(ROOT).as_posix(),
                source,
                source_manifest,
            ],
            "assumptions": [
                "archived scorer recorded recompute_kga=false",
                "current exact-rank KGA policy differs from the archived KGA policy",
            ],
            "allowed_wording": (
                "historical earlier-policy comparison; current-policy POEM/AETTA-style recomputation pending"
            ),
            "forbidden_wording": [
                "current-policy WIN",
                "release-eligible WIN",
                "official POEM reproduction",
                "official AETTA reproduction",
                "beats POEM on natural shifts",
            ],
            "numeric_release_eligible": False,
            "policy_synchronized": False,
            "current_policy_authority": False,
            "release_eligible_win": False,
            "ci_convention": CI_CONVENTION,
            "holm_scope": "archived p-values only; paired confidence intervals are unadjusted",
        }
    )

    camelyon_withdrawn = _claim(ledger, "KB-CLAIM-022")
    camelyon_withdrawn.update(
        {
            "supporting_artifacts": [
                "research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml",
                "audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json",
                "audits/integrity_2026-06-20/camelyon_reconciliation/VERDICT_phase1.md",
            ],
            "allowed_wording": (
                "historical pooled Protocol G headline withdrawn; separate n=18 OOD row is "
                "an opened, all-helpful diagnostic"
            ),
            "artifact_pointer_correction_2026_08_27": (
                "The sealed reconciliation result and verdict are present and hash-verified. "
                "They support withdrawal of the pooled Protocol G headline; they do not turn "
                "the separate opened OOD row into a prospective or beats-both result."
            ),
        }
    )
    camelyon_withdrawn.pop("artifact_pointer_correction_2026_07_26", None)

    imagenetc = _claim(ledger, "KB-CLAIM-011")
    imagenetc.update(
        {
            "claim_text": (
                "ImageNet-C SAR exact-LOO panel has a pooled point estimate below both fixed policies, "
                "but the seed-level freeze comparison touches zero and FA_u is 1/135."
            ),
            "status": "no-harm",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": "exact_rank_leave_one_condition_out_residual_calibration",
            "test_split": "27 cross-fitted conditions per seed x 5 seeds",
            "assumptions": ["controlled grid", "seed is the primary inference unit"],
            "allowed_wording": "pooled point-estimate no-harm; no CI-robust beats-both claim",
            "forbidden_wording": ["zero false adaptation", "CI-supported beats-both", "natural-shift win"],
        }
    )

    office = _claim(ledger, "KB-CLAIM-020")
    office.update(
        {
            "claim_text": (
                "Office-Home M-v2 exact-rank source replay ties always-freeze with zero ADAPT "
                "decisions under the locked release runtime; a separate test-stream replication "
                "has a small point edge, but the A7 stability premise and independent model-seed "
                "replication are absent."
            ),
            "status": "descriptive",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": "calibration-record LOO residuals + exact-rank transfer radius",
            "assumptions": ["dev-locked candidate", "A7 full-fit-versus-LOO stability not established"],
            "allowed_wording": (
                "primary descriptive no-harm tie; separate replication has a small point edge; no robust natural win"
            ),
            "forbidden_wording": ["CI-robust beats both", "natural-shift win", "uniform no-harm"],
        }
    )

    iwild = _claim(ledger, "KB-CLAIM-021")
    iwild.update(
        {
            "claim_text": (
                "The archived iWildCam numerical and action row is withheld because its scorer "
                "does not match the official WILDS label-present macro-F1 contract."
            ),
            "status": "withheld",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": None,
            "assumptions": [
                "an official-metric rerun is required",
                "the evaluation population must be sealed before scoring",
            ],
            "allowed_wording": "numerical and action claims withheld pending a population-sealed official-metric rerun",
            "forbidden_wording": [
                "beats both",
                "powered safety result",
                "multi-seed natural win",
                "descriptive no-harm tie",
            ],
            **_withheld_iwildcam_fields(),
        }
    )

    mixture = _claim(ledger, "KB-CLAIM-024")
    mixture.update(
        {
            "status": "diagnostic",
            "allowed_wording": (
                "historical researcher-constructed routing aggregate; rerun required under reconciled "
                "per-track decisions"
            ),
            "forbidden_wording": ["promoted beats-both claim", "natural-shift win", "transfer result"],
        }
    )

    pacs = _claim(ledger, "KB-CLAIM-041")
    pacs.update(
        {
            "status": "diagnostic",
            "supporting_artifacts": [
                source,
                source_manifest,
                "experiments/kbound/results/smoke_pacs_replay_v2/PACS_REPLAY_AUDIT.json",
            ],
            "allowed_wording": (
                "three-seed null diagnostic; aggregate seed summaries cross-validated; separate "
                "one-domain smoke validates the v2 replay schema"
            ),
            "forbidden_wording": ["decision replay complete", "beats both"],
        }
    )

    imagenetr = _claim(ledger, "KB-CLAIM-042")
    imagenetr.update(
        {
            "status": "diagnostic",
            "supporting_artifacts": [source, source_manifest],
            "calibration_method": "per-backbone/per-seed exact-rank LOO residual calibration",
            "claim_text": (
                "ImageNet-R Protocol D exact-LOO replay is a negative architecture-panel diagnostic: "
                "KGA is worse than always-adapt on 8/10 backbones."
            ),
            "allowed_wording": "negative four-seed architecture-panel diagnostic",
            "forbidden_wording": ["beats both", "single deployable policy", "7/10 backbones"],
        }
    )

    universal = _claim(ledger, "KB-CLAIM-050")
    universal["supporting_artifacts"] = [
        "docs/research/kbound/kbound_submission_body.tex",
        "docs/research/kbound/claim_ledger.json",
    ]
    universal["source_locator"] = "kbound_submission_body.tex, Limitations and Discussion"

    ledger.pop("_artifact_audit_2026_07_26", None)
    ledger["_artifact_audit_2026_08_27"] = {
        "checked": "every supporting_artifacts path, resolved against the repository root",
        "missing_count": 0,
        "missing": [],
        "note": (
            "Regenerated after repairing KB-CLAIM-022 and replacing the malformed KB-CLAIM-050 "
            "section pseudo-path with repository-relative files."
        ),
    }

    ledger["reconciliation_source"] = {
        "canonical_panel": source,
        "canonical_panel_sha256": _sha256(PANEL_PATH),
        "source_manifest": source_manifest,
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "comparison_ci_convention": CI_CONVENTION,
        "current_policy_family_sensitivity": {
            "artifact": CURRENT_CLUSTER_PATH.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha256(CURRENT_CLUSTER_PATH),
            "artifact_bytes": CURRENT_CLUSTER_PATH.stat().st_size,
            "status": CURRENT_CLUSTER_STATUS,
            "preregistered_six_comparison_gate": "failed_for_all_candidates",
        },
    }
    ledger["generated_at"] = "2026-08-27"

    closure_claims = [
        {
            "claim_id": "KB-CLAIM-043",
            "claim_text": "The controlled frontier/KGA bridge checks two distinct APIs against the shared benefit target Delta.",
            "claim_type": "protocol",
            "status": "diagnostic",
            "supporting_artifacts": ["experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json"],
            "allowed_wording": "controlled algebraic correspondence check with agreement and disagreement",
            "forbidden_wording": ["real-data beta estimate", "identical abstention sets"],
        },
        {
            "claim_id": "KB-CLAIM-044",
            "claim_text": "No verified unopened natural target was available at the 2026-08-24 prospective-closure audit.",
            "claim_type": "protocol",
            "status": "pending",
            "supporting_artifacts": [
                "experiments/kbound/results/natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json"
            ],
            "allowed_wording": "prospective natural claim awaits a genuinely unopened target",
            "forbidden_wording": ["current natural panel is prospective", "unopened target win"],
        },
        {
            "claim_id": "KB-CLAIM-045",
            "claim_text": "The exact split-conformal confirmation has a deterministic draft manifest but has not been sealed or executed.",
            "claim_type": "protocol",
            "status": "pending",
            "supporting_artifacts": ["research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json"],
            "allowed_wording": "draft disjoint fit/calibration/test confirmation design",
            "forbidden_wording": ["confirmatory result", "sealed preregistration"],
        },
        {
            "claim_id": "KB-CLAIM-046",
            "claim_text": "Official AETTA and POEM native reproduction requirements are not yet satisfied.",
            "claim_type": "protocol",
            "status": "pending",
            "supporting_artifacts": ["experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json"],
            "allowed_wording": "protocol-matched ports; official native reproduction pending",
            "forbidden_wording": ["official implementation result", "official reproduction complete"],
        },
        {
            "claim_id": "KB-CLAIM-047",
            "claim_text": (
                "Camelyon17 OOD exact-rank replay is an opened all-helpful diagnostic: KGA adapts "
                "on all 18 conditions and ties always-adapt, so it is neither prospective nor beats-both."
            ),
            "claim_type": "empirical",
            "claim_tier": "B",
            "protocol": "CAMELYON17_OOD_ARCHIVED_REPLAY",
            "dataset": "Camelyon17 OOD",
            "candidate_adapter": "eata_online",
            "calibration_method": "calibration-record LOO residuals + exact-rank transfer radius",
            "test_split": "opened OOD test conditions, seeds 2-4",
            "status": "diagnostic",
            "supporting_artifacts": [source, source_manifest],
            "assumptions": ["opened target", "all scored conditions favor adaptation"],
            "allowed_wording": "opened all-helpful OOD diagnostic that ties always-adapt",
            "forbidden_wording": ["prospective natural result", "beats both", "non-vacuous safety win"],
        },
        {
            "claim_id": "KB-CLAIM-048",
            "claim_text": (
                "RxRx1 SAR-online exact-rank replay is a harmful-dominated no-harm diagnostic: "
                "KGA freezes on all 60 primary conditions and ties always-freeze; three independently "
                "trained model seeds reproduce that tie."
            ),
            "claim_type": "empirical",
            "claim_tier": "B",
            "protocol": "RXRX1_PROTOCOL_J",
            "dataset": "RxRx1",
            "candidate_adapter": "sar_online",
            "calibration_method": "calibration-record LOO residuals + exact-rank transfer radius",
            "test_split": "test stream seeds 5-9",
            "status": "diagnostic",
            "supporting_artifacts": [source, source_manifest],
            "assumptions": ["harmful-dominated panel", "A7 stability not established"],
            "allowed_wording": "three-model-seed no-harm tie with always-freeze",
            "forbidden_wording": ["beats both", "mixed-regime routing win", "uniform no-harm"],
        },
        {
            "claim_id": "KB-CLAIM-049",
            "claim_text": (
                "CIFAR-10.1 Tent exact-rank replay is a locked negative cross-seed diagnostic: "
                "KGA makes no ADAPT decisions and ties always-freeze on 48 test conditions."
            ),
            "claim_type": "empirical",
            "claim_tier": "B",
            "protocol": "CIFAR10_1_PROTOCOL_K",
            "dataset": "CIFAR-10.1",
            "candidate_adapter": "tent",
            "calibration_method": "calibration-record LOO residuals + exact-rank transfer radius",
            "test_split": "test stream seeds 3-4",
            "status": "diagnostic",
            "supporting_artifacts": [source, source_manifest],
            "assumptions": ["locked cross-seed diagnostic", "A7 stability not established"],
            "allowed_wording": "negative exact-rank diagnostic that ties always-freeze",
            "forbidden_wording": ["FA_u failure", "beats both", "natural-shift win"],
        },
    ]
    by_id = {row["claim_id"]: row for row in ledger["claims"]}
    for row in closure_claims:
        if row["claim_id"] in by_id:
            by_id[row["claim_id"]].update(row)
        else:
            ledger["claims"].append(row)


def _frontier_row(row: dict[str, Any], n: int) -> dict[str, Any]:
    return {
        "regret": row["regret"],
        "adapt_rate": row["adapt_rate"],
        "FA_u": row["fa_u"],
        "FA_c": row["fa_c"],
        "n_adapt": row["adapt_count"],
        "n": n,
        "kappa": row["kappa"],
        "yield": row["yield"],
        "n_freeze": row["freeze_count"],
        "FF_u": 0.0,
    }


def _sync_frontier_dataset(frontier: dict[str, Any], score: dict[str, Any]) -> None:
    n = score["n"]
    sweep = [_frontier_row(row, n) for row in score["kappa_sweep"]]
    frontier["sweep"] = sweep
    frontier["kga_operating_point_kappa1"] = next(row for row in sweep if row["kappa"] == 1.0)
    frontier["trivial_baselines"]["always_adapt"]["regret"] = score["regret"]["always_adapt"]
    frontier["trivial_baselines"]["always_freeze"]["regret"] = score["regret"]["always_freeze"]
    frontier["trivial_baselines"]["always_abstain"]["regret"] = score["regret"]["always_freeze"]


def _sync_frontier(panel: dict[str, Any], frontier_data: dict[str, Any]) -> None:
    panels = panel["panels"]
    for key, panel_key in (("imagenetc", "imagenetc"), ("imagenetr", "imagenet_r")):
        grid = panels[panel_key]["panel"]
        _sync_frontier_dataset(frontier_data["frontier"][key]["__pooled__"], grid["architecture_panel_aggregate"])
        diagnostics = {"__pooled__": grid["architecture_panel_aggregate"]["radius_diagnostics"]}
        diagnostics.update({name: row["radius_diagnostics"] for name, row in grid["candidates"].items()})
        frontier_data["abstention"]["per_cell_tracks"][key] = diagnostics
    frontier_data["reconciliation_source"] = {
        "canonical_panel": PANEL_PATH.relative_to(ROOT).as_posix(),
        "note": "ImageNet-C and ImageNet-R kappa sweeps use source-replayed exact-LOO radii.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="refresh structured release data without writing generated TeX",
    )
    args = parser.parse_args()
    panel = _load(PANEL_PATH)
    current_cluster = _load(CURRENT_CLUSTER_PATH)
    table = _load(TABLE_PATH)
    ledger = _load(LEDGER_PATH)
    frontier = _load(FRONTIER_PATH)
    uniform = _load(UNIFORM_VERDICTS_PATH)
    decision_metrics = _load(DECISION_METRICS_PATH)
    # Validate every candidate before writing any generated release surface.
    for candidate in ("tent", "eata", "sar"):
        _normalized_current_cluster(current_cluster, candidate)
    _sync_table(panel, table, current_cluster)
    _sync_uniform_verdicts(panel, uniform, current_cluster)
    _sync_decision_metrics(panel, decision_metrics, current_cluster)
    _sync_ledger(ledger, current_cluster)
    _sync_frontier(panel, frontier)
    if not args.json_only:
        _write_current_cluster_table(current_cluster)
    _write(TABLE_PATH, table)
    _write(LEDGER_PATH, ledger)
    _write(FRONTIER_PATH, frontier)
    _write(UNIFORM_VERDICTS_PATH, uniform)
    _write(DECISION_METRICS_PATH, decision_metrics)
    print(f"updated {TABLE_PATH.relative_to(ROOT)}")
    print(f"updated {LEDGER_PATH.relative_to(ROOT)}")
    print(f"updated {FRONTIER_PATH.relative_to(ROOT)}")
    print(f"updated {UNIFORM_VERDICTS_PATH.relative_to(ROOT)}")
    print(f"updated {DECISION_METRICS_PATH.relative_to(ROOT)}")
    if not args.json_only:
        print(f"updated {CURRENT_CLUSTER_TABLE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
