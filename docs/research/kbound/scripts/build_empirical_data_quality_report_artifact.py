#!/usr/bin/env python3
"""Build the bounded MCP report artifact for the KBOUND forensic audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "docs/research/kbound/audits/empirical_data_quality_2026_08_27"
OUTPUT = AUDIT_DIR / "artifact.json"
GENERATED_AT = "2026-08-27T00:00:00-05:00"


def csv_rows(name: str) -> list[dict[str, Any]]:
    with (AUDIT_DIR / name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {
        "rank",
        "source_files",
        "record_files",
        "records",
        "missing_metric_records",
        "missing_metadata_dataset_records",
        "strict_json_failures",
        "n",
        "helpful",
        "harmful",
        "tied",
        "adapt_count",
        "freeze_count",
        "abstain_count",
        "model_seed",
        "conditions",
        "eata_sar_exact_prediction_cells",
    }
    float_fields = {
        "regret",
        "regret_kga",
        "regret_best_fixed",
        "mean_freeze_accuracy",
        "mean_sar_accuracy",
        "mean_oracle_accuracy",
        "sar_minus_freeze",
        "oracle_minus_sar",
        "score_out_of_10",
    }
    boolean_fields = {"point_beats_both", "numeric_release_eligible"}
    for row in rows:
        for field in integer_fields:
            if field in row and row[field] != "":
                row[field] = int(row[field])
        for field in float_fields:
            if field in row and row[field] != "":
                row[field] = float(row[field])
        for field in boolean_fields:
            if field in row:
                row[field] = row[field].strip().lower() == "true"
    return rows


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    language: str = "python",
    sql: str | None = None,
) -> dict[str, Any]:
    query = {
        "id": f"audit-2026-08-27-{source_id}",
        "description": description,
        "executed_at": GENERATED_AT,
        "language": language,
        "tables_used": [path],
        "metric_definitions": [
            "Benefit B = adapted score - frozen score.",
            "Regret = per-condition oracle score - policy score; lower is better.",
            "False adaptation = ADAPT with B <= 0.",
        ],
    }
    if sql is not None:
        query["sql"] = sql
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": query,
    }


def main() -> int:
    summary = json.loads((AUDIT_DIR / "audit_summary.json").read_text())
    findings = csv_rows("findings.csv")
    remediation = csv_rows("remediation_status.csv")
    profile = csv_rows("canonical_dataset_profile.csv")
    regret = csv_rows("natural_policy_regret.csv")
    opportunity = csv_rows("natural_opportunity.csv")
    checkpoints = csv_rows("focused_officehome_checkpoints.csv")
    scorecard = csv_rows("reviewer_scorecard.csv")

    accuracy_panels = {
        "CIFAR-10-C Tent (controlled)",
        "Office-Home primary",
        "Office-Home replication",
        "ImageNet-R",
        "PACS",
    }
    accuracy_regret = [
        row for row in regret if row["metric"] == "accuracy" and row["panel"] in accuracy_panels
    ]
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    findings.sort(key=lambda row: (severity_order.get(row["severity"], 9), row["rank"]))
    remediation.sort(key=lambda row: row["rank"])
    profile.sort(key=lambda row: row["records"], reverse=True)
    opportunity.sort(key=lambda row: row["regret_kga"], reverse=True)
    checkpoints.sort(key=lambda row: row["model_seed"])

    sources = [
        source(
            "audit",
            "Forensic audit script and bounded outputs",
            "docs/research/kbound/scripts/audit_empirical_data_quality_2026_08_27.py",
            "Recomputes hashes, row counts, intended-key uniqueness, benefit identities, historical route defects, lineage, metric parity, feasibility, denominators, release seals, and one-to-one remediation dispositions.",
        ),
        source(
            "canonical",
            "Canonical reconciled panel",
            "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
            "Hash-locked score/action panel used for claim-eligible manuscript rows and explicit withholding records.",
            "json",
        ),
        source(
            "source_manifest",
            "Canonical source manifest",
            "experiments/kbound/results/reconciled_panels_v1/source_manifest.json",
            "Provides original and compact SHA-256, byte counts, destinations, and recorded row counts for 106 files.",
            "json",
        ),
        source(
            "route_b",
            "Route-B implementation",
            "experiments/kbound/wilds/analysis.py",
            "Current fail-closed binary-accuracy implementation with bounded anchor orientation, rank/duplicate checks, and explicit unscorable states.",
        ),
        source(
            "route_theory",
            "Route-B numerical validator",
            "experiments/kbound/theory_validation/val_multicandidate_residual.py",
            "Inspected binary assumptions and the b_tilde-based validation decision rule.",
        ),
        source(
            "extractor",
            "Natural multiseed extractor",
            "docs/research/kbound/scripts/extract_multiseed_natural.py",
            "Current atomic lineage, duplicate-key rejection, partition isolation, seed-role, checkpoint, feasibility, and promotion controls.",
        ),
        source(
            "office_runner",
            "Office-Home runner",
            "experiments/kbound/officehome/run_officehome_kbound.py",
            "Current partial-resume identity, role/split isolation, population hash, checkpoint identity, and completion ledger.",
        ),
        source(
            "iwild_runner",
            "iWildCam runner",
            "experiments/kbound/wilds/run_iwildcam_kbound.py",
            "Current official label-present macro-F1, population manifest, resume keys, unsupported Route-B state, and completion ledger.",
        ),
        source(
            "provenance",
            "Natural target provenance audit",
            "experiments/kbound/results/natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json",
            "Inventories target-opening state and evidence hashes for eight natural tracks.",
            "json",
        ),
        source(
            "notebook",
            "Executed companion notebook",
            "docs/research/kbound/notebooks/kbound_empirical_data_quality_audit_2026_08_27.ipynb",
            "Re-executes the audit, assertions, diagnostic replay, paired checkpoint intervals, and policy-regret visualization.",
            "python",
        ),
        source(
            "headline_data",
            "Audit headline metrics",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/audit_summary.json",
            "Selects the initial critical count, quarantined-derivative count, unopened-track count, and unchanged natural-evidence score.",
            "sql",
            """SELECT
  a.critical_findings,
  a.audit_stages.post_remediation.invalid_derived_artifacts_quarantined,
  array_length(a.natural_target_provenance.verified_unopened_tracks) AS verified_unopened_natural_tracks,
  a.audit_stages.post_remediation.natural_shift_routing_evidence_score_out_of_10 AS natural_evidence_score
FROM read_json_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/audit_summary.json') AS a""",
        ),
        source(
            "accuracy_regret_data",
            "Canonical accuracy-regret comparison",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/natural_policy_regret.csv",
            "Selects tidy policy-regret rows for five accuracy panels; protocols remain separate.",
            "sql",
            """SELECT panel, metric, regime, policy, regret, claim_scope, numeric_release_eligible
FROM read_csv_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/natural_policy_regret.csv')
WHERE metric = 'accuracy'
  AND panel IN ('CIFAR-10-C Tent (controlled)', 'Office-Home primary',
                'Office-Home replication', 'ImageNet-R', 'PACS')
ORDER BY panel, policy""",
        ),
        source(
            "findings_data",
            "Ranked forensic findings",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/findings.csv",
            "Returns every immutable initial finding joined to its current remediation and release disposition.",
            "sql",
            "SELECT rank, severity, category, finding, evidence, minimum_safe_fix, remediation_status, release_disposition FROM read_csv_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/findings.csv') ORDER BY rank",
        ),
        source(
            "remediation_data",
            "Post-remediation disposition matrix",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/remediation_status.csv",
            "Maps all 15 initial findings to implemented controls, verification paths, release disposition, and remaining requirement.",
            "sql",
            "SELECT rank, remediation_status, remediation_action, verification_evidence, release_disposition, remaining_requirement FROM read_csv_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/remediation_status.csv') ORDER BY rank",
        ),
        source(
            "profile_data",
            "Canonical dataset profile",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/canonical_dataset_profile.csv",
            "Returns canonical file/row counts and schema-completeness counters by dataset folder.",
            "sql",
            "SELECT dataset, source_files, record_files, records, missing_metric_records, missing_metadata_dataset_records FROM read_csv_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/canonical_dataset_profile.csv') ORDER BY records DESC",
        ),
        source(
            "opportunity_data",
            "Natural opportunity and realized actions",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/natural_opportunity.csv",
            "Returns benefit-sign opportunity, actions, and regrets for selected natural panels.",
            "sql",
            "SELECT panel, n, helpful, harmful, tied, adapt_count, freeze_count, abstain_count, regret_kga, regret_best_fixed, point_beats_both, claim_scope, numeric_release_eligible FROM read_csv_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/natural_opportunity.csv') ORDER BY regret_kga DESC",
        ),
        source(
            "checkpoint_data",
            "Focused Office-Home checkpoint table",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/focused_officehome_checkpoints.csv",
            "Returns one row per distinct checkpoint with fixed, SAR, oracle, and candidate-equality metrics.",
            "sql",
            "SELECT * FROM read_csv_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/focused_officehome_checkpoints.csv') ORDER BY model_seed",
        ),
        source(
            "scorecard_data",
            "Senior-reviewer scorecard",
            "docs/research/kbound/audits/empirical_data_quality_2026_08_27/reviewer_scorecard.csv",
            "Returns the explicitly judgmental empirical-readiness rubric and evidence basis.",
            "sql",
            "SELECT stage, dimension, score_out_of_10, score_status, basis FROM read_csv_auto('docs/research/kbound/audits/empirical_data_quality_2026_08_27/reviewer_scorecard.csv') ORDER BY stage, score_out_of_10 DESC NULLS LAST",
        ),
    ]

    headline = {
        "critical_findings": summary["critical_findings"],
        "invalid_derived_artifacts_quarantined": summary["audit_stages"]["post_remediation"][
            "invalid_derived_artifacts_quarantined"
        ],
        "verified_unopened_natural_tracks": len(summary["natural_target_provenance"]["verified_unopened_tracks"]),
        "natural_evidence_score": summary["audit_stages"]["post_remediation"][
            "natural_shift_routing_evidence_score_out_of_10"
        ],
    }

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "KBOUND Empirical Data-Quality Audit and Remediation Record",
        "description": "Technical record separating the initial forensic findings, current hardening controls, release dispositions, and evidence that still requires a prospective rerun.",
        "generatedAt": GENERATED_AT,
        "sources": sources,
        "cards": [
            {
                "id": "critical_card",
                "dataset": "headline_metrics",
                "sourceId": "headline_data",
                "description": "Defects that invalidate or can manufacture promoted evidence.",
                "metrics": [{"label": "Critical defects", "field": "critical_findings", "format": "number"}],
            },
            {
                "id": "quarantine_card",
                "dataset": "headline_metrics",
                "sourceId": "headline_data",
                "description": "Invalid derived artifacts removed from the release tree with raw sources retained.",
                "metrics": [{"label": "Artifacts quarantined", "field": "invalid_derived_artifacts_quarantined", "format": "number"}],
            },
            {
                "id": "unopened_card",
                "dataset": "headline_metrics",
                "sourceId": "headline_data",
                "description": "Verified unopened natural targets available for a confirmatory claim.",
                "metrics": [{"label": "Unopened natural tracks", "field": "verified_unopened_natural_tracks", "format": "number"}],
            },
            {
                "id": "natural_evidence_card",
                "dataset": "headline_metrics",
                "sourceId": "headline_data",
                "description": "Current natural-shift routing evidence score; unchanged by code remediation because old results are not revalidated.",
                "metrics": [{"label": "Natural evidence / 10", "field": "natural_evidence_score", "format": "number"}],
            },
        ],
        "charts": [
            {
                "id": "accuracy_regret_chart",
                "title": "Accuracy regret by policy and empirical panel",
                "subtitle": "Compare policies within each panel only; lower is better and protocols are not pooled.",
                "type": "bar",
                "dataset": "accuracy_regret",
                "sourceId": "accuracy_regret_data",
                "intent": "comparison",
                "question": "Does KGA reduce regret below both fixed policies within each accuracy panel?",
                "rationale": "Grouped bars expose the controlled CIFAR-10-C strength and the lack of consistent natural-panel dominance.",
                "encodings": {
                    "x": {"field": "panel", "type": "nominal", "label": "Panel"},
                    "y": {"field": "regret", "type": "quantitative", "format": "number", "label": "Regret"},
                    "color": {"field": "policy", "type": "nominal", "label": "Policy"},
                    "tooltip": [
                        {"field": "panel", "type": "nominal", "label": "Panel"},
                        {"field": "policy", "type": "nominal", "label": "Policy"},
                        {"field": "regret", "type": "quantitative", "format": "number", "label": "Regret"},
                    ],
                },
                "layout": "full",
                "maxRows": 50,
                "emptyState": "No accuracy-regret rows are available.",
            }
        ],
        "tables": [
            {
                "id": "scorecard_table",
                "title": "Senior-reviewer empirical scorecard",
                "subtitle": "Judgment rubric; the component scores are not scientific metrics.",
                "dataset": "scorecard",
                "sourceId": "scorecard_data",
                "defaultSort": {"field": "score_out_of_10", "direction": "desc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "stage", "label": "Assessment stage", "type": "text"},
                    {"field": "dimension", "label": "Dimension", "type": "text"},
                    {"field": "score_out_of_10", "label": "Score / 10", "type": "number", "format": "number"},
                    {"field": "score_status", "label": "Score status", "type": "text"},
                    {"field": "basis", "label": "Evidence basis", "type": "text"},
                ],
            },
            {
                "id": "findings_table",
                "title": "Ranked empirical defects and minimum safe fixes",
                "dataset": "findings",
                "sourceId": "findings_data",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "rank", "label": "Rank", "type": "number", "format": "number"},
                    {"field": "severity", "label": "Severity", "type": "text"},
                    {"field": "category", "label": "Category", "type": "text"},
                    {"field": "finding", "label": "Finding", "type": "text"},
                    {"field": "evidence", "label": "Primary evidence", "type": "text"},
                    {"field": "minimum_safe_fix", "label": "Minimum safe fix", "type": "text"},
                    {"field": "remediation_status", "label": "Current status", "type": "text"},
                    {"field": "release_disposition", "label": "Release disposition", "type": "text"},
                ],
            },
            {
                "id": "remediation_table",
                "title": "Current control and remaining-evidence matrix",
                "subtitle": "Every initial finding has exactly one disposition; implemented controls apply to fresh runs only.",
                "dataset": "remediation",
                "sourceId": "remediation_data",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "rank", "label": "Rank", "type": "number", "format": "number"},
                    {"field": "remediation_status", "label": "Remediation status", "type": "text"},
                    {"field": "remediation_action", "label": "Control/action now in place", "type": "text"},
                    {"field": "release_disposition", "label": "Release disposition", "type": "text"},
                    {"field": "remaining_requirement", "label": "Remaining requirement", "type": "text"},
                ],
            },
            {
                "id": "profile_table",
                "title": "Canonical source-bundle profile",
                "subtitle": "Counts are compact canonical rows, not inference units.",
                "dataset": "canonical_profile",
                "sourceId": "profile_data",
                "defaultSort": {"field": "records", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "dataset", "label": "Dataset", "type": "text"},
                    {"field": "source_files", "label": "Manifest files", "type": "number", "format": "number"},
                    {"field": "record_files", "label": "Record files", "type": "number", "format": "number"},
                    {"field": "records", "label": "Records", "type": "number", "format": "number"},
                    {"field": "missing_metric_records", "label": "Missing metric", "type": "number", "format": "number"},
                    {"field": "missing_metadata_dataset_records", "label": "File metadata missing dataset", "type": "number", "format": "number"},
                ],
            },
            {
                "id": "opportunity_table",
                "title": "Natural-shift opportunity and realized actions",
                "subtitle": "Helpful/harmful counts describe candidate opportunity; zero regret in one-sided panels is not routing evidence.",
                "dataset": "natural_opportunity",
                "sourceId": "opportunity_data",
                "defaultSort": {"field": "regret_kga", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "panel", "label": "Panel", "type": "text"},
                    {"field": "n", "label": "n", "type": "number", "format": "number"},
                    {"field": "helpful", "label": "Helpful", "type": "number", "format": "number"},
                    {"field": "harmful", "label": "Harmful", "type": "number", "format": "number"},
                    {"field": "tied", "label": "Tied", "type": "number", "format": "number"},
                    {"field": "adapt_count", "label": "Adapt", "type": "number", "format": "number"},
                    {"field": "freeze_count", "label": "Freeze", "type": "number", "format": "number"},
                    {"field": "abstain_count", "label": "Abstain", "type": "number", "format": "number"},
                    {"field": "regret_kga", "label": "KGA regret", "type": "number", "format": "number"},
                    {"field": "regret_best_fixed", "label": "Best fixed regret", "type": "number", "format": "number"},
                    {"field": "claim_scope", "label": "Claim scope", "type": "text"},
                    {"field": "numeric_release_eligible", "label": "Numeric release eligible", "type": "text"},
                ],
            },
            {
                "id": "checkpoint_table",
                "title": "Focused Office-Home checkpoint audit",
                "subtitle": "Five distinct source checkpoints; three target-domain conditions per checkpoint.",
                "dataset": "focused_checkpoints",
                "sourceId": "checkpoint_data",
                "defaultSort": {"field": "model_seed", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "model_seed", "label": "Model seed", "type": "number", "format": "number"},
                    {"field": "checkpoint_sha256", "label": "Checkpoint SHA-256", "type": "text"},
                    {"field": "conditions", "label": "Conditions", "type": "number", "format": "number"},
                    {"field": "mean_freeze_accuracy", "label": "Freeze accuracy", "type": "number", "format": "number"},
                    {"field": "mean_sar_accuracy", "label": "SAR accuracy", "type": "number", "format": "number"},
                    {"field": "sar_minus_freeze", "label": "SAR - freeze", "type": "number", "format": "number"},
                    {"field": "oracle_minus_sar", "label": "Oracle - SAR", "type": "number", "format": "number"},
                    {"field": "eata_sar_exact_prediction_cells", "label": "EATA=SAR cells", "type": "number", "format": "number"},
                ],
            },
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# KBOUND Empirical Data-Quality Audit and Remediation Record\n\nTechnical senior-reviewer assessment · 27 August 2026",
                "sourceId": "audit",
            },
            {
                "id": "executive_summary",
                "type": "markdown",
                "body": "## Technical Summary\n\nThe initial audit found five critical defects capable of invalidating natural routing evidence. The current tree now fails closed on Route-B orientation and task compatibility, duplicate/stale extraction, resume identity, official-metric parity, infeasible calibration, incomplete cells, and non-finite JSON. Fourteen invalid derived artifacts were quarantined while raw sources were retained. These controls improve future-run validity; they do not repair old results or create a natural-shift win. Natural routing evidence remains 4.0/10, iWildCam numerical/action claims remain withheld, and no verified unopened natural target exists. CIFAR-10-C Tent remains the strongest defensible empirical result.",
                "sourceId": "audit",
            },
            {
                "id": "headline_strip",
                "type": "metric-strip",
                "cardIds": ["critical_card", "quarantine_card", "unopened_card", "natural_evidence_card"],
                "layout": "full",
            },
            {
                "id": "verdict_heading",
                "type": "markdown",
                "body": "## The initial 5.8/10 score is preserved, not relabeled\n\nThe **5.8/10** value is the pre-remediation senior-reviewer baseline. A new overall score is intentionally withheld until the hardened paths produce complete sealed reruns and a genuinely prospective natural evaluation. The current natural-evidence component remains **4.0/10**. A 9–9.5 rigor rating is therefore a future process-and-evidence target, not a score that code cleanup alone can justify.",
                "sourceId": "audit",
            },
            {"id": "scorecard_block", "type": "table", "tableId": "scorecard_table", "layout": "full"},
            {
                "id": "canonical_heading",
                "type": "markdown",
                "body": "## The hash-locked bundle reconciles, but eligibility is claim-specific\n\nThe audit bundle contains 12,619 rows across 10 dataset tracks. All 106 original and compact hashes pass, and no within-file duplicate, core non-finite value, range error, benefit-identity error, or aggregate inconsistency was found. This establishes internal integrity, not universal claim validity: iWildCam remains withheld under its archived metric contract. Schema completeness also remains weaker in legacy sources: 10,653 rows (84.42%) omit an explicit metric, while 8,349 rows are in files whose metadata omits dataset identity.",
                "sourceId": "source_manifest",
            },
            {"id": "profile_block", "type": "table", "tableId": "profile_table", "layout": "full"},
            {
                "id": "comparison_heading",
                "type": "markdown",
                "body": "## Only the controlled Tent panel shows material displayed beats-both value\n\nThe controlled CIFAR-10-C Tent panel is the only displayed accuracy row where KGA is materially below both fixed policies. Office-Home primary ties freeze; the replication has only a small point edge; ImageNet-R and PACS trail always-adapt. Panel protocols are not pooled. iWildCam is excluded because its archived metric contract is invalid for release, not merely because it uses a different metric.",
                "sourceId": "canonical",
            },
            {"id": "regret_chart_block", "type": "chart", "chartId": "accuracy_regret_chart", "layout": "full"},
            {
                "id": "defects_heading",
                "type": "markdown",
                "body": "## The initial defects remain part of the audit trail\n\nAcross 2,892 historical Route-B cells, 2,534 have a b_hat/b_tilde sign disagreement and 1,685 contain an unconstrained b_hat outside [-1,1]. All 138 historical ADAPT cells use a negative spectral anchor, and 105 strictly harmed the realized metric. Separately, the invalid perfect Office-Home multiseed result had 180 stored rows but only 90 unique seed-condition keys. These counts diagnose the old implementation; they are not estimates from the hardened route.",
                "sourceId": "audit",
            },
            {"id": "findings_block", "type": "table", "tableId": "findings_table", "layout": "full"},
            {
                "id": "remediation_heading",
                "type": "markdown",
                "body": "## Every finding now has an explicit disposition\n\nControls are implemented for fresh runs, invalid derivatives are quarantined, the PACS denominator is corrected, the checksum seal remains a final-freeze task, and prospective eligibility still requires a new unopened or hidden-label target. No status below claims that a code fix retroactively validates a historical result.",
                "sourceId": "remediation_data",
            },
            {"id": "remediation_block", "type": "table", "tableId": "remediation_table", "layout": "full"},
            {
                "id": "focused_heading",
                "type": "markdown",
                "body": "## Focused Office-Home is an opportunity diagnostic, not a routing result\n\nThe five checkpoint hashes are distinct, but every historical multicandidate action abstains because tau is 0.776–1.176 against a locked 0.52 threshold. The single-candidate route is structurally infeasible at alpha=.10 with only three cells per checkpoint: exact-rank LOO requires at least 10 total cells. SAR beats freeze by 0.0203 on average, yet the per-cell oracle improves over SAR by only 0.0027. EATA and SAR predictions are identical in 10/15 cells, reducing effective candidate diversity. The hardened serializers now emit null/INFEASIBLE/ABSTAIN instead of a publishable radius for this case.",
                "sourceId": "audit",
            },
            {"id": "checkpoint_block", "type": "table", "tableId": "checkpoint_table", "layout": "full"},
            {
                "id": "opportunity_heading",
                "type": "markdown",
                "body": "## Existing natural panels do not establish a win\n\nSeveral natural panels are one-sided: Camelyon17 OOD is entirely helpful and RxRx1 entirely harmful, so a fixed policy is already oracle. Mixed panels contain opportunity, but the archived/canonical router either abstains, ties a fixed policy, or trails the better fixed policy. The iWildCam official-metric diagnostic was recomputed under the pinned analysis runtime, yet it is still withheld because no sealed official-metric population rerun exists; KGA takes zero adapt actions and ties freeze. The table marks that row numerically ineligible for release.",
                "sourceId": "audit",
            },
            {"id": "opportunity_block", "type": "table", "tableId": "opportunity_table", "layout": "full"},
            {
                "id": "methodology_heading",
                "type": "markdown",
                "body": "## Scope, grain, and validation method\n\nCandidate rows are audited at dataset × checkpoint/model seed × stream seed × domain/location/backbone × split × composition × regime × aggressiveness × candidate. Route rows remove candidate. The audit uses strict JSON parsing including decoded overflow, SHA-256 and byte-count checks, composite-key uniqueness, B = adapted − frozen, range and finiteness checks, action-count/rate reconciliation, historical route reconstruction, official metric replay, candidate-prediction equality, exact-rank feasibility, quarantine verification, and release-checksum verification. The executed notebook reruns the bounded checks and asserts one remediation row per finding.",
                "sourceId": "notebook",
            },
            {
                "id": "limitations_heading",
                "type": "markdown",
                "body": "## Limitations: controls are stronger than the current evidence\n\nThe corrected iWildCam replay now executes under the pinned NumPy/scikit-learn analysis runtime, but its exact decimals remain diagnostic because it is a retrospective recomputation rather than a sealed official-metric run over a frozen population. Historical Route-B counts include archived copies and establish implementation scope, not a population effect. Current natural targets are opened and cannot support a new confirmatory claim. The scorecard is reviewer judgment, not a formal metric; the post-remediation overall score is intentionally withheld.",
                "sourceId": "audit",
            },
            {
                "id": "recovery_heading",
                "type": "markdown",
                "body": "## Remaining release and evidence work\n\n1. Keep the 14 invalid derivatives quarantined and preserve the raw-source/hash record.\n2. Run the focused regression suite and smoke the hardened runners only in fresh directories; do not promote historical Route-B or contaminated-resume outputs.\n3. Complete the final PDF/manuscript freeze, regenerate the outer SHA-256 seal, and rerun this audit until every seal entry matches.\n4. Freeze candidates and thresholds using source/development data only; require unique prediction rank, feasible exact-rank calibration, complete ledgers, and population/checkpoint hashes before a target run.\n5. Preserve CIFAR-10-C Tent as the controlled headline and present existing natural results as transparent diagnostic/null/boundary evidence.\n6. For a natural beats-both claim, evaluate one genuinely new cohort or hidden-label target after sealing code, checkpoints, sample IDs, population hashes, and the inference plan.",
                "sourceId": "audit",
            },
            {
                "id": "questions_heading",
                "type": "markdown",
                "body": "## Further questions\n\n- Can a source-only diversity screen produce a sufficiently ranked candidate set without target-label selection?\n- Is there a theory-backed multiclass replacement for Route B that targets the deployment metric directly?\n- Which external cohort or hidden-label benchmark can remain unopened until code and thresholds are sealed?\n- Can independent model checkpoints be generated under one frozen training contract for model-level uncertainty?",
                "sourceId": "audit",
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline_metrics": [headline],
                "accuracy_regret": accuracy_regret,
                "findings": findings,
                "remediation": remediation,
                "canonical_profile": profile,
                "natural_opportunity": opportunity,
                "focused_checkpoints": checkpoints,
                "scorecard": scorecard,
            },
        },
        "sources": sources,
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
