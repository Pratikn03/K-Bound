#!/usr/bin/env python3
"""Build and execute the KBOUND empirical data-quality companion notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "docs/research/kbound/notebooks/kbound_empirical_data_quality_audit_2026_08_27.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            """
# KBOUND empirical data-quality audit and remediation record

**Forensic companion notebook · 27 August 2026**

This notebook reads the frozen forensic outputs and refreshes only bounded release metadata. It keeps two stages separate: the initial findings and the post-remediation control state. Historical defects remain visible even when code is fixed, because a fix cannot retroactively validate an affected result. Diagnostic corrections are never promoted as confirmatory evidence.
"""
        ),
        markdown(
            """
## tl;dr

The initial audit found five critical defects that could corrupt or manufacture natural routing evidence. The current tree now fails closed on Route-B orientation/task mismatch, duplicate/stale extraction, resume contamination, metric mismatch, infeasible calibration, incomplete runs, and non-finite JSON. Fourteen invalid derived artifacts were quarantined while raw sources were retained. These controls improve future-run integrity; they do **not** create a natural-shift win. Natural routing evidence remains **4.0/10**, iWildCam numerical/action claims remain withheld, and no verified unopened natural target exists. CIFAR-10-C Tent remains the strongest defensible empirical result.
"""
        ),
        code(
            """
from pathlib import Path
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy import stats

ROOT = Path.cwd().resolve()
if not (ROOT / "docs/research/kbound").exists():
    raise RuntimeError(f"Run from the repository root; cwd={ROOT}")

AUDIT_SCRIPT = ROOT / "docs/research/kbound/scripts/audit_empirical_data_quality_2026_08_27.py"
AUDIT_DIR = ROOT / "docs/research/kbound/audits/empirical_data_quality_2026_08_27"
subprocess.run(
    [
        sys.executable,
        str(AUDIT_SCRIPT),
        "--out-dir",
        str(AUDIT_DIR),
        "--wording-only",
    ],
    check=True,
)

summary = json.loads((AUDIT_DIR / "audit_summary.json").read_text())
findings = pd.read_csv(AUDIT_DIR / "findings.csv")
remediation = pd.read_csv(AUDIT_DIR / "remediation_status.csv")
profile = pd.read_csv(AUDIT_DIR / "canonical_dataset_profile.csv")
regret = pd.read_csv(AUDIT_DIR / "natural_policy_regret.csv")
opportunity = pd.read_csv(AUDIT_DIR / "natural_opportunity.csv")
checkpoints = pd.read_csv(AUDIT_DIR / "focused_officehome_checkpoints.csv")
scorecard = pd.read_csv(AUDIT_DIR / "reviewer_scorecard.csv")

display(Markdown(
    f"**Executed with:** `{Path(sys.executable).name}`  \\n"
    f"**Canonical SHA-256:** `{summary['canonical']['canonical_sha256']}`  \\n"
    f"**Source-manifest SHA-256:** `{summary['canonical']['source_manifest_sha256']}`"
))
"""
        ),
        markdown(
            """
## Context & Methods

The intended candidate grain is dataset × checkpoint/model seed × stream seed × domain/location/backbone × split × composition × regime × aggressiveness × candidate. A route row removes the candidate dimension. Inference units, repeated condition evaluations, paired twins, and stream seeds are audited separately.

Checks include strict JSON parsing (including decoded overflow), SHA-256 and byte-count verification, recorded row counts, within-file key uniqueness, `B = adapted − frozen`, score/range checks, action-count and regret reconciliation, route-estimator orientation, resume-role isolation, extraction lineage, calibration feasibility, official metric parity, and release seals. Each initial finding is joined one-to-one to a remediation status, release disposition, verification path, and remaining requirement. The corrected iWildCam replay remains diagnostic even when executed in the pinned reconciliation runtime because it is not a sealed official-metric rerun over a frozen population.
"""
        ),
        code(
            """
grain = pd.DataFrame(
    [{"object": key, "intended_grain_or_rule": value} for key, value in summary["intended_grain"].items()]
)
display(grain)

assert summary["canonical"]["source_manifest_hash_matches_canonical"]
assert summary["canonical"]["aggregate_score_checks"]["problem_count"] == 0
assert summary["bottom_line"]["defensible_natural_beats_both_win"] is False
assert summary["bottom_line"]["controlled_cifar10c_retrospective_six_contrast_holm_win"] is False
assert summary["bottom_line"]["code_hardening_retroactively_repairs_historical_results"] is False
assert len(remediation) == len(findings) == 15
assert set(remediation["rank"]) == set(findings["rank"]) == set(range(1, 16))
assert summary["audit_stages"]["post_remediation"]["natural_shift_routing_evidence_score_out_of_10"] == 4.0
print("Core audit and remediation assertions passed.")
"""
        ),
        markdown(
            """
## Data

The hash-locked analytical base is `experiments/kbound/results/reconciled_panels_v1`. It contains 106 source-manifest entries, including 101 record files and five summary artifacts. Internal reconciliation establishes integrity, not universal claim eligibility: iWildCam remains explicitly withheld under its archived metric contract. The audit also reads archived `result_*.json` files only to identify historical pipeline defects; those artifacts are not silently promoted into release estimates.
"""
        ),
        code(
            """
bundle = summary["canonical"]["bundle"]
integrity = pd.DataFrame([
    {"check": "Original source hash failures", "value": bundle["original_hash_failures"]},
    {"check": "Compact hash failures", "value": bundle["compact_hash_failures"]},
    {"check": "Recorded row-count failures", "value": bundle["row_count_failures"]},
    {"check": "Exact duplicate records within a compact file", "value": bundle["exact_duplicate_records_within_file"]},
    {"check": "Duplicate intended keys within a compact file", "value": bundle["duplicate_dimension_keys_within_file"]},
    {"check": "Benefit identity failures", "value": bundle["benefit_identity_failures"]},
    {"check": "Core non-finite values", "value": bundle["core_nonfinite_values"]},
    {"check": "Aggregate score nodes checked", "value": summary["canonical"]["aggregate_score_checks"]["score_nodes_checked"]},
    {"check": "Aggregate score problems", "value": summary["canonical"]["aggregate_score_checks"]["problem_count"]},
])
display(integrity)
display(profile.sort_values("records", ascending=False).reset_index(drop=True))

print(f"Missing explicit metric: {bundle['missing_metric_records']:,}/{bundle['records_total']:,} "
      f"({bundle['missing_metric_rate']:.2%})")
print(f"Rows whose file metadata lacks dataset: {bundle['records_missing_metadata_dataset']:,}/{bundle['records_total']:,} "
      f"({bundle['records_missing_metadata_dataset_rate']:.2%})")
"""
        ),
        markdown(
            """
## Results

The canonical bundle passes its internal integrity checks. That does not validate every algorithm or historical artifact. The table below preserves the initial failures and then shows the present control state. “Implemented” means a new run fails closed under the repaired contract; it never means the old result was repaired.
"""
        ),
        code(
            """
severity_order = pd.CategoricalDtype(["Critical", "High", "Medium", "Low"], ordered=True)
findings_view = findings.copy()
findings_view["severity"] = findings_view["severity"].astype(severity_order)
display(findings_view[[
    "rank", "severity", "category", "finding", "remediation_status", "release_disposition"
]].sort_values(["severity", "rank"]).reset_index(drop=True))

display(remediation[[
    "rank", "remediation_status", "remediation_action", "remaining_requirement"
]].sort_values("rank").reset_index(drop=True))

route = summary["route_b_archive"]
route_table = pd.DataFrame([
    {"measure": "Archived route cells", "count": route["route_cells"], "denominator": route["route_cells"]},
    {"measure": "b_hat / b_tilde sign-disagreement cells", "count": route["b_hat_b_tilde_sign_disagreement_cells"], "denominator": route["route_cells"]},
    {"measure": "b_hat outside [-1,1] cells", "count": route["b_hat_outside_unit_interval_cells"], "denominator": route["route_cells"]},
    {"measure": "ADAPT cells", "count": route["adapt_cells"], "denominator": route["route_cells"]},
    {"measure": "ADAPT cells with negative spectral anchor", "count": route["adapt_negative_anchor_cells"], "denominator": route["adapt_cells"]},
    {"measure": "ADAPT cells with strictly harmful realized benefit", "count": route["adapt_negative_realized_benefit_cells"], "denominator": route["adapt_cells"]},
])
route_table["rate"] = route_table["count"] / route_table["denominator"]
display(route_table)
"""
        ),
        code(
            """
lineage = summary["multiseed_lineage"]
resume = summary["resume_contamination"]
metric = summary["iwild_metric"]

diagnostics = pd.DataFrame([
    {"diagnostic": "Office-Home invalid extracted rows", "value": lineage["officehome"]["rows"], "interpretation": "Historical evidence retained in quarantine manifest"},
    {"diagnostic": "Office-Home unique seed-condition keys", "value": lineage["officehome"]["unique_seed_condition_keys"], "interpretation": "90 unique among 180 historical rows"},
    {"diagnostic": "Office-Home reported oracle-action matches", "value": lineage["officehome"]["oracle_action_matches"], "interpretation": "Invalid perfect result under duplicate leakage"},
    {"diagnostic": "Non-test rows in target-test resume artifact", "value": resume["non_test_conditions_in_target_test_artifact"], "interpretation": f"{resume['non_test_fraction']:.1%} of artifact"},
    {"diagnostic": "iWild full-test benefit sign flips after official metric correction", "value": metric["full_test_864_record_diagnostic"]["benefit_sign_flips"], "interpretation": "60/864 candidate cells"},
    {"diagnostic": "Invalid derived artifacts quarantined", "value": summary["audit_stages"]["post_remediation"]["invalid_derived_artifacts_quarantined"], "interpretation": "Raw source runs retained"},
    {"diagnostic": "Release checksum mismatches", "value": summary["release_checksums"]["status_counts"].get("mismatch", 0), "interpretation": "Final reseal remains a release-stage task"},
])
display(diagnostics)

replay = metric["corrected_transfer_replay"]
display(pd.DataFrame([{
    "diagnostic_replay": "iWild official-metric correction",
    "epsilon": replay.get("epsilon"),
    "adapt": replay.get("action_counts", {}).get("adapt"),
    "freeze": replay.get("action_counts", {}).get("freeze"),
    "abstain": replay.get("action_counts", {}).get("abstain"),
    "regret_kga": replay.get("regret", {}).get("kga"),
    "regret_best_fixed": min(replay.get("regret", {"x": np.nan}).values()),
    "beats_both": replay.get("point_beats_both"),
    "status": replay.get("status"),
}]))
"""
        ),
        code(
            """
# Same-metric view: accuracy regret only. Compare policies within a panel; protocols and units differ.
plot_rows = regret[(regret["metric"] == "accuracy") & regret["panel"].isin([
    "CIFAR-10-C Tent (controlled)",
    "Office-Home primary",
    "Office-Home replication",
    "ImageNet-R",
    "PACS",
])]
pivot = plot_rows.pivot(index="panel", columns="policy", values="regret")
pivot = pivot[["KGA", "Always adapt", "Always freeze"]]

ax = pivot.plot(kind="bar", figsize=(11, 5.5), color=["#2f6f8f", "#d98c3f", "#8b8f97"])
ax.set_title("Accuracy regret to the per-condition oracle")
ax.set_ylabel("Regret (lower is better)")
ax.set_xlabel("")
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Policy", frameon=False)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

display(opportunity.sort_values("panel").reset_index(drop=True))
"""
        ),
        code(
            """
display(checkpoints)
focused = summary["focused_officehome"]
sar_gain = checkpoints["sar_minus_freeze"].to_numpy(float)
oracle_edge = checkpoints["oracle_minus_sar"].to_numpy(float)

def t_interval(values, confidence=0.95):
    values = np.asarray(values, float)
    mean = values.mean()
    half = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * stats.sem(values)
    return mean, mean - half, mean + half

sar_ci = t_interval(sar_gain)
oracle_ci = t_interval(oracle_edge)
focused_view = pd.DataFrame([
    {"contrast": "SAR minus freeze", "mean": sar_ci[0], "ci95_low": sar_ci[1], "ci95_high": sar_ci[2]},
    {"contrast": "Oracle minus SAR", "mean": oracle_ci[0], "ci95_low": oracle_ci[1], "ci95_high": oracle_ci[2]},
])
display(focused_view)

assert focused["route_action_counts"] == {"ABSTAIN": 15}
assert focused["single_candidate_route_feasible"] is False
assert focused["tau_min"] > focused["locked_tau_star"]
print(f"Strict-JSON failures: {focused['strict_json_files_failed']}/5 files; "
      f"literal Infinity values: {focused['literal_infinity_values']}")
"""
        ),
        code(
            """
display(scorecard)
post = summary["audit_stages"]["post_remediation"]
print(
    "Post-remediation overall score:", post["overall_score_status"],
    "| natural evidence remains", post["natural_shift_routing_evidence_score_out_of_10"], "/10"
)
"""
        ),
        markdown(
            """
## Takeaways

1. **Quarantine is complete for the invalid derived set.** Fourteen duplicate-leaked, stale-lineage, or downstream presentation artifacts are outside the release tree; raw source runs and hashes are retained for audit.
2. **The repaired paths now fail closed.** Scientific-config resume hashes, official metric parity, strict error/completeness ledgers, candidate rank and feasibility checks, atomic lineage, strict JSON, and explicit inference-unit fields protect fresh runs.
3. **Historical evidence remains historical.** Old Route-B, contaminated resume, iWildCam metric, and infeasible-calibration results stay non-promotable until rerun under the new contract.
4. **No natural win is claimed.** All current natural targets are opened, so they support only transparent diagnostic, null, or boundary statements. The natural-shift evidence score remains **4.0/10**.
5. **Preserve the bounded controlled result.** CIFAR-10-C Tent has a beats-both point estimate and positive ordinary six-family intervals, but p-values from retrospective Holm adjustment over the six prospectively named contrasts are both 0.09375. It is not a cluster-robust or confirmatory win.
6. **A new overall score is withheld.** The initial **5.8/10** readiness judgment is retained as a historical baseline, not relabeled as current. A 9–9.5 rigor score would require complete hardened reruns, a final checksum seal, and a genuinely new or hidden-label natural evaluation.
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "KBOUND audit (Python 3.12)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "audit": {
                "source_script": "docs/research/kbound/scripts/audit_empirical_data_quality_2026_08_27.py",
                "canonical_source": "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
                "generated_at": "2026-08-27T00:00:00-05:00",
            },
        },
    )
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute(cwd=str(ROOT))
    nbf.write(notebook, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
