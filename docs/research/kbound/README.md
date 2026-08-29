# K-Bound Research Guide

This directory is the maintained research surface for K-Bound and KGA. It contains the compact
manuscript, synchronized long companion, formalization, canonical result manifest, dashboard, and
physical-camera validation package.

> ## State of the project as of 2026-08-27 -- read this before anything else
>
> The maintained release keeps the compact submission and long companion synchronized while
> separating both from historical manuscript and audit records. Older drafts remain for provenance and do not override
> the source-hashed reconciled panel. What a reader arriving today needs to know:
>
> - `kbound_submission.tex` is the primary compact Phase-1 driver. `kbound_tmlr.tex` is the
>   maintained single-column long companion and consumes the same synchronized
>   `kbound_submission_body.tex`; neither driver is an independent numerical authority.
> - The population frontier uses $(M,\gamma,\beta)$; empirical KGA uses
>   $(\widehat\Delta,\varepsilon)$. Real-data KGA does not numerically receive $\beta$.
> - The canonical empirical panel is
>   `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`, generated from
>   106 source-hashed compact artifacts. Generated LaTeX tables and repeated numbers read this panel.
> - CIFAR-10-C Tent and EATA beat both fixed policies by current exact-rank point estimate.
>   A retrospective current-policy sensitivity over six corruption families gives Tent positive
>   ordinary bootstrap intervals against both baselines, but the preregistered six-comparison Holm gate fails ($p=0.09375$ for both Tent contrasts). No cluster-robust or confirmatory win is claimed.
>   The older cluster artifact remains historical, and the completed SAR rebuild is negative.
> - Office-Home, Camelyon17 OOD, and RxRx1 primarily support one-sided no-harm or endpoint
>   reproduction. The iWildCam numerical/action row is withheld pending an official-metric,
>   population-sealed rerun. PACS, ImageNet-R, and CIFAR-10.1 are retained as null or negative diagnostics.
> - No clean single-dataset natural-shift CI-robust beats-both claim and no real-camera result are
>   made. The camera package is a prospective validation protocol.
>
> The single-sentence version: K-Bound has a strong theory and a coherent deployable controller,
> one controlled mixed-regime routing result, and an intentionally narrow natural-shift claim.

## Start Here

| Goal | Entry point |
|---|---|
| Read the current provenance and data-quality verdict | [KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md](KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md) and [audits/empirical_data_quality_2026_08_27/audit_summary.json](audits/empirical_data_quality_2026_08_27/audit_summary.json) |
| **Execute the remaining empirical and release work** | **[KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md](KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md)** |
| Understand current source-hashed result verdicts | [KBOUND_SHORT_RESULT_AUDIT.md](KBOUND_SHORT_RESULT_AUDIT.md) and [KBOUND_SHORT_CLAIM_MANIFEST.md](KBOUND_SHORT_CLAIM_MANIFEST.md) |
| Inspect the historical freeze ledger | [SUBMISSION_LEDGER.md](SUBMISSION_LEDGER.md) — superseded for current result verdicts and maintained paths |
| Read the compact submission | [kbound_submission.tex](kbound_submission.tex) and [kbound_submission_body.tex](kbound_submission_body.tex) |
| Read the synchronized long companion | [kbound_tmlr.tex](kbound_tmlr.tex) — maintained single-column rendering of [kbound_submission_body.tex](kbound_submission_body.tex) |
| Inspect every canonical panel number | [../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json](../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json) |
| Audit claim-to-artifact links | [KBOUND_SHORT_CLAIM_MANIFEST.md](KBOUND_SHORT_CLAIM_MANIFEST.md) |
| Obtain the datasets | [../../../DATA.md](../../../DATA.md) — per-dataset version, split, licence, acquisition |
| Reproduce the submission | [REPRODUCE.md](REPRODUCE.md), then [REVIEWER_REPRO_PACKET.md](REVIEWER_REPRO_PACKET.md) (partially superseded) |
| Run an independent replication | [INDEPENDENT_REPLICATION_PROTOCOL.md](INDEPENDENT_REPLICATION_PROTOCOL.md) |
| See what is unreadable and why | [PLACEHOLDER_INVENTORY.md](PLACEHOLDER_INVENTORY.md) |
| Inspect the historical comparison-family census | [COMPARISON_FAMILY.md](COMPARISON_FAMILY.md) — superseded; not current multiplicity evidence |
| Read the corrected leakage audit | [PHASE6_LEAKAGE_AUDIT.md](PHASE6_LEAKAGE_AUDIT.md) |
| Understand tracked vs external artifacts | [EXTERNAL_STORAGE_POLICY.md](EXTERNAL_STORAGE_POLICY.md) / [STORAGE_MANIFEST.json](STORAGE_MANIFEST.json) |
| Inspect the historical CIFAR SAR quarantine | [CIFAR10C_SAR_QUARANTINE.md](CIFAR10C_SAR_QUARANTINE.md) -- superseded by the completed rebuild |
| Inspect theory-to-code mapping | [THEORY_TO_CODE_MAP.md](THEORY_TO_CODE_MAP.md) |
| Build the research dashboard | [dashboard/README.md](dashboard/README.md) |
| Start the physical study | [edge/PHYSICAL_STUDY_RUNBOOK.md](edge/PHYSICAL_STUDY_RUNBOOK.md) |
| Verify Lean files | [formal/README.md](formal/README.md) |

## Fixed Terminology

- **K-Bound**: population theory and the adapt/freeze/abstain framework.
- **KGA**: finite-sample empirical wrapper around a candidate adapter.
- **Population frontier**: M, gamma, and beta.
- **Empirical certificate**: Delta_hat and epsilon.
- **Abstain**: do not commit the update; continue prediction with the frozen fallback.

The population frontier and empirical certificate are related but distinct.
Real-data KGA does not receive beta, and empirical abstention does not by itself
prove structural non-identifiability.

## Evidence Tiers (reconciled 2026-08-20)

The canonical JSON and generated table are authoritative for current point estimates.

- **Controlled point-estimate routing:** CIFAR-10-C Tent and EATA beat both fixed policies under the
  current exact-rank replay. Tent's retrospective current-policy sensitivity has positive ordinary
  intervals over six observed corruption families, but the preregistered six-comparison Holm gate fails.
  The within-Tent two-contrast Holm value is post hoc; historical earlier-policy cluster
  evidence remains separate.
- **Completed negative candidate:** CIFAR-10-C SAR has zero observed false adaptations but loses to
  always-adapt. It is not pooled into a candidate-universal claim.
- **Candidate-dependent large-scale corruption:** ImageNet-C SAR has a pooled point edge without a
  promoted CI-robust claim; Tent ties freeze; EATA trails adapt.
- **One-sided natural diagnostics:** primary Office-Home reproduces freeze, Camelyon17 OOD reproduces
  adapt, and RxRx1 freezes throughout. The iWildCam numerical/action row is withheld pending an
  official-metric, population-sealed rerun. The separate Office-Home replication has a small point
  edge whose seed interval includes zero.
- **Negative diagnostics:** PACS loses to always-adapt, ImageNet-R is worse than adapt on eight of
  ten backbones, and CIFAR-10.1 ties freeze with no adapt decisions.
- **Constructed mixtures:** historical routing aggregates are not promoted as natural-shift wins or
  evidence of transfer to unseen shift families.

## Canonical Build

~~~bash
bash scripts/reproduce_submission.sh
bash scripts/build_dashboard.sh
~~~

Build the compact submission, canonical tables, and figures:

~~~bash
bash scripts/build_pdfs.sh
~~~

Maintained outputs are exactly `kbound_short_final_draft.pdf`, `kbound_tmlr.pdf`, and
`kbound_short_final_draft.docx`. The default command builds the compact PDF;
`BUILD_LONG_TMLR=1` also renders the synchronized long companion, and `BUILD_DOCX=1` exports the
Word file. `BUILD_HISTORICAL_TMLR=1` remains a backward-compatible alias for the long-build switch.
Historical compatibility PDFs are not refreshed by this pipeline.
`BUILD_DIAGNOSTIC_IEEE=1` renders the stale shared source through the legacy two-column driver for
diagnostic use only.

The generated result manifest is authoritative for repeated headline values.
Historical notes and archived runs are provenance, not automatic evidence.

**Caveat added 2026-07-26.** `bash scripts/reproduce_submission.sh` uses `set -euo pipefail`, so a
failure in step 1 silently prevents steps 2-9 from running; and several of its checks reference
files that are absent or unreadable. Read `REPRODUCE.md §1`'s "Known failures" box before treating
a green run as a clean bill.

## Dashboard

~~~bash
bash scripts/build_dashboard.sh
python3 -m http.server 8765 --directory .
~~~

Open http://127.0.0.1:8765/kbound_dashboard.html.

The dashboard reads the canonical paper manifest and the active
experiments/kbound/results/edge_real_phone_v1 tree. It never reads
archive/legacy_elara.

## Physical Validation

The edge code is a maintained, tested module rather than an informal demo. The
publication workflow is:

1. Prepare the protocol lock and deterministic checklists.
2. Capture S01-S02 and pass the source-model quality gate.
3. Capture S03-S06 and seal development plus conformal calibration.
4. Open S07-S08 once for held-out Phone A evaluation.
5. Capture S09-S10 on Phone B for replication.
6. Run the strict anti-leakage and publication gates.
7. Export camera tables and refresh the dashboard.

Start with:

~~~bash
python edge/scripts/preflight_r2.py
~~~

Browser preview, simulation, pilot data, and mock captures are connectivity or
software tests only. They cannot satisfy the publication gate.

## Formalization

~~~bash
cd formal
bash build.sh
~~~

The theorem map reports exactly which Lean declarations correspond to paper
statements. Do not describe the repository as a full foundational Mathlib
development: several measure-theoretic and deployment assumptions remain
external.

## Manuscript Policy

The primary compact paper is `kbound_submission.tex`; the maintained single-column companion is
`kbound_tmlr.tex`. Both consume `kbound_submission_body.tex`, so empirical corrections propagate to
both outputs. The long driver must not input the stale `kbound_short_body.tex` or
`kbound_short_appendix.tex`. Use `BUILD_LONG_TMLR=1` when the synchronized long artifact is
required.

A balanced version should retain:

- problem and validity boundary;
- three core theory results plus the multiclass bridge;
- KGA architecture and calibration protocol;
- controlled beats-both evidence;
- natural no-harm and negative evidence;
- concise limitations and reproducibility.

Keep extended minimax, one-bit, martingale, historical ELARA, and large
diagnostic ladders in the supplement unless a venue explicitly allows them.
The detailed keep/move policy is in [KBOUND_MANUSCRIPT_STRATEGY.md](KBOUND_MANUSCRIPT_STRATEGY.md).
