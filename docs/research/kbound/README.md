# K-Bound Research Guide

This directory is the maintained research surface for K-Bound and KGA. It contains the compact
manuscript, synchronized long companion, formalization, canonical result manifest, dashboard, and
physical-camera validation package.

> ## State of the project as of 2026-08-29 -- read this before anything else
>
> The maintained release keeps the compact submission and long companion synchronized while
> separating both from historical manuscript and audit records. Older drafts remain for provenance and do not override
> the source-hashed reconciled panel. What a reader arriving today needs to know:
>
> - `kbound_submission.tex` is the primary compact Phase-1 driver. `kbound_tmlr.tex` is the
>   maintained anonymous official-style TMLR driver and consumes the same synchronized
>   `kbound_submission_body.tex`; neither driver is an independent numerical authority.
> - The population frontier uses $(M,\gamma,\beta)$; empirical KGA uses
>   $(\widehat\Delta,\varepsilon)$. Real-data KGA does not numerically receive $\beta$.
> - The canonical empirical panel is
>   `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`, generated from
>   106 source-hashed compact artifacts. Generated LaTeX tables and repeated numbers read this panel.
>   Two later studies use separate receipt-linked authorities rather than being silently folded into
>   that older 106-artifact panel: the locked CCT-20 target result in
>   `paper/generated/cct20_release_manifest.json`, and the stopped So2Sat development result in
>   `experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/`.
> - CIFAR-10-C Tent and EATA beat both fixed policies by current exact-rank point estimate.
>   A retrospective current-policy sensitivity over six corruption families gives Tent positive
>   ordinary bootstrap intervals against both baselines, but the retrospective Holm adjustment over
>   the six prospectively named contrasts gives $p=0.09375$ for both Tent contrasts. The inference
>   is non-confirmatory; no cluster-robust or confirmatory win is claimed.
>   The older cluster artifact remains historical, and the completed SAR rebuild is negative.
> - Office-Home, Camelyon17 OOD, and RxRx1 primarily support one-sided no-harm or endpoint
>   reproduction. The iWildCam numerical/action row is withheld pending an official-metric,
>   population-sealed rerun. PACS, ImageNet-R, and CIFAR-10.1 are retained as null or negative diagnostics.
> - CCT-20 is a prospective safe-utility result: KGA issued 44 FREEZE, zero ADAPT, and one ABSTAIN
>   decision, tied always-freeze, and beat harmful always-adapt. It did not meet the locked strong
>   routing-success criterion because it had no ADAPT exposure.
> - So2Sat is a negative development-stage gate-fit result. Neither Tent nor SAR was feasible, so the
>   protocol stopped before gate calibration. Target inputs were empty and target pixel/label read
>   counts were zero; there is no So2Sat target natural-shift score.
> - No clean single-dataset natural-shift CI-robust beats-both claim and no real-camera result are
>   made. The camera package is a prospective validation protocol.
>
> The single-sentence version: K-Bound has a strong theory and a coherent deployable controller,
> one controlled mixed-regime routing result, and an intentionally narrow natural-shift claim.

## Start Here

| Goal | Entry point |
|---|---|
| Read the Phase-1 provenance and data-quality snapshot | [KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md](KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md) and [audits/empirical_data_quality_2026_08_27/audit_summary.json](audits/empirical_data_quality_2026_08_27/audit_summary.json) — point-in-time authorities for the reconciled panel, not the later CCT-20/So2Sat status |
| **Execute the remaining empirical and release work** | **[KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md](KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md)** |
| Understand current source-hashed result verdicts | [KBOUND_SHORT_RESULT_AUDIT.md](KBOUND_SHORT_RESULT_AUDIT.md) and [KBOUND_SHORT_CLAIM_MANIFEST.md](KBOUND_SHORT_CLAIM_MANIFEST.md) |
| Inspect the receipt-linked CCT-20 authority | [paper/generated/cct20_release_manifest.json](paper/generated/cct20_release_manifest.json) and its [receipt](paper/generated/cct20_release_manifest.json.receipt.json) |
| Inspect the stopped So2Sat development study | [../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md](../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md) |
| Inspect the historical freeze ledger | [SUBMISSION_LEDGER.md](SUBMISSION_LEDGER.md) — superseded for current result verdicts and maintained paths |
| Read the compact submission | [kbound_submission.tex](kbound_submission.tex) and [kbound_submission_body.tex](kbound_submission_body.tex) |
| Read the synchronized TMLR manuscript | [kbound_tmlr.tex](kbound_tmlr.tex) — maintained anonymous official-style driver over [kbound_submission_body.tex](kbound_submission_body.tex) |
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

## Evidence Tiers (reconciled 2026-08-29)

The canonical JSON and generated table are authoritative for the reconciled historical panel.
The later CCT-20 and So2Sat studies remain separate receipt-linked authorities so their prospective
roles and access boundaries are not blurred.

- **Controlled point-estimate routing:** CIFAR-10-C Tent and EATA beat both fixed policies under the
  current exact-rank replay. Tent's retrospective current-policy sensitivity has positive ordinary
  intervals over six observed corruption families, but the retrospective Holm adjustment over the
  six prospectively named contrasts is non-confirmatory.
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
- **Prospective CCT-20 safe utility:** the target study ties always-freeze and protects against
  harmful always-adapt, with zero ADAPT exposure. This is a valid no-harm result, not a beats-both
  routing win.
- **Stopped So2Sat development study:** Tent exposed both helpful and harmful cities, but its LOCO
  router lost 0.1544 percentage points to its best fixed policy; SAR lost 0.0842 points. With no
  feasible candidate, the locked protocol stopped before calibration or target access.
- **Negative diagnostics:** PACS loses to always-adapt, ImageNet-R is worse than adapt on eight of
  ten backbones, and CIFAR-10.1 ties freeze with no adapt decisions.
- **Constructed mixtures:** historical routing aggregates are not promoted as natural-shift wins or
  evidence of transfer to unseen shift families.

## Canonical Build

~~~bash
KBOUND_PYTHON=.venv/bin/python bash runbooks/release_candidate.sh all
~~~

This is the publication gate: it validates authorities, regenerates derived surfaces, runs the
software and formal checks, builds both PDFs and the required compact Word file, renders every PDF
page, and emits the final byte seal. It requires the Python 3.12 `requirements-research.txt` profile
because full test collection imports Torch/WILDS surfaces; it does not launch training.

For a manuscript-only local build after authorities are already validated:

~~~bash
bash scripts/build_pdfs.sh
~~~

Maintained outputs are exactly `kbound_short_final_draft.pdf`, `kbound_tmlr.pdf`, and
`kbound_short_final_draft.docx`. The release driver's `pdf` and `all` modes always request all
three. When `scripts/build_pdfs.sh` is invoked directly, its default builds the compact PDF;
`BUILD_LONG_TMLR=1` also renders the synchronized long companion, and `BUILD_DOCX=1` exports the
Word file. `BUILD_HISTORICAL_TMLR=1` remains a backward-compatible alias for the long-build switch.
Historical compatibility PDFs are not refreshed by this pipeline.
`BUILD_DIAGNOSTIC_IEEE=1` renders the stale shared source through the legacy two-column driver for
diagnostic use only.

The generated result manifest is authoritative for repeated headline values.
Historical notes and archived runs are provenance, not automatic evidence.

The older `scripts/reproduce_submission.sh` and showcase scripts are retained for historical
compatibility. They are not sufficient for a current release PASS; `REPRODUCE.md` records their
known scope and provenance limitations.

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

The primary compact paper is `kbound_submission.tex`; the maintained anonymous official-style TMLR driver is
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
