# K-Bound Research Guide

## Current routing qualification — 2026-09-20

The [living execution ledger](ACTIVE_NATURAL_STUDY_PLAN.md) supersedes the dated
status descriptions below. Historical panel summaries and completed component
checks are not current full-release acceptance. Do not restart SAR or Task3 from
an older command in this guide. The separate painting development attempt is
complete, not prospective routing confirmation.

For the release pipeline, use `bash docs/research/kbound/runbooks/release_candidate.sh all`
only when its source/runtime and resource gates permit. Retired notebooks and
reviewer packets remain in the authenticated archive, not at active paths.
The `archive/stale_publication_builds_2026-09-02/` inventory now preserves ten
historical Git blobs recovered on 2026-09-20. Its old three-artifact contract is
not current release acceptance; current PDFs were not rebuilt or promoted.

Retired local variants are preserved in the separate
[2026-09-20 archive](archive/preserved_worktree_2026-09-20/README.md), including
all six originals that differ from the older archive. Neither archive is a
current release entrypoint.

The two old PDF aliases are also preserved in
[`retired_pdf_aliases_2026-09-20`](archive/retired_pdf_aliases_2026-09-20/README.md).
Use the three canonical outputs above; archival placement does not trim or
rewrite a manuscript.

## Historical guide and evidence descriptions

This directory is the maintained research surface for K-Bound and KGA. It contains the manuscript,
historical extended manuscript, formalization, canonical result manifest, dashboard, and
physical-camera validation package.

> ## State of the project as of 2026-08-20 -- read this before anything else
>
> The maintained release now separates the compact submission, the complete manuscript, and
> historical audit records. Older July status documents remain for provenance and do not override
> the source-hashed reconciled panel. What a reader arriving today needs to know:
>
> - `kbound_submission.tex` builds the maintained compact paper; `kbound_tmlr.tex` builds the full
>   manuscript and supplement. Both use the same canonical numbers and load-bearing theory files.
> - The population frontier uses $(M,\gamma,\beta)$; empirical KGA uses
>   $(\widehat\Delta,\varepsilon)$. Real-data KGA does not numerically receive $\beta$.
> - The canonical empirical panel is
>   `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`, generated from
>   105 source-hashed compact records. Generated LaTeX tables and repeated numbers read this panel.
> - CIFAR-10-C Tent is the strongest routing result. EATA is a point-estimate beats-both result whose
>   adapt-side corruption-cluster interval includes zero. The completed SAR rebuild is negative.
> - Office-Home, iWildCam, Camelyon17 OOD, and RxRx1 primarily support one-sided no-harm or endpoint
>   reproduction. PACS, ImageNet-R, and CIFAR-10.1 are retained as null or negative diagnostics.
> - No clean single-dataset natural-shift CI-robust beats-both claim and no real-camera result are
>   made. The camera package is a prospective validation protocol.
>
> The single-sentence version: K-Bound has a strong theory and a coherent deployable controller,
> one controlled mixed-regime routing result, and an intentionally narrow natural-shift claim.

## Start Here

| Goal | Entry point |
|---|---|
| **Understand the current state and open items** | **[ACTIVE_NATURAL_STUDY_PLAN.md](ACTIVE_NATURAL_STUDY_PLAN.md)** — living execution ledger; a plan is not experiment evidence |
| Read the compact submission | [kbound_submission.tex](kbound_submission.tex) and [kbound_submission_body.tex](kbound_submission_body.tex) |
| Read the maintained full manuscript sources | [kbound_tmlr.tex](kbound_tmlr.tex), [kbound_submission_body.tex](kbound_submission_body.tex), and [kbound_submission_supplement.tex](kbound_submission_supplement.tex) |
| Inspect every canonical panel number | [../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json](../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json) |
| Audit claim-to-artifact links | [KBOUND_SHORT_CLAIM_MANIFEST.md](KBOUND_SHORT_CLAIM_MANIFEST.md) |
| Obtain the datasets | [../../../DATA.md](../../../DATA.md) — per-dataset version, split, licence, acquisition |
| Reproduce the submission | [REPRODUCE.md](REPRODUCE.md), then [REVIEWER_REPRO_PACKET.md](archive/superseded_empirical_authorities_2026-09-02/retired_tree/docs/research/kbound/REVIEWER_REPRO_PACKET.md) (partially superseded) |
| Run an independent replication | [INDEPENDENT_REPLICATION_PROTOCOL.md](INDEPENDENT_REPLICATION_PROTOCOL.md) |
| See what is unreadable and why | [PLACEHOLDER_INVENTORY.md](PLACEHOLDER_INVENTORY.md) |
| See the comparison family and arm inventory | [COMPARISON_FAMILY.md](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/COMPARISON_FAMILY.md) |
| Read the corrected leakage audit | [PHASE6_LEAKAGE_AUDIT.md](PHASE6_LEAKAGE_AUDIT.md) |
| Understand tracked vs external artifacts | [EXTERNAL_STORAGE_POLICY.md](EXTERNAL_STORAGE_POLICY.md) / [STORAGE_MANIFEST.json](STORAGE_MANIFEST.json) |
| Inspect the historical CIFAR SAR quarantine | [CIFAR10C_SAR_QUARANTINE.md](CIFAR10C_SAR_QUARANTINE.md) -- superseded by the completed rebuild |
| Inspect theory-to-code mapping | [THEORY_TO_CODE_MAP.md](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/THEORY_TO_CODE_MAP.md) |
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

- **CI-supported controlled routing:** CIFAR-10-C Tent, five model seeds and 432 cells per seed.
  The comparison survives resampling down to six corruption-family clusters.
- **Point-estimate controlled routing:** CIFAR-10-C EATA beats both fixed policies at the canonical
  operating point, but its adapt-side corruption-family interval includes zero.
- **Completed negative candidate:** CIFAR-10-C SAR has zero observed false adaptations but loses to
  always-adapt. It is not pooled into a candidate-universal claim.
- **Candidate-dependent large-scale corruption:** ImageNet-C SAR has a pooled point edge without a
  promoted CI-robust claim; Tent ties freeze; EATA trails adapt.
- **One-sided natural safety:** primary Office-Home and iWildCam reproduce freeze, Camelyon17 OOD
  reproduces adapt, and RxRx1 freezes throughout. The separate Office-Home replication has a small
  point edge whose seed interval includes zero.
- **Negative diagnostics:** PACS loses to always-adapt, ImageNet-R is worse than adapt on eight of
  ten backbones, and CIFAR-10.1 ties freeze with no adapt decisions.
- **Constructed mixtures:** historical routing aggregates are not promoted as natural-shift wins or
  evidence of transfer to unseen shift families.

The CIFAR-10-C current-policy family sensitivity is retrospective: retrospective Holm adjustment over the six
prospectively named contrasts gives adjusted Tent values of 0.09375 against both fixed policies,
so the result is non-confirmatory. The iWildCam numerical/action row is withheld because the
archived metric contract is not the official WILDS label-present contract; an official-metric,
population-sealed rerun is required before any numerical iWildCam claim can be promoted.

## Canonical Build

~~~bash
# From the repository root, only after its approved source/runtime gates pass:
bash docs/research/kbound/runbooks/release_candidate.sh all
~~~

For a source-validated review build (not a sealed release), build the maintained
compact and TMLR pair with their tables and figures:

~~~bash
BUILD_LONG_TMLR=1 BUILD_DOCX=1 bash docs/research/kbound/scripts/build_pdfs.sh
~~~

Primary outputs: `kbound_short_final_draft.pdf`, `kbound_tmlr.pdf` and
`kbound_short_final_draft.docx`. Historical compatibility output names are not
current deliverables. Page counts and file presence do not establish release sealing.

The generated result manifest is authoritative for repeated headline values.
Historical notes and archived runs are provenance, not automatic evidence.

**Historical caveat recorded 2026-07-26; archived command, do not execute.**
`bash scripts/reproduce_submission.sh` used `set -euo pipefail`, so a
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

The maintained compact submission is `kbound_submission.tex`; the complete single-column manuscript
is `kbound_tmlr.tex`. The old `kbound.tex` and `kbound_short.tex` compatibility
drivers are archived history, not alternate maintained build routes.

Because the venue is TMLR rather than a two-column conference, **no result is cut for length**. The
eight meta-tables the review flagged (`tab:regime-summary`, `tab:data-access`,
`tab:assumptions-role`, `tab:notation-main`, `tab:evidence-map`, `tab:failure-modes`,
`tab:claim-status`, `tab:baseline-faithfulness`) may be merged for readability, not for page count.

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
