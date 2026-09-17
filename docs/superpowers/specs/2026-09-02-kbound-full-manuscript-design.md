# Corrected Full K-Bound Manuscript Design

**Date:** 2026-09-02

**Status:** Proposed for implementation

**Scope:** `docs/research/kbound/`

## Goal

Create a new authoritative full-length K-Bound manuscript from the maintained, audited paper sources. The manuscript should be as long as the validated scientific content requires; it must not be padded to reproduce the historical 90--94 page count.

The historical `kbound.tex` and its old PDFs remain preserved as explicitly superseded archives. They are not build inputs and are not overwritten. The current compact and TMLR manuscripts remain unchanged as submission artifacts. The new full manuscript is a companion artifact with a distinct source driver and output PDF.

## Source of truth

The full manuscript will inherit its scientific core from the maintained sources:

- `kbound_submission_body.tex`
- `kbound_submission_supplement.tex`
- the current bibliography and shared LaTeX assets
- canonical result JSON, claim ledgers, audit notes, and `DOCS_INDEX.md`

When historical prose conflicts with these sources, the maintained source and canonical result artifacts win. Material from the superseded 94-page manuscript may be restored only after it is checked against current theory, experiments, terminology, and provenance.

## Manuscript structure

Add a dedicated full-paper driver, `kbound_full.tex`, which reuses the maintained shared body and supplement. Add a full-only supplement file if necessary for validated material that is useful in the long version but too detailed for the compact submissions.

The design intentionally avoids copying the historical monolithic manuscript. Shared current sections should remain shared so corrections propagate to the compact, TMLR, and full builds without claim drift.

## Required scientific correction: beta-estimation failure

Section 9's paragraph on the external population budget will explicitly report the negative beta-surrogate experiment. The text must preserve the supplied numbers but accurately distinguish the tested operational surrogate from the theorem's population quantity.

Proposed wording:

> We tested a source-development benefit-scale proxy as an operational surrogate for beta rather than declaring the population budget directly; this proxy does not estimate the theorem's disagreement-conditional calibration-residual bound. On CIFAR-10-C, the surrogate is 1.4--50 times too small on real evaluation cells: 24--73% of cells fall outside the corresponding declared class C_beta, and commit error is 0.4--16.6%. On ImageNet-C, 5 of 10 configurations return zero commitments on all 405 evaluation cells. We therefore withdraw this surrogate estimation route and declare beta from domain knowledge or an explicit transfer assumption instead. The empirical KGA does not estimate or numerically use beta.

This is deliberately more precise than describing the experiment as a direct estimate of beta. The available artifacts measure a source-development benefit-scale proxy and do not contain the target disagreement-conditional calibration residual needed to estimate the theorem's beta literally.

## Validated long-form additions

The full-only material may include the following, provided every number is rechecked against its authoritative artifact before insertion:

1. **Population-to-KGA bridge.** A worked seven-example controlled illustration with five agreement cases and two disagreement cases, used only to explain the relationship between the population objects `(M, beta)` and the empirical objects `(Delta_hat, epsilon)`. It must be labeled as an illustration, not evaluation evidence.
2. **Controlled two-view MNIST diagnostic (D33).** A supplement-only report of 130 conditions, 9 ADAPT, 119 FREEZE, and 2 ABSTAIN decisions; KGA accuracy 85.6785%, single-A accuracy 85.3554%, and always-fuse accuracy 58.3231%, with zero observed false ADAPT decisions among the nine ADAPT decisions. It must be labeled controlled evidence and must not be generalized to natural distribution shifts.
3. **Audit and reproducibility detail.** Additional provenance, claim-to-artifact mapping, theorem dependency notes, and reproducibility instructions, but only where they reflect the current repository state.

These additions are candidates, not permission to import surrounding historical claims wholesale.

## Explicit exclusions and claim controls

The full manuscript must not reintroduce any of the following unless new authoritative evidence is added and separately audited:

- withheld iWildCam numerical or action rows;
- claims that the POEM or AETTA ports are official implementations or establish superiority;
- confirmatory significance language for the retrospective CIFAR-10-C contrasts (the six-contrast Holm result is `p = 0.09375`);
- confidence-robust ImageNet-C SAR routing claims when only point estimates are supported;
- obsolete mixed-dataset aggregates;
- claims that a one-bit certificate closes the theory or completes all foundations;
- universal no-harm, default-guard, or zero-adaptation proof claims not supported by the maintained theorems;
- any numbers copied only from a superseded PDF or archival TeX source.

Current limitations must remain visible: CIFAR-10-C pooled point estimates favor KGA over the fixed policies for Tent/EATA, SAR favors always-adapt, no current natural dataset establishes confidence-supported selective routing against both fixed policies, CCT-20 is primarily a safe-utility result, So2Sat stopped before target evaluation, and the current ImageNet-C SAR interpretation is point-only.

## Build outputs

The implementation will produce:

- `docs/research/kbound/kbound_full.tex`
- `docs/research/kbound/kbound_full_supplement.tex` if full-only material is needed
- `docs/research/kbound/kbound_full.pdf`

The historical `kbound.tex` remains marked superseded. The existing `kbound_short_final_draft.pdf` and `kbound_tmlr.pdf` remain the compact submission builds.

## Verification

Before the full PDF is called complete:

1. Compile from a clean auxiliary-file state using the repository's supported LaTeX workflow.
2. Run the manuscript claim validator and canonical release validator that apply to K-Bound.
3. Check every added numerical claim against canonical JSON or the named authoritative experiment artifact.
4. Search the generated text for excluded or superseded claims and for unresolved references or citations.
5. Inspect the LaTeX log for errors, undefined references/citations, and serious overfull boxes.
6. Render every PDF page to images, inspect a contact sheet, and inspect representative pages at full resolution for clipping, broken equations, unreadable figures, and layout regressions.
7. Confirm the PDF identifies itself as the full companion manuscript and does not claim to be the compact or anonymous TMLR submission.

## Acceptance criteria

The work is complete when:

- `kbound_full.pdf` compiles successfully from maintained sources;
- the beta-surrogate failure is explicit in the main limitations section with the audited numbers and correct interpretation;
- every restored long-form result has an authoritative source and appropriately limited claim language;
- no known obsolete result or withdrawn theoretical claim appears as current evidence;
- automated checks pass, or any pre-existing validator limitation is documented precisely;
- visual inspection finds no material PDF defects; and
- archival and compact manuscripts remain intact.
