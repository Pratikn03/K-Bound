# Family-D v2 — Dataset and Protocol Decision

**Phase:** 2.2B.2 / Step 10
**Status:** **`V2_FREEZE_BLOCKED_PENDING_USER_DECISIONS`**

This document is the "Required first file" of Step 10 in the Phase 2.2B.2 spec. It honestly enumerates which v2 candidates clear the eligibility bar locked in [FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md) and which do not.

## A. Candidates explicitly removed

| Candidate | Status | Reason |
|---|---|---|
| VisA | `INELIGIBLE_FOR_FAMILY_D` | Registry-locked into Family A (A-POWERED-4) as `derived_view_proxy`; cannot also be Family-D held-out. Already removed from any v2 candidate list. |
| MPDD | `INELIGIBLE_FOR_INDEPENDENT_MULTIMODAL_CONFIRMATION` (current state) | Official multimodal (RGB + depth/3D) modality manifest not verified in the v1 inventory. v2 inclusion gated on official-source verification of paired depth files in the exact release. **Unresolved.** |

## B. Eyecandies — eligible candidate, pending protocol decision

`ELIGIBLE_MULTIMODAL_CANDIDATE_PENDING_PROTOCOL_FIX`

- **Modality:** RGB + normal map + depth + multi-view (per official Eyecandies release).
- **Split structure:** official train and validation are **anomaly-free**; anomalies appear **only** in the test split.
- **Implication:** a naive "supervised-paired" protocol that reads test-set anomaly labels for validation tuning is **not admissible** (it would violate the held-out invariant).

**Two scientifically valid protocols remain. Exactly one must be locked before v2 freeze:**

1. **Canonical one-class multimodal evaluation.**
   - Train on official anomaly-free train.
   - Tune on official anomaly-free validation (normality metrics only — no anomaly labels available on validation by design).
   - Evaluate ROC-AUC on the official test split exactly once, under the frozen v2 contract.

2. **Validation-only synthetic-corruption protocol.**
   - Train on official anomaly-free train.
   - Generate synthetic anomalies on the official anomaly-free **validation** split using a frozen corruption operator (must be specified in full in the v2 contract; must not reference any property of the official test split).
   - Tune RGA+ heads against these synthetic validation anomalies.
   - Evaluate on the official test split exactly once.

   If chosen, the v2 contract MUST specify (with no placeholders):
   - the exact synthetic corruption operator and its parameter set;
   - the random seed governing corruption sampling;
   - a hash of the generated synthetic validation labels;
   - a proof that the operator parameters are computed from training statistics only.

## C. Search for an additional untouched RGB+depth / RGB+normal / RGB+point-cloud candidate

`SEARCH_OPEN` (no new candidate identified in this audit).

Candidates listed in [FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md) §D — Real-IAD, 3D-AnomalyMNIST, DDTGS-3D — remain unverified for:
- official modality manifest;
- license / academic-access status;
- zero prior inspection in this repo.

None of these verification lines is closed.

## D. Confirmation target (cannot be locked without C above)

Even if Eyecandies one-class is chosen, the confirmation **target** depends on whether RGA+ supervised-head selection is admissible without official test-anomaly labels. The cleanest bounded target is:

> "Base RGA reliability-aware fusion vs fixed static_attention reference under naturally paired multimodal Eyecandies one-class evaluation."

The supervised-head variant (router vs boost) requires labelled anomalous validation, which is **not available** in canonical one-class Eyecandies. If protocol 2 (synthetic corruption) is chosen, supervised-head selection becomes admissible **only** under the synthetic operator.

## E. Why v2 cannot be frozen in this Phase 2.2B.2 task

The v2 freeze rules forbid:

- `TBD`
- `TO_BE_FILLED`
- `TO_BE_RECORDED`
- placeholder
- pending hash after freeze

A v2 freeze in this task would require choices that exceed what I can responsibly make without user research input:

1. **Choose Eyecandies protocol** (one-class vs synthetic-corruption). This is a research decision that should belong to the research team.
2. **Specify the synthetic-corruption operator** in full (operator family, parameter values, seed, hash of generated labels) **if** protocol 2 is chosen. Requires user research input.
3. **Download Eyecandies release** to record SHA256 of the archive. Requires authorization, network access, and storage budget approval.
4. **Verify the official modality manifest** for Eyecandies (and optionally MPDD) against the exact release tag.
5. **Identify a second untouched RGB+depth/normal/point-cloud candidate**, or accept Eyecandies as the sole v2 cell.

Until items 1–5 are resolved by the research team, no v2 freeze can land without placeholders.

## F. Decision

> **`V2_FREEZE_BLOCKED_PENDING_USER_DECISIONS`**

The companion file [PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md](./PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md) (Step 11 blocked branch) documents the exact missing requirements and the next allowed task.
