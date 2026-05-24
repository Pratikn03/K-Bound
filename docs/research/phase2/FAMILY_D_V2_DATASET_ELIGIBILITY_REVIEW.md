# Family D v2 — Dataset Eligibility Review

**Status:** OPEN review. Must close before any v2 contract is frozen.

This document evaluates each candidate Family-D dataset against the
held-out, independent-multimodal, untouched-by-prior-work criteria
required for confirmatory replication.

## A. MPDD (Metal Parts Defect Detection)

**Decision:** `INELIGIBLE_FOR_INDEPENDENT_MULTIMODAL_CONFIRMATION` unless
an official source verifies paired depth/3D files in the exact
intended release.

**Findings:**
- The v1 Family-D inventory described MPDD as RGB + 3D without
  citing the official modality manifest.
- Public descriptions of MPDD that have been verified to date describe
  RGB-only image data; the "3D" claim in v1 was unverified.
- Without official confirmation of paired multimodal files, MPDD cannot
  serve as the primary multimodal Family-D benchmark.

**Disposition:**
- MPDD may be **retained only as optional RGB-only sensitivity
  evidence**, never as the primary multimodal Family-D benchmark.
- Before any inclusion in v2 (even as RGB-only sensitivity evidence),
  the official MPDD release must be cited with its exact version tag
  and the modality manifest must be quoted verbatim.

## B. VisA

**Decision:** `INELIGIBLE_FOR_FAMILY_D` because VisA already appears in
inspected Phase-2 Family-A analyses as `A-POWERED-4 — VisA, RGB+edge
supervised-paired` (registry-locked).

**Findings:**
- VisA cannot be both an inspected Family-A cell and an untouched
  Family-D candidate.
- The v1 inventory description of VisA as "never touched" is incorrect.

**Disposition:**
- Remove VisA from all v2 Family-D candidates.
- VisA stays in Family A as the registry-locked A-POWERED-4 cell.

## C. Eyecandies

**Decision:** `ELIGIBLE_MULTIMODAL_CANDIDATE_PENDING_PROTOCOL_FIX`.

**Findings:**
- Eyecandies is genuinely multimodal (RGB, normal map, depth,
  multi-view) and has not been inspected in prior ELARA Phase-A or
  Phase-B work.
- The official Eyecandies release provides anomaly-free train and
  validation splits; anomalies appear only in the test split.
- A naive "supervised-paired" protocol that requires labelled
  anomalous validation evidence is therefore impossible to define
  without reading official test labels. Doing so would invalidate the
  held-out nature of the family.

**Two valid protocols are available; one must be chosen before v2 freeze:**

1. **Canonical one-class multimodal evaluation.**
   Train on the official anomaly-free train split.
   Tune on the official anomaly-free validation split (using normality
   metrics only — no anomaly labels exist on validation, by design).
   Evaluate ROC-AUC on the official test split exactly once, under the
   frozen v2 contract.

2. **Validation-only synthetic-corruption protocol.**
   Train on the official anomaly-free train split.
   Generate synthetic anomalies on the official anomaly-free
   **validation** split using a frozen corruption mechanism (defined in
   the v2 contract; cannot reference any property of the official
   test split). Tune RGA+ heads against these synthetic validation
   anomalies. Evaluate on the official test split exactly once.

   If this protocol is chosen, the v2 contract MUST specify:
   - the exact synthetic corruption operator and parameter set;
   - the random seed governing corruption sampling;
   - a hash of the generated synthetic validation labels;
   - the proof that the operator parameters are computed from training
     statistics only, never from test-split statistics.

**Disposition:**
- Eyecandies remains an eligible v2 candidate **only after** one of
  the two protocols above is fully specified.
- If RGA+ requires labelled anomalous validation evidence, the v2
  contract MUST follow protocol (2) and document the synthetic-
  corruption operator in full before freeze.

## D. Search for an additional genuinely untouched RGB+depth / RGB+normal / RGB+point-cloud candidate

**Status:** `SEARCH_OPEN`. No new candidate has been identified inside
the Phase-2.1 stop boundary (which forbids new code execution).

**Required for v2 closure:** at least one additional eligible
multimodal anomaly-detection dataset that satisfies all of:

1. Genuinely paired multimodal (RGB+depth or RGB+normal or
   RGB+point-cloud) with an official modality manifest.
2. Official train / validation / test split structure documented by
   the dataset authors.
3. License / access status compatible with academic reproducibility.
4. **Never previously inspected** in ELARA Phase-A, Phase-B, or
   Phase-C work. Verified by:
   - `grep -ri "<dataset_name>"` returning no functional code paths in
     `src/elara/`;
   - the dataset name not appearing in any Family-A or Family-B
     registry row;
   - the dataset name not appearing in the prediction-archive index.
5. Test split has not been read in any prior repository state.

**Candidates to consider for the search (status to be verified before
inclusion):**

| Candidate | Modalities (claimed) | Pre-Phase-2.1 verification needed |
|---|---|---|
| Real-IAD | RGB + depth (claimed) | (a) official modality manifest; (b) license; (c) confirm zero prior inspection in this repo |
| 3D-AnomalyMNIST / 3D-MAD-Sim | synthetic RGB + 3D | (a) official source; (b) suitability for confirmatory work versus synthetic-only nature |
| DDTGS-3D / DTD-3D | proposed RGB + 3D | (a) confirm public availability; (b) license |

None of these candidates is approved for v2 inclusion until each
verification line above is closed in a follow-up review.

## E. Closure conditions for the eligibility review

All of the following must be true before v2 may be frozen:

- [ ] MPDD modality status resolved with an official-source quote.
- [ ] VisA removed from all v2 candidate lists.
- [ ] Eyecandies protocol choice fixed; if synthetic-corruption,
      operator fully specified.
- [ ] At least one genuinely untouched RGB+depth / RGB+normal /
      RGB+point-cloud candidate verified.
- [ ] License / access status recorded for every retained candidate.
- [ ] No `TO_BE_FILLED`, `TO_BE_RECORDED`, `TBD` placeholders remain in
      the v2 partition manifest.

Until every line above is satisfied, the v2 design is `V2_DESIGN_PENDING`.
