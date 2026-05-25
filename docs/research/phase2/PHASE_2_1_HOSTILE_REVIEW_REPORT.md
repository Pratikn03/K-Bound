# Phase 2.1 — Hostile Review Report

Posture: senior empirical-ML methods reviewer + reproducibility engineer auditing the Phase 2.1 contract-repair output.

## C1 — "Did the repair preserve the only valid evidence?"

**Reviewer:** *"You touched four contract documents and added eight tests. Did you accidentally invalidate the one analysis that was actually executed?"*

**Response:** No. The A-POWERED-1 prediction archive, the per-seed metrics CSV, the selection log, the ensemble-inference CSV, and the Holm-results CSV are all byte-for-byte unchanged. The K = 10 output is preserved with the new label `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` and retains its statistical validity in that narrower scope. See [PHASE_2_1_PILOT_PRESERVATION_REPORT.md](./PHASE_2_1_PILOT_PRESERVATION_REPORT.md).

## C2 — "K = 10 vs K = 5 — which one is the correct Holm?"

**Reviewer:** *"You now declare two surfaces, primary cell-level (K = 5) and secondary all-comparator (K = 10). Is that just a relabel?"*

**Response:** Not just a relabel. The two surfaces answer different questions:

- `PRIMARY_FAMILY_A_CELL_LEVEL`: "RGA+ vs `static_attention`, corrected across 5 benchmark cells." This is the locked Family-A inference. **Cannot be reported until all 5 cells exist.**
- `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT`: "RGA+ vs each of 10 comparators on this one cell, corrected within-cell." This is the existing pilot output, narrower in family scope.

Both surfaces are valid. They are not interchangeable. The v1 report claimed the secondary surface as the primary Family-A statement; that claim is withdrawn in v2.

## C3 — "Why is `static_attention` the locked primary comparator?"

**Reviewer:** *"Why not `random_forest` or `late_fusion_ensemble`, which were the harder baselines on A-POWERED-1?"*

**Response:** Two reasons.

1. **Frame-alignment with Phase 1.** Phase-1 mechanism endpoints (B1 +0.0506, B2 +0.0319) are reported as "RGA vs static." The Family-A primary comparator should align with that framing so that Phase-2 audited reproduction speaks directly to the Phase-1 result.
2. **Avoiding K = 5 family inflation by selection.** If the primary comparator were chosen per cell on validation AUC, that selection is itself a multiplicity (one more degree of freedom). Locking `static_attention` for all 5 cells removes that degree of freedom and keeps K = 5 honest.

The "harder baselines" (`random_forest`, `late_fusion_ensemble`, the TTA adapters) remain in the `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` and are reported under that label.

## C4 — "Family D v1 was frozen with placeholders. Is the invalidation enough?"

**Reviewer:** *"You wrote an invalidation notice but left v1 files in place. Doesn't the file system still 'contain' a freeze contract that pretends to be valid?"*

**Response:** The invalidation notice + the contract-integrity test
[tests/test_family_d_v1_never_executable.py](../../../tests/test_family_d_v1_never_executable.py) together ensure that:

- the v1 NOT-RUN execution-commands file remains marked NOT RUN;
- no `src/scripts/*.py` references the v1 NOT_RUN command set as active;
- the invalidation notice itself contains the literal `INVALID_FOR_EXECUTION` string and the five specific grounds (placeholder mutation, MPDD modality unverified, VisA prior inspection, Eyecandies test-label leak, wrong claim boundary).

A future agent who tries to "complete" v1 by editing the manifest's placeholders will fail [tests/test_family_d_v2_no_placeholders_before_freeze.py](../../../tests/test_family_d_v2_no_placeholders_before_freeze.py) on any v2-named manifest, and the invalidation notice's "v1 must not be edited" rule blocks the alternative path of editing v1 in place.

## C5 — "Is the eligibility review actually closed?"

**Reviewer:** *"You wrote `FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md` with checkboxes — but every checkbox is open. Are we any further along?"*

**Response:** Yes, in two specific ways:

1. **VisA is conclusively removed** from any future Family-D consideration because it is registry-locked into Family A. That conclusion does not need further review.
2. **The four valid Eyecandies protocol options are reduced to two**: canonical one-class or validation-only synthetic-corruption. The "supervised-paired" path is closed because it requires reading test labels for validation tuning.

The remaining open items (MPDD modality manifest, Eyecandies operator specification, additional untouched candidate) are scoped to a follow-up review, not a freeze. This is the correct outcome for a Phase 2.1 contract-repair task whose stop boundary explicitly forbids new model runs.

## C6 — "Are the manuscript / claim layers still drift-free?"

**Reviewer:** *"Forbidden Phase-2 claims could still leak through unguarded LaTeX prose. Is there an automated guard?"*

**Response:** Yes — [tests/test_phase2_no_forbidden_claims_in_manuscripts.py](../../../tests/test_phase2_no_forbidden_claims_in_manuscripts.py) greps both `PAPER_DRAFT_v1.tex` and `THESIS_CHAPTER_v1.tex` for 11 verbatim forbidden strings, and the test runs as part of the standard `pytest` invocation. The full suite currently reports 431 passed / 7 skipped, so neither manuscript contains a verbatim forbidden phrase right now.

## C7 — "What is the operational decision?"

**Reviewer:** *"Pick one of the four allowed Phase-2.1 decisions."*

**Decision:** `PASS FOR CONTINUED FAMILY-A/B COMPUTE ONLY`.

Justification:

- Registries, claim matrix, statistical policy, and Family-A report are repaired and tested.
- Eight contract-integrity tests pass (including the five new Family-D guards and three new registry-schema/alignment guards).
- Family-D v2 design is correctly `V2_DESIGN_PENDING` and cannot be frozen in this task because the eligibility review has open items.
- Therefore future Family-A (A-POWERED-2..5) and Family-B (B-MECH-1..4, B-CERT-1) compute may proceed under v2.
- Future Family-D execution is forbidden until v2 is fully resolved and independently reviewed.

## C8 — "Anything still unresolved?"

**Reviewer:** *"Be candid about what was deferred."*

Deferred to follow-up tasks (not Phase 2.1):

1. Closing the Family-D v2 eligibility review (MPDD modality, Eyecandies operator, additional candidate search).
2. Executing A-POWERED-2..5 + all Family-B cells under the v2 policy.
3. Re-running the A-POWERED-1 primary surface against `static_attention` only and labelling it `PRIMARY_FAMILY_A_CELL_LEVEL` (a recompute, not a retrain — the existing prediction archive is sufficient).
4. Adding a `tests/test_phase2_manuscript_audit_label.py` that asserts every Phase-2 number cited in the LaTeX sources is also present in the corresponding `_v2` CSV. (Currently the suite checks the forbidden-claim direction only; the cited-number direction is a follow-up.)
