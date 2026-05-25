# ELARA Phase 2 Statistical Policy — v2

**Status:** repaired policy following Phase 2.1 drift detection. **v1 is preserved unchanged** at [PHASE_2_STATISTICAL_POLICY.md](./PHASE_2_STATISTICAL_POLICY.md). Where v1 and v2 disagree, v2 wins.

## 1. Family taxonomy (unchanged)

Identical to v1 §1. Family A is K = 5 cell-level. Family B is K = 2 (B1+B2 mechanism endpoints). Family C is descriptive (no Holm). Family D is frozen-contract only.

## 2. Selection rules — REPAIRED

Repaired from v1 §2:

- RGA+ head selection (router vs boost): **validation-only**, per cell, per seed (unchanged).
- **Primary comparator selection:** the primary comparator for Family A is **locked as `static_attention` across all five Family-A cells.** No per-cell or per-seed validation-based selection of a primary comparator. This makes the K = 5 family directly interpretable as "RGA+ vs static across five public benchmark cells."
- A secondary all-comparator audit may be reported per cell (see §3 below). It is **never** the primary inferential surface and is **never** Holm-corrected as if it were the family.
- Threshold / hyperparameter tuning: **validation-only** (unchanged).
- **Forbidden:** any selection that reads test-fold metrics. Any cell with `selection_used_test_metrics=True` is automatically rejected (unchanged).

## 3. Primary vs secondary surfaces per Family-A cell — REPAIRED

For every Family-A cell:

**Primary surface — `PRIMARY_FAMILY_A_CELL_LEVEL`:**

1. Stack per-seed test prediction vectors per method.
2. Compute seed-averaged ensemble prediction vector for RGA+ and for `static_attention`.
3. Run DeLong paired test on the ensemble vectors (RGA+ ensemble vs `static_attention` ensemble). **One p-value per cell**, not per comparator.
4. Compute paired bootstrap over test samples for the 95% CI on `AUROC(RGA+_ensemble) − AUROC(static_attention_ensemble)`. **10 000 iterations, fixed seed 0.**
5. Apply Holm–Bonferroni **across the K = 5 Family-A cells**, not within one cell.

**Secondary surface — `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT`:**

- Same DeLong + bootstrap path, run against every other named comparator in the locked pool.
- Holm-corrected **within the cell across the comparator set** for reporting transparency.
- Tagged with the literal label `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` in every output CSV and report.
- May not be cited under the K = 5 Family-A multiplicity-family.

**Labelling rule:** every reported p-value or CI is labelled either `PRIMARY_FAMILY_A_CELL_LEVEL` or `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT`. Unlabelled rows are rejected.

## 4. Holm K = 5 cannot be final until all five cells exist — REPAIRED

Family-A K = 5 Holm-adjusted p-values **cannot** be reported until all five primary cells have completed runs. Partial-family reporting is forbidden.

If only A-POWERED-1 is complete (as is the case at the close of Phase 2.1), the corresponding statement is:

> "A-POWERED-1 primary surface (RGA+ vs `static_attention`) is computed but its Holm-adjusted Family-A K = 5 p-value is undefined; the cell is reported under its raw DeLong p-value plus a `PARTIAL_FAMILY` flag."

The full K = 5 Holm correction is applied only when all five cells are present.

## 5. Descriptive seed-stability evidence (unchanged)

Identical to v1 §4.

## 6. Practical-effect-size bands (unchanged)

Identical to v1 §5.

## 7. Forbidden statistical patterns — REPAIRED

v1 §6 plus these additions:

| Forbidden | Why |
|---|---|
| Holm correction within one Family-A cell across the comparator set, reported as if it were the K = 5 family correction | Phase 2.1: family is K = 5 cell-level, not K = 10 comparator-level. |
| Reporting a Family-A K = 5 Holm-adjusted p-value with fewer than five completed cells | Phase 2.1: partial-family correction understates the correction. |
| Silently substituting different benchmark cells for the registry's locked A-POWERED-2..5 cells | Phase 2.1: registry / report contradiction. |

## 8. Raw prediction archive contract (unchanged)

Identical to v1 §7.

## 9. RGA-v2 promotion criteria (unchanged)

Identical to v1 §8.

## 10. Family D rule — REPAIRED

Family D v1 contract is `INVALID_FOR_EXECUTION` (see [FAMILY_D_V1_INVALIDATION_NOTICE.md](./FAMILY_D_V1_INVALIDATION_NOTICE.md)). A v2 design is required and is **not yet locked** — its status is `V2_DESIGN_PENDING` pending the eligibility review and protocol-level resolutions documented in [FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md).

Confirmatory language remains reserved for Family D **and** only after a valid v2 contract is locked **and** independently reviewed **and** executed.

## 11. Provenance

This v2 policy is created in response to a senior empirical-ML methods review that found within-cell K = 10 Holm being used in place of cross-cell K = 5 Holm on Family A; see [PHASE_2_1_FAMILY_A_POLICY_RECONCILIATION.md](./PHASE_2_1_FAMILY_A_POLICY_RECONCILIATION.md).
