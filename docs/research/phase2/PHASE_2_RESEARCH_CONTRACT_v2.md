# ELARA Phase 2 Research Contract — v2

**Status:** repaired contract following Phase 2.1 drift detection. **v1 is preserved unchanged** at [PHASE_2_RESEARCH_CONTRACT.md](./PHASE_2_RESEARCH_CONTRACT.md). Where v1 and v2 disagree, v2 wins.

This v2 contract supersedes v1 only for analyses that start after the v2 commit. The A-POWERED-1 prediction archive itself is unchanged and remains valid as recorded in [PHASE_2_1_PILOT_PRESERVATION_REPORT.md](./PHASE_2_1_PILOT_PRESERVATION_REPORT.md).

---

## 1. Primary Phase-2 questions

Unchanged from v1 §1.

## 2. Non-goals

Unchanged from v1 §2, plus the following additions:

- No within-cell all-comparator Holm correction in place of the K = 5 cell-level Family-A correction.
- No silent replacement of Family-A cells with EfficientAD or Real3D expansions; such expansions must enter the registry as separately numbered exploratory cells under Family C.
- No Family-D v1 execution. v1 is invalidated for execution; see [FAMILY_D_V1_INVALIDATION_NOTICE.md](./FAMILY_D_V1_INVALIDATION_NOTICE.md).

## 3. Mandatory terminology

Unchanged from v1 §3, plus the additions:

- `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` — the within-cell K = 10 surface from A-POWERED-1 may be cited only under this label.
- `PRIMARY_FAMILY_A_CELL_LEVEL` — the K = 5 cell-level Family-A inference. Cannot be reported until all five cells are computed.

## 4. Locked Phase-2 statistical contract (summary, v2)

Full text in [PHASE_2_STATISTICAL_POLICY_v2.md](./PHASE_2_STATISTICAL_POLICY_v2.md). Headline rules:

1. **Validation-only selection** (unchanged).
2. **One primary comparator per Family-A cell.** For Phase-2 v2 the primary comparator across all five Family-A cells is fixed as `static_attention`. This is the locked choice to align with the Phase-1 RGA-vs-static framing and to keep the K = 5 family directly interpretable.
3. **Family-A multiplicity:** Holm–Bonferroni across `K = 5` cells, not within-cell across comparators.
4. **Family-A registry cells (locked, restored from the original registry):**
   - A-POWERED-1: MVTec 3D-AD, PatchCore supervised-paired.
   - A-POWERED-2: MVTec 3D-AD, PatchCore held-out category.
   - A-POWERED-3: MVTec LOCO-AD, PatchCore supervised-paired.
   - A-POWERED-4: VisA, RGB+edge supervised-paired.
   - A-POWERED-5: UNSW-NB15, flow/conn/context.
5. **Seed-ensemble inference path** (unchanged): paired DeLong on seed-averaged ensemble predictions + paired sample bootstrap (10 000 iter, fixed seed) for a 95% AUROC-Δ CI.
6. **Per-cell secondary surface allowed.** Each Family-A cell may also report a within-cell `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` (Holm across the named comparator set on that cell). This surface is **secondary** and never the primary inferential statement.
7. **Family-A K = 5 Holm p-values are not final until all five cells exist.** Partial-family reporting is forbidden.
8. **Practical-effect-size band** is reported alongside every p-value (unchanged).
9. **Family D** is governed by [FAMILY_D_V1_INVALIDATION_NOTICE.md](./FAMILY_D_V1_INVALIDATION_NOTICE.md) and [FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md). No execution under v1; v2 design pending.

## 5. Compute / scope deviation

Identical to v1 §5. Layer-1 work for v2 in this task is **contract / registry / test repair only** — no new compute. Layer-2 work (the actual A-POWERED-2..5 runs, all Family-B runs, RGA-v2 sweeps, KS power sweeps, certificate audits) remains compute-budgeted.

## 6. Commit cadence

1. Lock v2 contracts (this commit set).
2. Future Family-A / Family-B compute under v2.
3. Future Family-D v2 design freeze, after eligibility review closes.
4. No commit may mix manuscript claims with Family-D outcomes.

## 7. Provenance

This v2 contract is created in response to a senior empirical-ML methods review that found three classes of drift in v1:
- Family-A analysis used K = 10 within-cell comparator Holm instead of K = 5 cell-level Holm.
- Family-A cell identities in the post-pilot report diverged from the locked registry.
- Family-D v1 contract was frozen with placeholders that would alter the file when filled at execution time.

All three are addressed by v2 plus the sibling reconciliation documents.
