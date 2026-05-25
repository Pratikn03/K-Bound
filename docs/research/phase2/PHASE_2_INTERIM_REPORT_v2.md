# Phase 2 Interim Report — v2

**Status:** repaired interim report following Phase 2.1 drift detection. **v1 is preserved unchanged** at [PHASE_2_INTERIM_REPORT.md](./PHASE_2_INTERIM_REPORT.md). v2 supersedes v1 for all citations.

## 1. What is actually executed and valid after Phase 2.1

| Item | State |
|---|---|
| A-POWERED-1 prediction archive (MVTec 3D-AD PatchCore supervised-paired, 30 seeds) | valid, unchanged |
| A-POWERED-1 secondary all-comparator pilot audit (K = 10) | valid, **relabelled `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT`** |
| A-POWERED-1 primary Family-A surface (RGA+ vs `static_attention`, K = 5 family) | **not yet computed** |
| A-POWERED-2..5 prediction archives | `pending_compute` |
| B-MECH-1..4, B-CERT-1 | `pending_compute` (scaffolds only) |
| Family D v1 | `INVALID_FOR_EXECUTION` — see [FAMILY_D_V1_INVALIDATION_NOTICE.md](./FAMILY_D_V1_INVALIDATION_NOTICE.md) |
| Family D v2 | `V2_DESIGN_PENDING` — see [FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md) |

## 2. Headline statement permitted after Phase 2.1

Exactly one inferential statement is permitted from the executed evidence:

> "On the seed-ensemble predictor for one MVTec 3D-AD supervised-paired pilot cell, RGA+ separates from five of ten named comparators under the `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT`."

Allowed companion descriptive statements:

- "RGA+ seed-ensemble ROC-AUC = 0.7420 on n_test = 278 (217 pos, 61 neg) over 30 seeds."
- "The seed-ensemble pools 19 deterministic boost predictions and 11 router variants; the boost head is deterministic and only the router carries SGD-trained seed-dependent variation."
- "Validation-frozen head distribution: 19 boost / 11 router. Selection rule recorded with `selection_used_test_metrics=False`."

## 3. What is NOT permitted after Phase 2.1

- "Family A completed."
- "Family A confirmed."
- "Broad generalization."
- "RGA+ beats every baseline."
- "Family D confirmed."
- "ELARA is universal / SOTA / production-ready / clinically validated."
- "Public benchmark results prove broad cross-domain superiority."
- "Real3D supports generalization."
- "Fixed-seed p-values prove robust method superiority."
- Any Family-A K = 5 Holm-adjusted p-value from fewer than five completed cells.
- Any Family-D outcome.

## 4. Layer-2 work cleared for execution under v2

The following may be initiated under v2 (without touching Family D):

- A-POWERED-2..5: run the locked registry cells per [PHASE_2_RESEARCH_CONTRACT_v2.md](./PHASE_2_RESEARCH_CONTRACT_v2.md) §4. Use the same pilot driver path; record validation-frozen head selection per seed; emit prediction archives with the existing 28-column schema.
- B-MECH-1..4 (Family-B mechanism replication): reuse archived predictions where available; otherwise run additional seeds.
- B-CERT-1 (risk-dominance + switching certificate): consume archived predictions only.
- RGA-v2 gate sweep under [configs/phase2/rga_v2_gate_contract.yaml](../../../configs/phase2/rga_v2_gate_contract.yaml).

Family-D execution remains forbidden until a complete v2 contract exists and independent review unfreezes it for execution.

## 5. Phase-2.1 commit boundary

This Phase-2.1 task is contract / registry / test repair only. No new compute. No paper / thesis edits. No Family-D action. Final decision is recorded in [PHASE_2_1_HOSTILE_REVIEW_REPORT.md](./PHASE_2_1_HOSTILE_REVIEW_REPORT.md).
