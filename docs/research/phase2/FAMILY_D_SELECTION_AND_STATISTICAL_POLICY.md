# Family D — Selection and Statistical Policy (frozen pre-execution)

**Status:** FROZEN — must match this text at execution time, byte for byte.

This policy is a strict subset of [PHASE_2_STATISTICAL_POLICY.md](./PHASE_2_STATISTICAL_POLICY.md). Where the two disagree, this Family-D file wins, because it is the explicit pre-registration.

## 1. Validation-only selection

For every Family-D cell:

1. The RGA+ head is selected as `argmax over the validation split of {rga_meta_router_val_auc, rga_boosted_fusion_val_auc}` with tie-break `rga_boosted_fusion`. The selection MUST emit a `selection_log` row with `selection_used_test_metrics=False`.
2. The primary comparator is `static_attention` (D-H1, D-H3, D-H4) or `craf_attention` (D-H2, D-H5), as set in [FAMILY_D_HYPOTHESES.csv](./FAMILY_D_HYPOTHESES.csv). No test-set comparator picking — the comparator is set by the hypothesis ID, **not** chosen.

Any deviation from these two rules invalidates the cell.

## 2. Statistical test on the seed-ensemble predictor

- DeLong's paired ROC test (Sun & Xu 2014 fast-DeLong) on the seed-averaged ensemble prediction vectors over 30 seeds.
- Paired bootstrap over test samples: 10 000 iterations, fixed `seed=0`, on the seed-averaged ensemble vectors. 95% CI reported.
- All p-values and CIs MUST be emitted by [src/elara/evaluation/ensemble_inference.py](../../../src/elara/evaluation/ensemble_inference.py:audited_analysis) — the exact same function used for Family A.

## 3. Family-wise correction

Holm–Bonferroni within `K = 5` (D-H1..D-H5). If a hypothesis is not executable in the compute window, it is **dropped before the test read** and `K` is recomputed by removing it. Post-hoc dropping is forbidden.

## 4. Inferential scope

Each CONFIRMED hypothesis entitles the manuscript to:

> "On the seed-ensemble predictor for {dataset} {protocol}, RGA+ separates from {comparator} at Holm-adjusted α = 0.05 (DeLong, K=5), with the paired-bootstrap 95% CI excluding zero and a {band} practical effect."

It does **not** entitle the manuscript to:

- "RGA+ beats every baseline."
- "RGA+ is universally better than {comparator}."
- "Family D confirms ELARA is SOTA."
- "Family D confirms ELARA is production-ready / deployment-ready / validated for clinical deployment."

## 5. Pre-registration integrity contract

- This file is appended to the freeze commit before Family-D test reads.
- Hash of this file at execution time MUST equal the hash recorded in the freeze commit. A mismatch invalidates the Family-D run.
- If any text in this file is materially edited after the freeze commit, the family becomes `INVALID` and the freeze must be re-done as `FAMILY_D_CONTRACT_v2.md`.
