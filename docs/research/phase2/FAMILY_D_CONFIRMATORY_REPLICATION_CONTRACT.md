# Family D — Confirmatory Replication Contract

**Status:** **FROZEN — NOT YET EXECUTED.** All artefacts in this folder define what a true confirmatory replication of the RGA+ mechanism would look like. Execution is **out of scope** for the current session per the explicit stop boundary in [PHASE_2_RESEARCH_CONTRACT.md](./PHASE_2_RESEARCH_CONTRACT.md).

> This contract is freezing-time text. Any post-hoc edit to this file
> after the first confirmatory dataset has been touched invalidates
> the confirmatory status of that dataset.

## 1. Why Family D exists

Families A (audited public benchmark), B (mechanism replication), and C (exploratory) are all **post-hoc** with respect to the dataset choice and the comparator suite. The headline RGA+ ROC-AUC deltas reported in Family A and the mechanism deltas in Family B are vulnerable to two epistemic threats:

1. **Researcher degrees of freedom** — the gate threshold, the head-selection rule, the comparator set, the protocol selection, and the splitting policy were all chosen with at least partial knowledge of how the mechanism behaves on these data.
2. **Benchmark-specific quirks** — fixed-seed p-values on one benchmark do not prove robust method superiority (this is the published Phase-1 negative finding; see [PHASE_2_STATISTICAL_POLICY.md](./PHASE_2_STATISTICAL_POLICY.md)).

Family D is the **only** part of the Phase 2 program that can produce a confirmatory inferential statement about RGA+. It does so by pre-registering every degree of freedom **before** touching test data on a held-out benchmark.

## 2. Pre-registration boundary

The pre-registration freezes the following before any Family-D test split is read:

- Datasets: see [FAMILY_D_DATASET_INVENTORY.md](./FAMILY_D_DATASET_INVENTORY.md).
- Hypotheses with directionality and effect-size thresholds: see [FAMILY_D_HYPOTHESES.csv](./FAMILY_D_HYPOTHESES.csv).
- Splits and seeds: see [FAMILY_D_PARTITION_MANIFEST.json](./FAMILY_D_PARTITION_MANIFEST.json).
- Selection and statistical rules: see [FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md](./FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md).
- Exact execution commands: see [FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md](./FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md).

The git commit that locks this contract MUST predate the commit that produces any Family-D test-split artefact.

## 3. Locked inference rule (matches Phase 2 statistical policy)

- 30 seeds per cell.
- Validation-frozen RGA+ head: `argmax_val{rga_meta_router, rga_boosted_fusion}`, tie-break `rga_boosted_fusion`.
- Validation-frozen primary comparator: chosen on **validation** AUC only, then frozen before any test read.
- Inference on the seed-averaged ensemble: DeLong paired test (Sun & Xu 2014 fast-DeLong) + paired sample bootstrap (10 000 iterations, seed 0) on the seed-averaged ensemble prediction vectors.
- Holm–Bonferroni within the Family-D family of hypotheses (K = 5; see HYPOTHESES.csv).
- Practical-effect bands per Phase 2 statistical policy §5.

## 4. Pass / partial / fail decision rules

A Family-D hypothesis is **CONFIRMED** iff all three hold:

1. Holm-adjusted DeLong p ≤ 0.05.
2. Paired-bootstrap 95% CI strictly excludes 0 in the hypothesized direction.
3. The point-estimate delta sits in the "moderate" or "large" band (|Δ| ≥ 0.005) and matches the pre-registered direction.

If (1) is met but (2) or (3) are not, the hypothesis is **DIRECTIONALLY SUPPORTED**.
If (1) is not met, the hypothesis is **NOT CONFIRMED**.
If the validation-frozen selection ever falls back on test metrics, the cell is **INVALID** and does not contribute to the family.

## 5. What confirmation of Family D would and would not unlock

**Would unlock** (only after a CONFIRMED outcome on at least 1 Family-D hypothesis):
- The phrase "**confirmatory** evidence on a held-out benchmark" in the manuscript abstract.
- Removal of the "audited reanalysis" qualifier from the corresponding Family-A claim.

**Would not unlock — even if every Family-D hypothesis confirms:**
- ELARA is universal.
- ELARA is SOTA.
- ELARA is production-ready or deployment-ready.
- ELARA is validated for clinical deployment.
- Public benchmark results prove broad cross-domain superiority.
- Real3D supports generalization.

These forbidden claims are preserved verbatim from the Phase 2 research contract.

## 6. Out-of-scope deviations recorded up front

The locked compute / scope deviation is identical to Phase 2 overall:

- No GPU re-runs of M3DM / AST / BTF / EasyNet under matched seeds.
- No new localization (pixel-AUROC) head.
- The Family-D pilot **as designed** is CPU-feasible within ≈ 4 h on a workstation; the actual budget is deferred to a Phase-2-followup compute window.

## 7. Versioning rule

Any future change to this contract MUST be a separately committed `FAMILY_D_CONTRACT_v{N+1}.md` file, with the v1 file left intact and the diff documented in the commit message. Family-D results may only ever be reported under the version of the contract that was frozen **before** the test reads.
