# Phase 2.1 — Pilot Preservation Report

**Status:** the A-POWERED-1 prediction archive and its K=10 audited inference output remain valid evidence. This report locks the scope of that evidence before further compute is executed.

## 1. What is preserved as valid

- The 30-seed prediction archive under [experiments/phase2/predictions/A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired/](../../../experiments/phase2/predictions/A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired/) remains unchanged.
- The index [experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv](../../../experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv) remains unchanged.
- The audited-inference output [experiments/phase2/statistics/family_a_powered_ensemble_inference.csv](../../../experiments/phase2/statistics/family_a_powered_ensemble_inference.csv) remains unchanged.
- The Holm summary [experiments/phase2/statistics/family_a_powered_holm_results.csv](../../../experiments/phase2/statistics/family_a_powered_holm_results.csv) remains unchanged.

No deletion. No overwrite. No recalculation in this task.

## 2. New analytic label

The existing K=10 all-comparator output is preserved with the label:

> `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT`

It is a secondary, all-comparator pilot audit of one MVTec 3D-AD supervised-paired pilot cell. It is **not** the locked Family-A primary cell-level inference under the original `A-POWERED-K5` multiplicity-family policy.

## 3. What this evidence supports — exact wording

It may support **only** the following statement:

> "On the seed-ensemble predictor for one MVTec 3D-AD supervised-paired pilot cell, RGA+ separates from five of ten named comparators under the all-comparator pilot audit."

The five comparators are: `static_attention`, `craf_attention`, `early_fusion_mlp`, `confidence_weighted_mean`, `eata_score_adapter`.

## 4. What this evidence does NOT support — exact wording

It does **not** support any of:

- "Family A completed."
- "Family A confirmed."
- "broad generalization."
- "RGA+ beats every baseline."
- "Family D confirmed."
- "ELARA is universal."
- "ELARA is SOTA."
- "ELARA is production-ready or deployment-ready."
- "ELARA is validated for clinical deployment."
- "Public benchmark results prove broad cross-domain superiority."
- "Real3D supports generalization."
- "Fixed-seed p-values prove robust method superiority."
- "Existing Family A cells are confirmatory."
- "Existing Family A cells are preregistered."

## 5. Honest reading of the underlying seed-ensemble

The 30 seeds vary the lightweight RGA+ heads on top of a fixed PatchCore score basis. The boost head is deterministic across seeds; the router is the only SGD-trained component with seed-dependent variation. The seed-ensemble RGA+ predictor used for A-POWERED-1 is therefore the average of 19 identical boost predictions plus 11 router variants — **not** 30 independently retrained full pipelines. Future Family-A cells must either (a) record this explicitly in their own preservation reports, or (b) drop the seed-ensemble pooling in favour of per-seed per-model inference.

## 6. Forward boundary

Until the Family-A primary cell-level design is restored under [PHASE_2_1_FAMILY_A_POLICY_RECONCILIATION.md](./PHASE_2_1_FAMILY_A_POLICY_RECONCILIATION.md), no further Family-A compute may be initiated, and the pilot evidence must be cited only under its `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` label.
