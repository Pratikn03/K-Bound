# Phase 2.1 — Family-A Policy Reconciliation

**Status:** REPAIR. Identifies the policy drift in the original Phase-2 Family-A analysis and specifies the corrected design.

## 1. Drift summary

The locked Phase-2 statistical policy (`PHASE_2_STATISTICAL_POLICY.md`) and the experiment registry both define Family A as a **K = 5 benchmark-cell** audited-reproduction family:

- Each cell has **one validation-frozen primary comparator** (validation-only selection, frozen before any test read).
- Holm–Bonferroni correction is applied **across the 5 cells**, with `multiplicity_family = A-POWERED-K5`.

The actual A-POWERED-1 analysis output produced a **K = 10 within-cell all-comparator** Holm correction instead. This is a policy violation in two directions:

1. It overcorrects within one cell against 10 comparators rather than applying the family-wide K = 5 correction across cells.
2. It does not select **one** primary comparator on validation only — every comparator was compared in parallel.

## 2. Two analyses, two labels

Both analyses are preserved. They are not the same analysis and must not be reported as the same thing.

| Analytic surface | Label | What it is | What it supports |
|---|---|---|---|
| K = 10 within-cell all-comparator Holm | `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` (existing, frozen) | exploratory pilot audit across all named comparators on one cell | "On the pilot cell, RGA+ separates from 5 of 10 named comparators under the all-comparator pilot audit." |
| K = 5 cell-level Family-A primary | **TO BE COMPUTED** once all 5 cells exist | one validation-frozen primary comparator per cell, Holm across 5 cells | the locked Phase-2 Family-A statement |

## 3. Outputs reclassified

| Existing output | New label |
|---|---|
| [experiments/phase2/statistics/family_a_powered_ensemble_inference.csv](../../../experiments/phase2/statistics/family_a_powered_ensemble_inference.csv) | `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` |
| [experiments/phase2/statistics/family_a_powered_holm_results.csv](../../../experiments/phase2/statistics/family_a_powered_holm_results.csv) | `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` |
| [docs/research/phase2/FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md](./FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md) | superseded by `FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT_v2.md` (this report); v1 left in place as historical drift evidence |

The v1 report file is retained verbatim for audit. The v2 report adds the correct labels and adds the required recompute notice.

## 4. Required recompute for A-POWERED-1 under primary one-comparator rule

When future compute opens:

1. Pick **one** primary comparator on the validation split per A-POWERED-1 seed (e.g. `argmax over validation AUC across the comparator set`, or — preferred — a single fixed `static_attention` baseline as set by the registry's "audited Δ vs val-frozen primary comparator" wording).
2. Record the per-seed selection in [experiments/phase2/statistics/family_a_selection_log.csv](../../../experiments/phase2/statistics/family_a_selection_log.csv) with `selection_used_test_metrics=False`.
3. Compute the seed-ensemble DeLong p-value + paired bootstrap CI against that one comparator only.
4. Apply Holm–Bonferroni **across the 5 cells** with `K = 5`, not within one cell with `K = 10`.

A reasonable forward choice — locked by `PHASE_2_RESEARCH_CONTRACT_v2.md` §4 below — is `static_attention` as the universal primary comparator for every Family-A cell. This makes the K = 5 family directly interpretable and aligns with the existing Phase-1 framing of "RGA vs static."

## 5. Family-wide K = 5 Holm cannot be final until all five cells exist

The Holm-adjusted Family-A p-values across the 5 cells **cannot** be reported until all five cells have completed runs. Reporting only A-POWERED-1's primary-comparator p-value with a K = 5 Holm correction would understate the correction (because the comparison set is still partial).

Stop boundary: do not report any Family-A K = 5 p-value until all five primary cells have completed their 30-seed runs and produced validated prediction archives.

## 6. Versioning of dependent contract files

To avoid mutating frozen artefacts, the corrected analysis policy lives in new sibling files:

- [PHASE_2_RESEARCH_CONTRACT_v2.md](./PHASE_2_RESEARCH_CONTRACT_v2.md)
- [PHASE_2_STATISTICAL_POLICY_v2.md](./PHASE_2_STATISTICAL_POLICY_v2.md)
- [FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT_v2.md](./FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT_v2.md)
- [PHASE_2_INTERIM_REPORT_v2.md](./PHASE_2_INTERIM_REPORT_v2.md)

The v1 files are left byte-for-byte unchanged. Any conflict between v1 and v2 is resolved in favour of v2; v1 stays in place as the historical record of the drift that v2 repairs.
