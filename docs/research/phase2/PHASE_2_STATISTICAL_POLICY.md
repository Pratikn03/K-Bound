# ELARA Phase 2 Statistical Policy

**Locked before any new Phase-2 evaluation is run.**

## 1. Family taxonomy

| Family | Status | Multiplicity | Naming |
|---|---|---|---|
| A — public-benchmark powered audited reproduction | Inspected before; Phase 2 reruns are audited reproduction, NOT confirmatory | Holm-Bonferroni within `A-POWERED-K5` (K=5) | "audited reproduction" / "powered audited reproduction" |
| B — mechanism evidence on ELARA-Bench-LA | Mechanism replication of B1/B2 + new mechanism tests (RGA-v2, mixture shift, KS power) | Holm-Bonferroni within `B-MECH-K2` (B1+B2 endpoints) | "mechanism evidence" / "mechanism replication" |
| C — exploratory audits (Real3D, noise-floor, healthcare replay) | Descriptive only; no Holm | none | "exploratory" |
| D — future locked confirmatory replication | FROZEN CONTRACT ONLY in this task; NOT executed | Holm-Bonferroni within `D-CONFIRMATORY-FAMILY` (K=TBD when locked) | "confirmatory" / "pre-registered" — ONLY after independent unfreezing |

## 2. Method-head and comparator selection rules

- RGA+ head selection (router vs boost): **validation-only**, per cell, per seed unless an ensemble-selection protocol is separately frozen.
- Comparator selection: **validation-only**, from the locked candidate pool (`RF, MLP, LFE, Conf-mean, Tent, TTT, EATA, SAR`). Selection rule = seed-mean validation ROC-AUC; deterministic name-tie-break.
- Threshold / hyperparameter tuning (incl. RGA-v2 `tau_min`, top-q): **validation-only**. No test-fold inspection during selection.
- **Forbidden:** any selection that reads test-fold metrics. Any cell with selection_used_test_metrics=true is automatically rejected.

## 3. Primary inferential statistic (per audited cell)

For every Phase-2 cell with archived per-seed test predictions:

1. Stack per-seed test prediction vectors per method.
2. Compute **seed-averaged ensemble prediction vector** per method.
3. Run **DeLong paired test** on the ensemble vectors (RGA+ ensemble vs validation-frozen comparator ensemble). One p-value per cell.
4. Compute **paired bootstrap over test samples** for the 95% CI on `AUROC(RGA+_ensemble) - AUROC(comparator_ensemble)`. **10 000 iterations, fixed seed 0**.
5. Apply Holm-Bonferroni within the cell's family (A-POWERED-K5 or B-MECH-K2 etc).

**Explicit labelling rule:** every reported p-value or CI from this procedure is labelled "ensemble audited analysis" or "ensemble paired sample bootstrap". This is an inferential statement about the seed-ensemble predictor; not a typical-single-trained-model claim.

## 4. Descriptive seed-stability evidence

Alongside the ensemble-paired test, every cell reports:

- per-seed ROC-AUC (mean, SD, min, max);
- per-seed Δ = RGA+ROC-AUC − comparator ROC-AUC;
- sign-consistency count (how many of N seeds have positive Δ);
- per-seed DeLong p-values (descriptive only — never Fisher-combined).

Failure to report seed-mean ± SD = automatic claim rejection.

## 5. Practical-effect-size bands

| Band | Absolute Δ AUROC | Reporting rule |
|---|---|---|
| negligible | < 0.001 | report as "negligible practical effect" alongside any p-value |
| very small | 0.001 ≤ Δ < 0.005 | report as "very small practical effect" |
| small | 0.005 ≤ Δ < 0.01 | report as "small practical effect" |
| moderate | 0.01 ≤ Δ < 0.03 | report as "moderate practical effect" |
| large | Δ ≥ 0.03 | report as "large practical effect" |

These bands are interpretive reporting categories — they are **not** universal clinical / operational thresholds. The bands must always be reported with this disclaimer.

## 6. Forbidden statistical patterns

| Forbidden | Why |
|---|---|
| Fisher combination of seed-level DeLong p-values | Phase 0.6 / Phase 1.D: seeds share the test fold; not i.i.d. |
| Bootstrap over seeds as primary significance test | Phase 0.5 update: estimates training-randomness, not test-population uncertainty |
| Test-fold method / comparator / threshold selection | Rule 4 (Phase 0.6 AR-1, AR-2) |
| Calling Family A "confirmatory" | AR-11 |
| Reporting p-value without practical-effect band | Phase 2 policy 5 |
| Hiding sign-flipped or negative cells | Phase 2 reporting integrity |

## 7. Raw prediction archive contract

Every Phase-2 cell must produce, per seed, a prediction archive containing the columns specified in `experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv` schema (see `src/elara/evaluation/prediction_archive.py`). Cells without a validated archive may NOT be used for inferential claims.

## 8. RGA-v2 promotion criteria (locked)

RGA-v2 may be called an improvement only if all are satisfied (re-stated from the Phase 2 contract):

1. Clean false-fire ≤ locked budget (≤ 1.0% activation on clean fold, OR ≤ base mean-gate clean activation + 0.5pp, whichever is less restrictive).
2. Improves ≥ 2 of {k=1, k=2, k=3} partial-failure regions over G0 (base mean-gate) under the locked primary attack family.
3. Does not worsen k=4 coherent-collapse by more than 0.005 AUROC.
4. Produces positive switching certificate (LCB > 0) on at least one partial-failure regime.
5. Selection validation-only (Rule 4 compliance).
6. Same gate policy across cells — no per-cell test-driven re-tuning.

Failure on any criterion → reported as `MECHANISM_IMPROVEMENT_PARTIAL` or `NOT_IMPROVED`. Test-driven tuning → `INVALID_SELECTION`.

## 9. Family D rule (frozen-contract layer only in this task)

- Family D test partitions must not have been inspected before contract lock.
- Family D execution requires an independent review of the frozen contract before activation.
- This Phase-2 task does NOT execute Family D. The Family D contract artifacts (under `docs/research/phase2/FAMILY_D_*`) are layouts only.
