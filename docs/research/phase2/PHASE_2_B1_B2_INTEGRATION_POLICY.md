# Phase 2 — B1 / B2 Integration Policy

**Phase:** 2.2B.2 / Step 2
**Status:** LOCKED. Final wording authority for any future manuscript-update phase.

## 1. B1 final status

> **`VERIFIED_REPRODUCED`**

Phase-2 estimate Δ AUC = **+0.0507** (95% CI [+0.0364, +0.0650]; Holm K=2 p = 4.31 × 10⁻¹²). Phase-1 target = **+0.0506**. Per-seed sign-consistency 29/30.

Permitted wording in any future manuscript-update phase:

> "Under the Phase-2 30-seed ensemble-pooled archived-prediction pipeline, B1 (zero_attack k=4 mean-gate τ=0.66) reproduces the Phase-1 target Δ AUC = +0.0506 with Δ AUC = +0.0507 (95% CI [+0.0364, +0.0650])."

## 2. B2 final status

> **`COMPARABLE_BUT_ESTIMATOR_CHANGED_POSITIVE_RESULT`**

Phase-2 estimate Δ AUC = **+0.0939** (95% CI [+0.0741, +0.1149]; Holm K=2 p < 10⁻¹⁵). Phase-1 target = **+0.0319** (CI [+0.005, +0.062]).

The Phase-1 number and the Phase-2 number target the same underlying endpoint (max_attack, k=4, mean gate, τ=0.66, ELARA-Bench-LA) but use different aggregation estimators:

- Phase-1: per-seed AUC then mean over ≈ 5 seeds.
- Phase-2: 30-seed ensemble-pooled AUC (pool all predictions, compute one AUC).

Permitted wording in any future manuscript-update phase (verbatim):

> "B2 remains positive under the Phase-2 archived-prediction pipeline. The Phase-1 estimate (+0.0319) and Phase-2 30-seed ensemble-pooled estimate (+0.0939) correspond to comparable endpoint definitions but different aggregation estimators; future manuscript integration must report both numbers rather than replacing one with the other."

## 3. Forbidden wording (verbatim, preserved)

The following sentences may NOT appear in any future manuscript update without a separate methods change that re-runs the Phase-1 protocol:

- "B-MECH-1 reproduced ×2"
- "B2 exact replication"
- "B2 may replace the historical number"
- "Phase-1 B2 was wrong / inflated / deflated"
- "The Phase-1 B2 number was superseded"
- "B2 is +0.0939" (without the dual-number explanation)
- "B2 is +0.0319" (without the dual-number explanation)

## 4. Reference documents

- [B2_MAGNITUDE_COMPARABILITY_AUDIT.md](./B2_MAGNITUDE_COMPARABILITY_AUDIT.md) — full audit closing with `COMPARABLE_BUT_ESTIMATOR_CHANGED`.
- [experiments/phase2/mechanism/b2_phase1_vs_phase2_comparability.csv](../../../experiments/phase2/mechanism/b2_phase1_vs_phase2_comparability.csv) — 4-row Phase-1 vs Phase-2 side-by-side artifact.
- [experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv](../../../experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv) — primary Phase-2 inference.

## 5. Test guard

[tests/test_phase2_b2_dual_number_policy.py](../../../tests/test_phase2_b2_dual_number_policy.py) asserts:
- Both Phase-1 and Phase-2 B2 numbers are present in the comparability CSV.
- Forbidden replacement phrases do not appear in any Family-B-region report.
