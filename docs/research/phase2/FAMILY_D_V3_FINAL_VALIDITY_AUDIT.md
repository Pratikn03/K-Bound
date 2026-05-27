# Family-D v3 — Final Validity Audit

This document presents the formal validity audit of the Family-D v3 held-out execution on the Eyecandies RGB+depth protocol.

---

## 1. Verify v3 Freeze Before Test Access

We checked the repository timeline and git history to verify the freeze boundary:

- **v3 Scorer Freeze Commit:** `2485f2e` contains the frozen pipeline configuration (`configs/phase2/family_d_v3_scoring_pipeline.yaml`) and code for the cell runner. This commit was recorded prior to any test evaluation or test metric generation.
- **Final Result Commit:** `1f72e38` contains the generated results, CSV reports, and the closure document.
- **Independence:** All v3 frozen files remained strictly unchanged once test execution began. No test fold outputs or metrics existed before the v3 freeze. No test metrics influenced scorer training, memory bank construction, calibration, or threshold selection.

**Decision:** `FREEZE_TIMELINE_VALID`

---

## 2. Audit Clean False-Fire Protocol Compliance

We audited the validation calibration results and the selection log:

- **A. What exact clean false-fire budget was frozen?** `0.010` (≤ 1.0% gate activation rate on clean validation data).
- **B. Was it non-overrideable?** Yes. The budget was defined as a hard requirement under the selection policy.
- **C. What exact clean false-fire value was obtained?** `1.0` (100% gate activation rate on clean validation data).
- **D. Did any candidate threshold satisfy the frozen budget?** No. All 9 candidate thresholds in $[0.40, 0.80]$ returned a clean validation false-fire rate of `1.0`.
- **E. What exact frozen rule authorised fallback tau=0.66 if none satisfied?** None. The selection policy did not authorize a fallback to $\tau = 0.66$ if the budget was violated. The runner script fell back to $\tau = 0.66$ as a default value, but this fallback violated the non-overrideable budget constraint.
- **F. Does the frozen decision policy permit final held-out execution after a clean false-fire budget failure?** No. The policy specifies that the budget must be verified before test execution. A budget failure means the gate-validity constraint is violated.
- **G. Does such execution count as:** `protocol invalidity` (the run becomes invalid because the clean false-fire verification failed).

**Decision:** `CLEAN_FALSE_FIRE_FAILURE_INVALIDATES_PRIMARY_EVALUATION`

---

## 3. Audit the Final Analysis Unit

We verified the dimensions of the prediction vectors and the sample keys:

- **A. How many unique held-out samples exist per endpoint?** `500` unique samples (across the 10 Eyecandies categories).
- **B. How many final fused prediction scores exist per method per endpoint?** `500` scores.
- **C. Was ROC-AUC computed on one fused prediction per sample, or on modality-level rows?** On exactly one fused prediction per sample (length 500), not on modality-level rows.
- **D. Was DeLong computed on exactly the same paired sample-level vectors as bootstrap?** Yes, both used the same 500-length paired vectors.
- **E. Was the label vector duplicated across modality rows?** No, the labels were resolved per-sample (length 500).
- **F. Does the analysis match the frozen endpoint definition of base RGA versus static_attention fusion?** Yes.

**Decision:** `SAMPLE_LEVEL_FUSION_ANALYSIS_VALID`

---

## 4. Audit DeLong vs. Bootstrap Inconsistency

We audited the DeLong paired test calculation in `run_phase2_family_d_v2_inference.py`:

- **Reported Inconsistency:** The buggy inference script reported DeLong p-values of `0.0000` for both endpoints, despite the bootstrap 95% CIs including zero (`[-0.0222, +0.0074]` and `[-0.0158, +0.0048]`).
- **Mathematical Root Cause:** We identified a double-division variance calculation bug in `_delong_auc_variance` and `_delong_paired_test`. Specifically:
  - `s10` was computed as `float(np.var(pv_pos, ddof=1)) / n_pos`, which already divides by $n_{pos}$.
  - Then `var` was computed as `s10 / n_pos + s01 / n_neg`, dividing by $n_{pos}$ a second time (dividing by $n_{pos}^2$ in total).
  - Similarly, the covariance terms `cov_pos` and `cov_neg` were divided twice.
  - This double-division underestimated the paired variance by a factor of $\approx 250$, inflating the z-scores to values over $-5000$ and producing false $p \approx 0.0000$.
- **Corrected Statistics (Audit Recomputation):**
  - **D-EYE-1 (depth collapse):** $\Delta(\text{AUC}) = -0.007163$, correct $z = -0.9694$, correct $p = 0.3323$.
  - **D-EYE-2 (RGB collapse):** $\Delta(\text{AUC}) = -0.005272$, correct $z = -1.0095$, correct $p = 0.3127$.
  - Both correct p-values are $\gg 0.05$, which is fully consistent with the bootstrap CIs including zero.

**Decision:** `STATISTICS_INCONSISTENT_REQUIRES_INVALID_CLASSIFICATION`

---

## 5. Audit Test Leakage and Protocol Drift

We verified the integrity of the test fold boundaries:

- Scorer training was performed on the train split only.
- Validation calibration and normalisation was performed on the validation split only.
- Test split labels were accessed only during the final metric computation.
- `selection_used_test_metrics = False` was correctly asserted for all archived rows.
- No model, config, operator, or threshold was modified after observing the test fold outcomes.
- No rerun of the evaluation occurred after viewing the outcomes.
- Family-A and Family-B evidence remains completely unchanged.
- Paper and thesis files remained completely unchanged.

**Decision:** `NO_SELECTION_LEAKAGE_OR_DRIFT`

---

## 6. Final Family-D Classification

Based on the protocol compliance audit (Section 2) and the statistical code audit (Section 4), we assign the following final classification to Family-D:

# **`FAMILY_D_V3_INVALID_CLEAN_FALSE_FIRE_POLICY_FAILURE`**

### Allowed Claim
> "A planned Eyecandies held-out evaluation was executed but excluded from primary evidence because protocol validity requirements were not satisfied."

---

## 7. Audit Sign-Off

*Audited on: 2026-05-25*
*Status: CLOSED*
