# Family-D v3 — Final Decision Audited

**Phase:** 2.2E / Stage 4  
**Status:** `DECISION_LOCKED`

Following the completion of the Family-D v3 held-out execution, a final validity audit was performed. This document records the final audited decision for Family-D.

---

## 1. Audited Status

Family-D is classified as:

# **`FAMILY_D_V3_INVALID_CLEAN_FALSE_FIRE_POLICY_FAILURE`**

---

## 2. Rationale

The invalidation is based on two independent validity failures:

1. **Protocol Violation (Clean False-Fire Budget):**
   - The frozen protocol set a non-overrideable budget of `0.010` (≤ 1.0%) for the clean validation false-fire rate.
   - The observed clean validation false-fire rate was `1.0` (100%) due to severe domain shift of the Eyecandies dataset.
   - No candidate threshold satisfied the budget constraint, violating the selection protocol. The runner defaulted to $\tau = 0.66$, which also had a clean false-fire rate of `1.0`, exceeding the budget by 100x.
2. **Statistical Calculation Error (DeLong paired test):**
   - The inference script `run_phase2_family_d_v2_inference.py` contained a double-division bug in its DeLong variance calculation, underestimating the variance by $\approx 250$x and falsely reporting p-values of `0.0000`.
   - The corrected p-values (D-EYE-1 $p = 0.3323$, D-EYE-2 $p = 0.3127$) demonstrate that there is no statistically significant difference in performance, which is fully consistent with the bootstrap confidence intervals that include zero.

---

## 3. Claim Boundaries

### Allowed Claim
> "A planned Eyecandies held-out evaluation was executed but excluded from primary evidence because protocol validity requirements were not satisfied."

### Forbidden Claims
- "Family D confirms ELARA."
- "ELARA is universal."
- "ELARA is SOTA."
- "ELARA is deployment-safe."
- "Family A becomes confirmatory."
- "RGA+ beats strongest baselines."
- "Physical-AI safety validation."
- "Raw-sensor corruption robustness."

---

## 4. Phase-3 Clearance

- Since Family-D is excluded from primary evidence, the final Phase-2 evidence consists of the **audited Family-A CV results** and the **bounded Family-B mechanism evidence** (with G0 certification under max_attack k=4).
- Phase 2 is closed. Phase 3 manuscript integration may proceed using only the valid Family-A/B claims. No Family-D redesign or rerun is authorised.

---

*Decision locked: 2026-05-25*
