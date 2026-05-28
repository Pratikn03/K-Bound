# Family-D v3 — Final Decision Audited

**Phase:** 2.2E / Stage 4  
**Status:** `DECISION_LOCKED`

Following the completion of the Family-D v3 held-out execution, a final validity audit was performed. This document records the final audited decision for Family-D.

---

## 1. Audited Status

Family-D is classified as:

# **`FAMILY_D_V3_NOT_CONFIRMED`**

---

## 2. Rationale

The classification is based on the following findings:

1. **Protocol Validity (Clean False-Fire Budget Satisfied):**
   - The frozen protocol set a non-overrideable budget of `0.010` (≤ 1.0%) for the clean validation false-fire rate.
   - While the original uncalibrated run violated this budget (observed FFR = 1.000), applying the calibration fix (`re_fit_ks_reference`) successfully satisfied the budget by selecting a threshold of $\tau = 0.55$, yielding an observed clean false-fire rate of `0.000` (0%) on both validation and test folds. The protocol validity requirements are satisfied.
2. **Statistical Non-Significance (Per-Seed Paired t-test):**
   - The initial inference script contained a double-division bug in its DeLong variance calculation, underestimating variance by $\approx 250\times$.
   - Additionally, running DeLong paired tests on the 30-seed ensemble outputs results in numerically degenerate z-statistics due to near-perfect score correlation. 
   - Performing a per-seed paired $t$-test across the 30 seeds reveals a statistically non-significant positive delta for D-EYE-1 ($p = 0.3632$, mean delta $+0.0019$) and a non-significant negative delta for D-EYE-2 ($p = 0.4468$, mean delta $-0.0011$).
   - Since the 95% bootstrap confidence intervals overlap with zero (D-EYE-1: $[-0.0114, +0.0092]$; D-EYE-2: $[-0.0254, +0.0034]$) and the target minimum practical delta of 0.01 is not met, the transfer attempt is classified as not confirmed.

---

## 3. Claim Boundaries

### Allowed Claim
> "A planned Eyecandies held-out evaluation was executed but not confirmed due to statistically non-significant performance differences."

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

- Since Family-D is valid but not confirmed, the final Phase-2 evidence consists of the **audited Family-A CV results**, the **bounded Family-B mechanism evidence** (with G0 certification under max_attack k=4), and the **not-confirmed Family-D transfer evidence**.
- Phase 2 is closed. Phase 3 manuscript integration may proceed using the valid Family-A/B/D claims under the respective limitations.

---

*Decision locked: 2026-05-27*
