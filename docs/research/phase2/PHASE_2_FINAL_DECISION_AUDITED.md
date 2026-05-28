# Phase 2 — Final Decision Audited

**Phase:** 2.2E / Phase 2 Closure  
**Status:** `PHASE_2_CLOSED_AND_LOCKED`

Following the validity audit of the Family-D v3 execution, this document presents the final audited decision for the entire ELARA Phase 2 empirical study.

---

## 1. Final Phase-2 Decision

The final Phase-2 status is locked as:

# **`PHASE_2_COMPLETE_WITH_FAMILY_D_NOT_CONFIRMED_AND_BOUNDED_FAMILY_A_B_EVIDENCE`**

---

## 2. Evidence Registry Status

| Family | Scope | Final Classification | Citation Claim |
|---|---|---|---|
| **Family A** | Audited CV static-reference | `COMPLETE — VALID` | "Audited static-reference cv evidence under K=5 cross-validation." |
| **Family B** | Bounded mechanism evidence | `COMPLETE — VALID` | "G0 mean-gate certified under max_attack k=4 (LCB +0.0085); B1 positive but not certified (LCB -0.0050)." |
| **Family D** | Eyecandies held-out replication | `COMPLETE — NOT CONFIRMED` | "A planned Eyecandies held-out evaluation was executed but not confirmed due to statistically non-significant performance differences." |

---

## 3. Allowed Claims for Manuscript Integration

1. **Family-A static-reference CV:** We have valid, audited cv evidence showing positive attention fusion benefits across MVTec 3D-AD, ELARA-Bench-LA, UNSW-NB15, MVTec LOCO-AD, VisA, and Real3D-AD.
2. **Family-B mechanism isolation:** We have verified RGA-v2 mechanism results. The mean-gate G0 is certified under maximum attack ($k=4$, LCB $+0.0085$). G1/G2/G3 suffer from severe false-fire rates on clean data due to batch-level minimum pooling and are not promoted. Domain-composition shift results show that domain-aware reference estimation can reduce false-fire rates. KS power sweeps show a clear window-size trade-off, peaking at window size 512.
3. **Family-D held-out replication:** The Eyecandies held-out evaluation satisfied the clean false-fire budget under the calibration fix (observed 0.000 vs budget 0.010) but did not show statistically significant performance improvements, and is classified as not confirmed.

---

## 4. Forbidden Claims (Locked Claim Ceiling)

The manuscript must NOT claim:
- ELARA is universal.
- ELARA is SOTA.
- ELARA is deployment-safe.
- Family A becomes confirmatory.
- RGA+ beats strongest baselines.
- Physical-AI safety validation.
- Raw-sensor corruption robustness.

---

## 5. Phase-3 Authorization

- **Phase 3 Manuscript Integration is hereby AUTHORISED.**
- No further Phase-2 experiments are permitted.
- No Family-D redesign or rerun is authorised for this paper.

---

*Decision locked: 2026-05-27*
