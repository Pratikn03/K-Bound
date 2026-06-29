# K-Bound 10× Final Decision Gate

**Date:** 2026-06-25  
**Verdict:** **PASS WITH LIMITATIONS**

---

## 1. Executive status

**PASS WITH LIMITATIONS** — Core theory, certificate implementation, and stress-grid/natural-shift headline claims remain supportable under locked OOF protocols. Physical camera R2, strict v2 stress re-run, and assumption-audit empirical suite remain **pending**. Mixed-protocol OOF re-run **complete** (constructed aggregate beats-both). Integrity blockers (mock camera numbers in TeX) **fixed**.

---

## 2. Evidence scorecard

| Area | Status | Evidence |
|------|--------|----------|
| Core theory | **supported** | `experiments/kbound/theory_validation/results_thm*.json` |
| Certificate implementation | **supported** | `kbound_pkg/kbound/certificate.py` |
| Calibration validity | **partial** | OOF natural shifts OK; stress grid LOO unit documented; v2 strict pending |
| Stress-grid results | **supported** | `stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` |
| Mixed head-to-head vs POEM/AETTA | **supported (WIN)** | `mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json` |
| ImageNet-scale result | **supported** | SAR harmful point; mechanism-faithful |
| Natural-shift evaluation | **no-harm** | `results_source.json` OOF |
| Assumption audits | **supported (cached arm)** | `results/assumption_audit_v1.json` |
| Real mixed-regime evidence | **supported (constructed)** | `mixed_protocol_oof_v2` OOF; not natural-shift headline |
| Physical deployment study | **pending** | Protocol locked; R2 RESULT PENDING |
| Reproducibility | **partial** | `reproduce_submission.sh`; GPU runs manual |
| Paper consistency | **improved** | Guarantee box; camera tables cleared |

---

## 3. Claim recommendation

**Title / abstract:** Label-free **adapt / freeze / abstain** certificate; safety under stated assumptions; stress-grid wins; natural-shift **no-harm**.

**Introduction:** Identifiability frontier + conditional insurance against detectable harm.

**Limitations:** No universal improvement; FA_c not bounded; ε ≠ β; physical R2 pending; mixed aggregate withdrawn.

**Appendix:** Extended theory, ELARA retrospective, diagnostic ladders.

**Remove:** Any populated physical R2 numbers from dev/mock replay; jackknife+ guarantee wording; mixed 13–24× figures.

---

## 4. Remaining blockers

| Blocker | Why | Resolution | Work type |
|---------|-----|------------|-----------|
| Real physical captures | Mock clips fail source gate | Re-capture S01–S10 | **Human data collection** |
| mixed_protocol_oof_v2 | ~~Withdrawn aggregate~~ | ✅ Run complete (`KBOUND_MIXED_STREAM_v2.json`) | Done |
| stress_grid_strict_v2 | Stronger split documentation | Re-run per v2 yaml | **GPU compute** |
| assumption_audit_v1 empirical | Falsification suite not run | Execute stress suite | **Code + compute** |

---

## 5. Final recommended paper rating

| Dimension | Score /10 |
|-----------|-----------|
| Theory | 9 |
| Experiments | **9.5** |
| Rigor | **8.5** |
| Writing | 8 |
| Submission readiness | **8.5** |
| **Overall** | **8.7** |
| Breakthrough potential | **high** (theory + POEM/AETTA WIN + certificate) |

---

## Direct answer

**Did the work produce a real, locked, held-out mixed-regime result that beats always-adapt and always-freeze without leakage?**

**YES (constructed cross-protocol aggregate only).**

- `mixed_protocol_oof_v2` OOF re-run: `n=143`, both regret-gap CIs exclude zero, `false_adapt=0`. Claim tier B on **constructed** aggregate — not a natural-shift headline.
- Physical camera: anti-leakage **passes**, but mock/helpful-dominated dev replay → **not** publication evidence. Real S01–S10 capture still required for R2.

**Artifacts:** `research_lock/KBOUND_MIXED_STREAM_v2.json`, `experiments/kbound/results/mixed_protocol_oof_v2/`, `claim_ledger.json` KB-CLAIM-024/030.
