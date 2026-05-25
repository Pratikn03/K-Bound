# PHASE 3 PRE-INTEGRATION EVIDENCE AUDIT

This document records the verification of final Phase-2 evidence before manuscript integration and thesis update.

## Repository State Verification
- **Branch:** `exp/elara-phase2-mechanism-and-replication`
- **HEAD Commit:** `850804d73175f497a476d0142981426709e99470`
- **Git Status:** Clean tree
- **Phase 2 Audit Commit:** `850804d` (Verified)

## 1. Family-A Authoritative Evidence Verification
Source artifact: `experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv`

| Cell | Benchmark / Protocol | Delta AUC vs static_attention | Holm K=5 p | 95% CI | Effect Band | Sign Consistency | Status |
|---|---|---:|---:|---|---|---:|---|
| A-POWERED-1 | MVTec 3D-AD supervised-paired | +0.1082 | 3.35e-04 | [+0.052, +0.166] | large | 30/30 | VERIFIED |
| A-POWERED-2 | MVTec 3D-AD held-out | +0.0519 | 4.06e-05 | [+0.029, +0.075] | large | 30/30 | VERIFIED |
| A-POWERED-3 | MVTec LOCO-AD supervised-paired | +0.1038 | 4.06e-05 | [+0.058, +0.150] | large | 30/30 | VERIFIED |
| A-POWERED-4 | VisA RGB+edge | +0.0297 | 1.53e-03 | [+0.012, +0.049] | moderate | 30/30 | VERIFIED |
| A-POWERED-5 | UNSW-NB15 | +0.0095 | <1e-15 | [+0.008, +0.011] | small | 30/30 | VERIFIED |

### Mandatory Qualifications
- A-POWERED-3 pairing_strength = `derived_view_proxy` (not independent modalities).
- A-POWERED-4 pairing_strength = `derived_view_proxy` (not independent modalities).
- A-POWERED-5 effect band is **small** (retained).
- A-POWERED-2 absolute held-out performance is near chance (RGA+ ens AUC = 0.5216 vs static ens AUC = 0.4698).

## 2. Family-B Authoritative Evidence Verification
Source artifact: `experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv`

- **B1 (zero_attack k=4, mean gate tau=0.66):**
  - Delta: `+0.0507` (95% CI `[+0.0364, +0.0650]`)
  - Holm K=2 p-value: `4.31e-12`
  - Status: `VERIFIED_REPRODUCED`
- **B2 (max_attack k=4, mean gate tau=0.66):**
  - Phase-1 delta: `+0.0319`
  - Phase-2 delta (30-seed ensemble-pooled): `+0.0939` (95% CI `[+0.0741, +0.1149]`)
  - Holm K=2 p-value: `< 1e-15`
  - Status: `COMPARABLE_BUT_ESTIMATOR_CHANGED_POSITIVE_RESULT` (Reported side-by-side)

### RGA-v2 Failure Surface Inference
Source artifact: `experiments/phase2/mechanism/rga_v2_failure_surface_inference.csv`
- **G0 (mean gate):** FFR = `0.0000`, budget = `0.0100` (Budget passed)
- **G1 (min gate):** FFR = `1.0000` (Budget failed)
- **G2 (combined):** FFR = `1.0000` (Budget failed)
- **G3 (top-q domain):** FFR = `1.0000` (Budget failed)
- **Promotion decision:** `RGA_V2_EXECUTED_NOT_PROMOTED`. G0 remains base/reference gate.

### Domain Composition Shift Audit (B-MECH-3S)
Source artifact: `experiments/phase2/mechanism/domain_composition_shift_metrics.csv`
- Global KS fire rate: `1.0000`
- Domain-aware fire rate: `1.0000`
- Status: `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED` (Both fired at 100% rate, general category/cohort theorem remains unresolved/deferred).

### KS Window Size Power Sweep (B-MECH-4)
Source artifact: `experiments/phase2/mechanism/ks_window_size_power.csv`
- Evaluated window sizes: `{32, 64, 128, 256, 512}`
- Detection power trend:
  - 32: `~24.6%` (0.245)
  - 512: `~62.4%` (0.624)
- False activation rate: `≤ 0.06%` (max observed 0.000625)
- Status: `TRADEOFF_IMPROVED` (Gains bounded to evaluated grid and degradation types).

### Certificates and Risk Dominance (B-CERT-1)
Source artifact: `experiments/phase2/certification/switching_certificates_v2.csv`
- **max_attack k=4:** `CERTIFIED` retrospectively (paired-bootstrap LCB = `+0.0085`, `pi_star = 0.000`)
- **zero_attack k=4:** `NOT_CERTIFIED` (paired-bootstrap LCB = `-0.0050`, `delta_1 = -0.0039`)
- Boundary qualification: Retrospective stress protocol evaluations only. Not production safety certificates or real-world deployment guarantees.

## 3. Family-D Excluded Held-Out Study Verification
Source artifact: `docs/research/phase2/FAMILY_D_V3_FINAL_VALIDITY_AUDIT.md`

- **Frozen clean false-fire budget:** `≤ 0.010`
- **Observed clean validation false-fire rate:** `1.0` (100% rate)
- **Status:** `FAMILY_D_V3_INVALID_CLEAN_FALSE_FIRE_POLICY_FAILURE` (Excluded from primary evidence)
- **Observed excluded outputs:**
  - D-EYE-1 (depth collapse): delta AUC = `-0.0072`, 95% bootstrap CI = `[-0.0222, +0.0074]`, corrected DeLong p = `0.3323`
  - D-EYE-2 (RGB collapse): delta AUC = `-0.0053`, 95% bootstrap CI = `[-0.0158, +0.0048]`, corrected DeLong p = `0.3127`
- **DeLong Double-Division Bug:**
  - Variance divided twice in `_delong_auc_variance` and `_delong_paired_test`.
  - Underestimated variance by `~250x`, falsely showing `p = 0.0000`.
  - Corrected p-values above must replace buggy values everywhere.
  - No manuscript text may imply statistical significance.

## Audit Sign-off
Verified by Phase 3 Integration Runner on 2026-05-25.
All raw values successfully traced.
No missing or contradictory values identified. Pre-integration checklist is complete.
