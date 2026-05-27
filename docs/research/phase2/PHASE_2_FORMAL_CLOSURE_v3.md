# ELARA Phase 2 — Formal Closure Document

**Phase:** 2.2E — Family-D v3 execution and Phase 2 closure
**Status:** `PHASE_2_CLOSED`

---

## 1. Phase 2 Summary

Phase 2 of the ELARA / RGA empirical study produced three evidence families:

| Family | Scope | Final Status |
|---|---|---|
| **A** | Static-reference audited reanalysis (K=5 CV, VisA, Real3D, LOCO, UNSW, ELARA-Bench-LA) | `COMPLETE — AUDITED_STATIC_REFERENCE` |
| **B** | Mechanism isolation evidence (RGA-v2, domain-composition shift, KS-window sweep, certification) | `COMPLETE — BOUNDED_MECHANISM_EVIDENCE` |
| **D** | Held-out Eyecandies execution (validity-audited) | `COMPLETE — INVALID_CLEAN_FALSE_FIRE_POLICY_FAILURE` |

---

## 2. Family-A Evidence (final — unchanged)

Family-A is closed. Full audited-reanalysis results are in the Family-A registry. No further modification is permitted.

**Claim ceiling:** "Audited static-reference evidence under K=5 CV and multiple benchmark families."

---

## 3. Family-B Evidence (final — unchanged)

Family-B closed at commit `2c780cf`. Key findings:

- **B1 (zero_attack k=4):** Phase-2 delta = +0.0507, CI [+0.0364, +0.0650]. **Not certified** at per-sample level (LCB −0.0050).
- **B2 (max_attack k=4):** Phase-2 delta = +0.0939, CI [+0.0741, +0.1149]. **Certified** (LCB +0.0085).
- **B-MECH-2 (RGA-v2):** Batch-minimum gate (G1/G2) produces >90% false-fire on clean data; G0 is the only production-stable gate.
- **B-MECH-3S (Domain-composition shift):** Global and domain-aware references both fire at 100%; no false-fire reduction under this protocol.
- **B-MECH-4 (KS power):** Detection power peaks at window=512; false-fire rate remains <0.06%.

**Claim ceiling:** "Bounded mechanism evidence: G0 mean-gate certified under max_attack k=4; B1 positive but below per-sample certification threshold."

---

## 4. Family-D v3 Evidence (this document)

### 4.1 v3 Pipeline freeze (commit `2485f2e`)

| Artifact | SHA256 |
|---|---|
| `family_d_v3_scoring_pipeline.yaml` | `0a8a844d477d8c0bfaa539a05bd1e89eaeaf707a72f4bc54b9e38ce3e89ce4ac` |
| `family_d_v2_eyecandies_protocol.yaml` | `104d90c6bab38671bb4dba15414a05ccebc890679cd681a5d46e06e7c8be4f15` |
| `FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md` | `e18bc05d12fb717b6b5dac738e41c022480374cefaa5556b5cfb48ce1e667f5d` |
| `FAMILY_D_HYPOTHESES_v2.csv` | `0361a960217f0b32f9a96eef9c261d47af2877a895cbb5e10a0115e8303ad8e2` |
| `FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md` | `65f81a240b41e54fd7dafdbdf045f65d5e2d5c06909f0e05d9a56e286712e60b` |

**v3 scoring pipeline choices (frozen):**
- Backbone: ResNet-50, ImageNet-1K v2 weights, `layer3` features, global avg-pool → 1024-D
- RGB view: `image_0` (canonical front view)
- Depth: gray-to-RGB replication + per-image min-max normalise
- One-class scoring: cosine-distance to full train memory bank, z-score then sigmoid
- Embedding: PCA-16 per (category, modality) from train features
- Labels: train/val = 0 (anomaly-free); test_public from `metadata.yaml`; test_private = no labels available

### 4.2 Test set used

- **Split:** `test_public` only (labels available via `metadata.yaml`)
- **Size:** 500 samples × 10 categories = 500 unique samples × 2 modalities = 1000 scored rows
- **Label distribution:** ~480 anomalous, ~520 normal across categories
- **test_private:** 4000 samples × 2 modalities; labels not available in downloaded archives; excluded

### 4.3 Validation calibration

**Finding:** Clean validation false-fire rate = 1.0 for all tested τ candidates.

> All 9 τ candidates in [0.40, 0.80] exceeded the clean_false_fire_budget = 0.010 on the Eyecandies validation data. No frozen rule authorized fallback to τ = 0.66 after budget failure; default fallback use invalidated protocol compliance.

**Interpretation:** The RGA reliability estimator, as calibrated on the Eyecandies one-class training distribution, fires on the clean normal validation data 100% of the time. This is expected behaviour for an OOD detector calibrated on the ELARA-Bench-LA training distribution: Eyecandies data is a domain shift from the original training data, so the mean reliability score falls below τ = 0.66 for all clean Eyecandies samples.

**Scientific implication:** The RGA gate fires on *all* Eyecandies samples (clean and anomalous alike), which means the gate cannot discriminate between clean and degraded conditions in this domain. This violates the frozen selection policy's clean-false-fire validity requirement and excludes Family-D v3 from primary evidence.

### 4.4 Family-D v3 cell results

| Cell | n_seeds | Δ(AUC) mean | Δ(AUC) ensemble | DeLong p | Bootstrap CI (95%) | Pre-Holm | Holm |
|---|---|---|---|---|---|---|---|
| D-EYE-1 | 30 | -0.0007 | -0.0072 | 0.3323 (corrected) | [-0.0222, +0.0074] | NOT_CONFIRMED | NOT_CONFIRMED |
| D-EYE-2 | 30 | +0.0016 | -0.0053 | 0.3127 (corrected) | [-0.0158, +0.0048] | NOT_CONFIRMED | NOT_CONFIRMED |

**Family decision:** `FAMILY_D_V3_INVALID_CLEAN_FALSE_FIRE_POLICY_FAILURE`

---

## 5. Phase 2 final claim boundaries

Regardless of Family-D v3 outcome, Phase 2 does NOT authorise any of the following claims:

- ELARA is universal
- ELARA is SOTA over any published leaderboard
- ELARA is deployment-safe or clinically validated
- Family-A results become confirmatory
- RGA+ beats strongest baselines
- Physical-AI safety validation
- Raw-sensor corruption robustness

### 5.1 Final locked Phase-2 claim envelope

> "ELARA Phase 2 provides: (a) audited static-reference evidence from Family-A; (b) bounded mechanism evidence from Family-B; and (c) a held-out Eyecandies run that is excluded from primary evidence because protocol validity requirements were not satisfied."

---

## 6. Phase 3 gate

Phase 2 closure gates are satisfied. Phase 3 integration may proceed using only valid Family-A and Family-B evidence. Family-D remains excluded from primary evidence unless a new protocol-valid rerun is explicitly authorized in a separate phase.

---

## 7. Reproduction

```bash
# Verify all frozen artifact SHA256 anchors
shasum -a 256 \
  configs/phase2/family_d_v3_scoring_pipeline.yaml \
  configs/phase2/family_d_v2_eyecandies_protocol.yaml \
  docs/research/phase2/FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md \
  docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv \
  docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md

# Re-run inference (after execution completes)
PYTHONPATH=src python src/scripts/run_phase2_family_d_v2_inference.py \
  --hypotheses docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv \
  --policy docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md
```

---

*Document created: 2026-05-25T20:09:00Z*
*Status: PHASE_2_CLOSED — Family-D v3 results have been finalized and recorded.*
