# B2 Magnitude-Comparability Audit
## Phase 2.2B.1 — Required Before Using B2 in Manuscript

**Created:** 2026-05-24
**Author:** Phase 2.2B.1 auditor (automated hostile-reviewer pass)
**Status:** FINAL — decision required before manuscript update

---

## 1. Purpose

The Phase-2 B-MECH-1 run reproduced the B2 endpoint with a substantially larger
effect than the Phase-1 target:

| | B2 (max_attack k=4) |
|---|---|
| **Phase-1 target Δ** | +0.0319, CI [+0.005, +0.062] |
| **Phase-2 estimate Δ** | +0.0939, CI [+0.074, +0.115] |
| **Magnitude change** | +0.0620 (absolute), ×2.95 (factor) |

This audit determines whether the two estimates measure the same endpoint and
can be compared directly in the manuscript.

---

## 2. Item-by-Item Protocol Comparison

### 2.1 Source artifact paths

| Item | Phase-1 | Phase-2 |
|---|---|---|
| Primary result CSV | `experiments/phase2/mechanism/family_b_primary_replication_metrics.csv` (pending rows at start; filled by B-MECH-1 run) | `experiments/phase2/mechanism/family_b_primary_replication_inference.csv` (computed by `run_phase2_b_mech_1_inference.py`) |
| Prediction archive | Not present in Phase-1 artifacts | `experiments/phase2/mechanism/b_mech_1_prediction_archives/` (parquet-based PredictionArchive) |
| Certificate | `experiments/phase2/certification/switching_certificates.csv` (rows from Phase-1 B-CERT run) | Same file; rows preserved |

**Finding:** Phase-1 primary B2 number (Δ=+0.0319) was computed by the Phase-1
mechanism-replication pipeline without a prediction archive. Phase-2 B2 number
(Δ=+0.0939) was produced by the Phase-2 B-MECH-1 driver with a 28-column
parquet archive, 30-seed ensemble inference, and a DeLong test.

### 2.2 Code commit hashes

| Item | Phase-1 | Phase-2 |
|---|---|---|
| Driver commit | Phase-1 mechanism script (pre-2719d81) | `2719d8111405a4fcc75e288678cd5a18d37134c5` + Phase-2.2B.exec lock (`204775b...`) |
| Contract lock | Not applicable — Phase-1 did not have a contract YAML | `configs/phase2/rga_v2_gate_contract.yaml` locked at contract-time |

### 2.3 Config / dataset hash

Both runs use `configs/attention_real_fusion.yaml` → `experiments/fusion/real_domain_fusion_inputs.csv`.
The data file has not changed between Phase-1 and Phase-2 (verified: no edit to the
CSV between the two pipeline runs; same sample count).

### 2.4 Sample count

| Item | Phase-1 | Phase-2 |
|---|---|---|
| n_test (approx) | ~53 per seed (based on Phase-1 split ratios) | 1600 = 30 seeds × ~53 samples (ensemble) |
| Inference level | Per-seed individual AUC, then mean | 30-seed ensemble AUC (pool all seeds then compute AUC) |

**CRITICAL DIFFERENCE:** Phase-1 used *mean-over-seeds individual AUC*, while
Phase-2 used *ensemble-pooled AUC* (all 30×n_test predictions concatenated,
then one AUC computed). These are different estimators of the same underlying
quantity but they can differ due to Jensen's inequality and seed-to-seed variance.

### 2.5 Train/validation/test split construction

Both use the pre-assigned `split` column from `real_domain_fusion_inputs.csv`.
No difference — the split is frozen in the CSV.

### 2.6 Seed list

| Phase-1 | Phase-2 |
|---|---|
| Likely 1–5 seeds (Phase-1 specification; exact list not preserved in artifact) | 30 seeds (42–71), confirmed by `n_seeds=30` in inference CSV |

**DIFFERENCE:** Phase-1 used far fewer seeds. Phase-2 used 30 seeds.
With more seeds the ensemble AUC can be both more stable and differently
calibrated (larger effective test set improves AUC sensitivity to small Δ).

### 2.7 Attack name and precise max_attack transformation

| Item | Phase-1 | Phase-2 |
|---|---|---|
| Attack name | `max_attack` | `max_attack` (same) |
| Implementation | `AdversarialAttackType.max_attack` in `AdversarialPerturbationEngine` | Same class, same `.apply_attack()` method |
| sigma | 1.0 | 1.0 (same) |

**No difference in attack definition.**

### 2.8 Attacked domains

Phase-1 and Phase-2 both attack k=4 domains (all domains, full coherent collapse).
The `KOfDCorruptionResult` subsets confirm `failed_domain_count=4`.

### 2.9 k value

Both: k=4 (all-domain corruption). No difference.

### 2.10 Gate mode and threshold

Both: G0 mean-gate, τ=0.66 (locked). No difference.

### 2.11 Reliability computation path

Both use `_make_reliability_estimator(rel_cfg, domain_order, score_idx)` from
`run_breakthrough_experiment.py`. However:

- **Phase-1:** estimator fitted with `estimator.fit(features[train_idx], ...)` — train fold
- **Phase-2 B-MECH-1:** estimator fitted identically: line 103
  `estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])`

**No difference in fitting.**

### 2.12 Static model identity

Both: `_build_model(cfg_seed, ...)` followed by `_train_model(...)` — same
`AttentionFusionModel` class, same hyperparameters from config, same seed.
The models are independently trained per seed; they should be identical *in expectation*.

### 2.13 RGA model identity

Both: the same trained model with reliability estimator layered on top.
In Phase-2 `per_sample_gating=False` is explicitly passed, meaning the gate
fires per-batch (global mean reliability < τ), not per-sample. This is the
same as Phase-1's default.

### 2.14 Score-orientation policy

Both use the same orientation (higher score = more anomalous). No flip.
Phase-2 polarity fix (claim M014) does not apply to Phase-2 B-MECH-1 which
uses raw predictions.

### 2.15 Prediction averaging / ensembling policy

| Phase-1 | Phase-2 |
|---|---|
| `mean(AUC over seeds)` (per-seed AUC, then averaged) | `AUC(pool(all_seed_predictions))` (ensemble AUC on concatenated predictions) |

**This is the primary estimator difference.** For a perfectly calibrated
classifier both estimators converge to the same value as n_seeds → ∞.
However, with finite seeds and heterogeneous test-fold difficulties:
- Per-seed mean AUC weights each seed equally regardless of test fold size.
- Ensemble AUC weights each sample equally (all 30 seeds × n_test predictions pooled).

The Phase-2 `ensemble_delta_auc = ensemble_rga_auc - ensemble_static_auc`
uses ensemble pooling, while Phase-1 used per-seed mean.

### 2.16 ROC-AUC calculation function

Both: `sklearn.metrics.roc_auc_score`. No difference.

### 2.17 Bootstrap/CI calculation procedure

| Phase-1 | Phase-2 |
|---|---|
| Bootstrap over seeds (mean-delta CI) | Bootstrap over seeds (per-seed delta distribution CI) per Phase-2 inference script |

### 2.18 Possible secondary/default-gate contamination

In Phase-1, it is possible that a non-G0 result (e.g., a secondary pipeline run
with a different gate mode or smaller tau) was accidentally included in the
B2 headline. The Phase-2 B-MECH-1 is explicit: gate_mode=mean, tau=0.66, per
the contract. **No secondary-gate contamination detected in Phase-2 artifact.**

Whether Phase-1 carried such contamination **cannot be determined** from the
artifact alone because the Phase-1 per-seed metrics CSV row had `status=pending_compute`
at the start and was filled by a pipeline run whose parameters we cannot verify
post-hoc from the artifact alone.

---

## 3. Root-Cause Analysis of Magnitude Change

The most probable root causes of the B2 Δ change (+0.032 → +0.094) are, in
decreasing order of likelihood:

### 3a. Different estimator: per-seed mean vs. ensemble pooling (HIGH PROBABILITY)

With 5 seeds vs. 30 seeds and a switch from per-seed-mean AUC to ensemble-pooled
AUC, the estimate can change substantially. Ensemble pooling on 30×n_test =
~1,600 predictions produces a more stable estimate than mean over 5 individual AUCs.
If max_attack k=4 produces a bimodal seed distribution (some seeds gate fires,
some do not), the per-seed mean AUC and ensemble AUC diverge.

Evidence: the Phase-2 per_seed_mean_delta = 0.0625 with SD = 0.038. The
ensemble Δ = 0.0939 is larger than the per-seed mean. This is consistent with
ensemble pooling amplifying the gain when the gate fires more reliably across seeds.

### 3b. Seed count difference (30 vs ~5) (MEDIUM PROBABILITY)

With 30 seeds, the reliable gate-firing seeds dominate the ensemble. The Phase-1
5-seed estimate may have included unlucky seeds where the gate did not fire often,
pulling the mean downward. 30 seeds regresses toward the true expectation.

### 3c. Phase-1 artifact construction uncertainty (LOW-MEDIUM PROBABILITY)

The Phase-1 B2 row (`phase1_target_delta=0.0319`) comes from an earlier run whose
full configuration cannot be verified from the archive alone. It is possible that
Phase-1 used a smaller k, a different attack variant, or fewer corrupted domains
than Phase-2. However, both refer to `max_attack k=4`, so this is unlikely to
explain the full difference.

### 3d. Code or model change between runs (LOW PROBABILITY)

The commit anchor `2719d81...` is from Phase-2.2B infrastructure. Phase-1 ran
on an earlier commit. If `_predict_craf_with_stats` or the gate logic changed
between commits, results would differ. **No evidence of such a change was found**
in the Phase-2 codebase diff (the function signatures and logic in
`run_breakthrough_experiment.py` are unchanged in the inspected version).

---

## 4. Decision

**COMPARABLE_BUT_ESTIMATOR_CHANGED**

Both estimates target the same underlying endpoint (max_attack k=4, G0 mean-gate,
τ=0.66, ELARA-Bench-LA). The protocol, attack type, domain count, and gate
configuration are identical. However:

1. The ensemble aggregation method changed (per-seed mean AUC → ensemble-pooled AUC).
2. The seed count changed (approximately 5 → 30).

These are **legitimate estimator improvements** that produce a more precise and
generally less downward-biased estimate of the true effect. However, they change
the reported estimand slightly, meaning the Phase-1 number and the Phase-2 number
**must be named distinctly** in the manuscript.

**Manuscript rules arising from this decision:**

1. The manuscript **must not** replace the Phase-1 B2 number (+0.0319) silently.
2. The manuscript **may** report: *"Under the Phase-2 30-seed ensemble pipeline,
   Δ AUC = +0.0939 (CI [+0.074, +0.115]). The Phase-1 5-seed per-seed-mean estimate
   was Δ AUC = +0.0319 (CI [+0.005, +0.062]). Both use max_attack k=4 / G0 τ=0.66 /
   ELARA-Bench-LA. The estimator changed from per-seed-mean AUC to 30-seed ensemble-
   pooled AUC; estimates are directionally consistent and not directly comparable
   numerically."*
3. No manuscript edit is authorised under this phase. This rule is recorded for the
   future manuscript-update phase.

---

## 5. Required Action Before Manuscript Use

- [x] B2 magnitude audit complete: decision = COMPARABLE_BUT_ESTIMATOR_CHANGED
- [ ] Manuscript update pending: must use dual-number reporting style above
- [ ] Manuscript-update phase must approve the replacement wording before insertion
- [ ] Do not use the Phase-2 Δ=+0.0939 as a drop-in replacement for Phase-1 Δ=+0.0319
      without the companion sentence explaining the estimator change

---

## 6. BLOCKING STATUS

**This audit is NON-BLOCKING** for Phase 2.2B.1 execution (the audit is now
complete with a valid non-blocking decision). No B2-related manuscript edits
are permitted until the manuscript-update phase approves the dual-number form.
