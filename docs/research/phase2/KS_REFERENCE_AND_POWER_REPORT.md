# KS Reference + Power Report (B-MECH-3 + B-MECH-4)

**Status:** scaffold + contract; execution **pending_compute** in this task.

---

## 1. Three experiments

### F1 — Pure mixture-shift control (B-MECH-3)

Construct a controlled scenario where:
- per-category / per-domain score distributions are unchanged across val and test;
- only category or cohort proportions change (`π_val ≠ π_test`);
- no true detector corruption is injected.

Compare gate behaviour under:
- global KS reference (existing default);
- category-aware KS reference (`CategoryAwareReliabilityEstimator`, already implemented in Phase 1.C);
- cohort-aware reference (where available);
- any learned reference only if locked before evaluation.

Measure: gate false-activation rate, reliability score shift, ROC-AUC effect of unnecessary switching, calibration effect where interpretable.

Permitted claim: "Category-aware reliability reduces false firing under the evaluated pure mixture-shift controls." Forbidden: any "solves distribution shift" / "guarantees no false alarms" / "valid for every deployment cohort" framing.

### F2 — True degradation power

Apply real corruption (score collapse, score noise, missingness, miscalibration) and measure detection power per reference type. Reports detection-power vs false-alarm-rate tradeoffs.

### F3 — Window-size / sample-size power sweep

KS window in {32, 64, 128, 256, 512}. Measures the window/power tradeoff. No claim outside the evaluated window set.

## 2. Driver

`src/scripts/run_phase2_ks_reference_power.py` (skeleton present; full implementation deferred). The driver reuses the existing `CategoryAwareReliabilityEstimator` (Phase 1.C) and `PerSampleReliabilityEstimator` (Tier-C).

## 3. Output artifacts

| Path | Schema | Status |
|---|---|---|
| `experiments/phase2/mechanism/ks_mixture_shift_control.csv` | reference_type, n_val, n_test, π_val, π_test, gate_false_activation_rate, reliability_score_shift, roc_auc_effect_of_unnecessary_switching, calibration_effect | scaffold only |
| `experiments/phase2/mechanism/ks_true_degradation_power.csv` | degradation_type, reference_type, detection_power, gate_activation_rate, false_negative_rate, roc_auc_delta | scaffold only |
| `experiments/phase2/mechanism/ks_window_size_power.csv` | window_size, false_activation_rate, true_degradation_detection_power, roc_auc_effect | scaffold only |

## 4. Status

**pending_compute.** Driver and contract are frozen; future compute-budgeted sessions can execute without protocol drift.
