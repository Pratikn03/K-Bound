# RGA-v2 Partial-Failure Report — v2

**Cell:** B-MECH-2
**Status:** **EXECUTED — 15-seed gate sweep complete.**

## 1. Execution status

The B-MECH-2 driver `src/scripts/run_phase2_rga_v2_gate_sweep.py` was executed with 15 seeds (42–56), satisfying the minimum seed plan requirement for statistical inference.

For each candidate gate, thresholds were selected using validation-fold corrupted data only (zero test visibility during tuning). Performance was then evaluated across the test-fold failure surface (k ∈ {0,1,2,3,4} × {zero_attack, max_attack, gaussian_noise}).

### Promotion Decisions

| Gate | Name | Mean Clean False-Fire Rate | Budget | C1 (False Fire) | C2 (Improve partial k) | C3 (k=4 unchanged) | C5/C6 (Tuning/Policy) | Promotion Decision |
|---|---|---|---|---|---|---|---|---|
| **G0** | Mean Gate (baseline) | 0.0000 | 0.0100 | **Pass** | Fail (0/2+) | Pass | Pass | **BASELINE_REFERENCE** |
| **G1** | Min Gate | 1.0000 | 0.0100 | **Fail** | Fail (0/2+) | Pass | Pass | **NOT_IMPROVED** |
| **G2** | Hybrid Gate | 1.0000 | 0.0100 | **Fail** | Fail (0/2+) | Pass | Pass | **NOT_IMPROVED** |
| **G3** | Top-q Gate | 1.0000 | 0.0100 | **Fail** | Fail (0/2+) | Pass | Pass | **NOT_IMPROVED** |

Source: [experiments/phase2/mechanism/rga_v2_failure_surface_inference.csv](../../../experiments/phase2/mechanism/rga_v2_failure_surface_inference.csv).

---

## 2. Findings and Empirical Analysis

1. ** Tunisie / Threshold Selection:**
   Tunable gates (G1, G2, G3) were successfully tuned on validation-fold corruption grids only, with zero leakage of test-fold metrics. G1 chose $\tau_{min} = 0.34$, G2 chose $\tau_{min} = 0.34$, and G3 chose $q=1, \tau_q=0.34$.
   
2. **False Fire Budget Failure (C1):**
   The minimum and hybrid gates (G1, G2, G3) fired 100% of the time (activation rate of 1.0000) on clean test data. Under the batch-level gating implementation, if *any* domain of *any* sample in a batch has a reliability weight falling below the selected threshold, the gate fires for the entire batch. This batch-level minimum pooling is highly sensitive, causing the gate to activate on every clean test batch.
   
3. **No Promotion:**
   No RGA-v2 candidate is promoted to production. G0 (the default mean-gate at $\tau=0.66$) remains the locked production-of-record gate. The audit confirms that minimum-based gating at the batch level is too unstable for false-fire control.

---

## 3. Provenance and Integrity

- **Locked Contract Compliance:** The driver strictly read `configs/phase2/rga_v2_gate_contract.yaml` and refused to deviate from the locked candidate gate formulations, search grids, or clean budget formula.
- **Archive Completeness:** Full prediction archives are saved to `experiments/phase2/mechanism/rga_v2_prediction_archives/`.
- **Test Integrity:** No test metrics were visible to the tuning loop. The selection trail has been verified as validation-only.
