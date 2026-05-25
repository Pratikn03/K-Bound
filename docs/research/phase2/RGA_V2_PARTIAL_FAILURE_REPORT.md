# RGA-v2 Partial-Failure Gate Report (B-MECH-2)

**Status:** contract + scaffold; execution **pending_compute** in this task.

**Locked weakness (Phase 1 evidence):** the base mean-gate G0 reliably activates only at the saturated k=4 coherent-collapse regime. At k=1, k=2, k=3 the gate is closed (mean reliability remains > τ=0.66), and the per-row Δ values across the k-of-D sweep show near-zero improvement for partial failures. RGA-v2 aims to extend the useful operating region without inflating the clean false-fire rate.

The full evaluation contract is locked in `configs/phase2/rga_v2_gate_contract.yaml`. Headlines below; see that file for the authoritative spec.

---

## 1. Candidate gate family

| id | name | rule | validation tuning |
|---|---|---|---|
| G0 | existing mean gate (baseline) | `I(mean(r) < 0.66)` | none (τ locked) |
| G1 | validation-selected minimum gate | `I(min(r) < τ_min)` | τ_min over locked grid, val-only |
| G2 | validation-selected hybrid gate | `I(mean(r) < 0.66 OR min(r) < τ_min)` | τ_min over locked grid, val-only |
| G3 | top-q critical-domain gate | `I(r_{q-th smallest} < τ_q)` | (q, τ_q) joint search, val-only |
| G4 | learned low-capacity gate (optional) | logistic on reliability features | architecture locked; val-only training |

## 2. Fault surface

| factor | values |
|---|---|
| k (failed domains) | 0, 1, 2, 3, 4 |
| attack | zero_attack, max_attack, gaussian_noise |
| missing_domain_failure | enabled |
| miscalibration_without_score_collapse | optional |

## 3. Clean false-fire budget (LOCKED before evaluation)

`rule = max(0.010, base_G0_clean_activation_rate + 0.005)`

Computed on the validation fold, before inspecting test outcomes. Cannot be overridden after evaluation begins.

## 4. Promotion criteria (all required)

| id | criterion |
|---|---|
| C1 | within_false_fire_budget |
| C2 | improves at least 2 of {k=1, k=2, k=3} over G0 (zero + max attacks) |
| C3 | does not worsen k=4 by more than 0.005 ROC-AUC vs G0 |
| C4 | positive switching certificate (LCB > 0) on at least one partial-failure regime |
| C5 | selection is validation-only (Phase 2.B contract verified) |
| C6 | same gate policy across all cells (no per-test-cell re-tuning) |

## 5. Promotion decisions

- `PROMOTED_CANDIDATE` — all C1..C6 pass.
- `MECHANISM_IMPROVEMENT_PARTIAL` — some partial-failure benefit; not all criteria pass.
- `NOT_IMPROVED` — does not improve the locked weakness.
- `INVALID_SELECTION` — any test-driven tuning found.

## 6. Driver

`src/scripts/run_phase2_rga_v2_partial_failure.py` (skeleton present; full implementation deferred to compute-budgeted session). The driver MUST:
1. Read `configs/phase2/rga_v2_gate_contract.yaml`.
2. Refuse to deviate from the locked τ_mean, the locked grids, or the locked clean false-fire rule.
3. Archive predictions per gate × seed × split into `experiments/phase2/mechanism/rga_v2_prediction_archives/`.
4. Emit `rga_v2_threshold_selection.csv`, `rga_v2_failure_surface_metrics.csv`, `rga_v2_clean_false_fire.csv`.

## 7. Status in this task

**pending_compute** for the full failure-surface sweep (5 gates × 5 k values × 3 attacks × 30 seeds ≈ many hours). Contract is frozen so future runs cannot drift.
