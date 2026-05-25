# Phase 2.2B — Family-B Infrastructure Audit (pre-execution)

**Status:** read this before running any Family-B compute.

Before executing Phase 2.2B I audited which Family-B cells are infrastructure-ready in the current repo and which require new model code. Honest summary below.

## B-MECH-1 — primary B1/B2 replication on ELARA-Bench-LA

**Infrastructure status:** **READY** (with one wrap-up step).

- ELARA-Bench-LA config exists at [configs/attention_real_fusion.yaml](../../../configs/attention_real_fusion.yaml).
- Prepared data exists at `experiments/fusion/real_domain_fusion_inputs.csv` (28 110 rows, 4 domains, with fusion_split column).
- k-of-D corruption code exists at [src/scripts/run_breakthrough_experiment.py](../../../src/scripts/run_breakthrough_experiment.py):
  - `_k_domain_corruption_conditions()` builds the (attack, k) corrupted feature tensors.
  - `_evaluate_k_domain_corruption()` runs static + RGA at each (attack, k, gate_mode) and reports static_auc / craf_auc / gate stats.
- G0 mean-gate at τ=0.66 is already the locked baseline in the runner.

**Gap to close:** the existing `_evaluate_k_domain_corruption()` returns only aggregate AUC per (attack, k, gate_mode). For the Phase-2.B contract I need **per-sample raw prediction archives** (parquet) at (zero_attack, k=4, mean-gate) and (max_attack, k=4, mean-gate) for static and RGA. I will write a focused driver that calls the existing `_predict_static()` / `_predict_craf_with_stats()` primitives directly and archives raw probabilities in the Phase-2.B 28-column parquet schema (same pattern as the A-POWERED pilot driver).

**Wall-clock estimate:** ELARA-Bench-LA × 30 seeds × (zero + max attacks at k=4 mean-gate) ≈ similar to A-POWERED-5 (UNSW), roughly **30–60 minutes**.

## B-MECH-2 — RGA-v2 partial-failure gate sweep

**Infrastructure status:** **BLOCKED — material new model code required.**

The locked YAML at [configs/phase2/rga_v2_gate_contract.yaml](../../../configs/phase2/rga_v2_gate_contract.yaml) requires evaluation of G0..G3 (G4 optional). Current repo state:

- G0 mean gate: implemented in [ReliabilityEstimator](../../../src/uais/fusion/attention/reliability_estimator.py).
- G1 minimum gate: implemented.
- G2 hybrid gate: implemented.
- **G3 top-q critical-domain gate: NOT implemented.** No code path consumes `q_search_grid: [1, 2]` or `tau_q_search_grid`.
- **G4 learned low-capacity gate: NOT implemented.** Marked optional in the contract.

Additionally:

- The contract requires `tau_min` and `tau_q` to be selected on **validation-fold corruption injections** (`validation_tuning_data: validation-fold corruption injections only`). The existing `_evaluate_k_domain_corruption` injects corruption only on the test fold. A validation-fold corruption injection step is **NOT implemented**.

**Decision:** B-MECH-2 cannot be executed in this session without (a) implementing G3 top-q gate logic and (b) adding validation-fold corruption injection. Both are non-trivial model-code changes that fall outside a single execution phase.

**Status label:** `EXECUTION_BLOCKED_INFRASTRUCTURE_G3_TOPQ_AND_VALIDATION_CORRUPTION`.

## B-MECH-3 — pure mixture-shift false-fire control

**Infrastructure status:** **BLOCKED — material new model code required.**

The contract requires:
- A controlled mixture-shift sampler that holds within-category score distributions constant while varying category proportions.
- A category-aware (cohort-aware) reliability reference, callable separately from the global KS path.

Current repo state:
- The current `ReliabilityEstimator` supports a `category_aware` flag (line 1959 of `run_breakthrough_experiment.py`) but only as a flag toggle inside the existing training loop — there is no driver that constructs the pure-mixture-shift controlled sampler.
- No mixture-shift evaluation harness exists.

**Status label:** `EXECUTION_BLOCKED_INFRASTRUCTURE_MIXTURE_SHIFT_SAMPLER`.

## B-MECH-4 — KS true-degradation power × window-size sweep

**Infrastructure status:** **BLOCKED — material new model code required.**

The contract requires evaluating detection power × false-activation-rate across window sizes ∈ {32, 64, 128, 256, 512}. The current `ReliabilityEstimator` does not parameterize the KS reference window size in a way that is sweep-able; the window-size sweep harness does not exist.

**Status label:** `EXECUTION_BLOCKED_INFRASTRUCTURE_KS_WINDOW_SWEEP`.

## B-CERT-1 — risk-dominance + retrospective switching certificate

**Infrastructure status:** **CONDITIONALLY READY.**

The certificate code itself is fully implemented:
- [src/elara/certification/risk_dominance.py](../../../src/elara/certification/risk_dominance.py) — `estimate_risk_dominance()` computes (q0, q1, Δ0, Δ1, π*).
- [src/elara/certification/switching_certificate.py](../../../src/elara/certification/switching_certificate.py) — `fired_subset_certificate()` produces the paired-bootstrap LCB certificate.
- 4 tests passing at [tests/test_phase2_certification.py](../../../tests/test_phase2_certification.py).

B-CERT-1 can run **iff** there exists a paired clean/degraded prediction archive that includes per-sample reliability and gate-fired vectors. B-MECH-1's archived k=4 zero-attack vs clean baseline is the natural input.

**Status:** runnable after B-MECH-1 archives are produced.

---

## Execution plan summary

Given the infrastructure audit:

| Cell | Status | Action this phase |
|---|---|---|
| B-MECH-1 | READY | execute 30-seed run; produce archives + replication decision |
| B-MECH-2 | BLOCKED (G3 not implemented; validation-fold corruption not implemented) | mark `EXECUTION_BLOCKED_INFRASTRUCTURE`; document required code work for a follow-up Phase 2.2B.1 |
| B-MECH-3 | BLOCKED (mixture-shift sampler + category-aware KS reference not implemented) | mark `EXECUTION_BLOCKED_INFRASTRUCTURE` |
| B-MECH-4 | BLOCKED (KS window-size sweep harness not implemented) | mark `EXECUTION_BLOCKED_INFRASTRUCTURE` |
| B-CERT-1 | CONDITIONAL on B-MECH-1 archives | execute against B-MECH-1 archives after they complete |

**Implication for the final decision:** the best achievable Phase 2.2B outcome in this session is **`PASS FOR MECHANISM REPLICATION ONLY`** (or **`FAIL`** if B1/B2 do not reproduce). The `THEORY-CLOSURE` and `RGA-v2 ADVANCEMENT` decisions are infrastructure-gated to a future phase.

I am proceeding with the partial scope: B-MECH-1 + B-CERT-1, with B-MECH-2/3/4 marked `EXECUTION_BLOCKED_INFRASTRUCTURE`. The required new code for the blocked cells is enumerated in [PHASE_2_2B_REMAINING_GAPS.md](./PHASE_2_2B_REMAINING_GAPS.md) (written at end of this phase).
