# Family B Primary Mechanism Replication Report (B-MECH-1)

**Status:** scaffold + contract; execution **pending_compute** in this task.

**Locked target (Phase 1.1 PRIMARY B1/B2 endpoints):**
- B1 zero-attack all-domain at locked τ=0.66 (k-of-D k=4 mean-gate): **Δ ROC-AUC = +0.0506** [95% CI 0.0315, 0.0681].
- B2 max-attack all-domain at locked τ=0.66 (k-of-D k=4 mean-gate): **Δ ROC-AUC = +0.0319** [95% CI 0.0050, 0.0617].

Source: `experiments/fusion/craf_real_k_domain_results.json`,
`table_10_k_domain_corruption` rows where `attack ∈ {zero_attack, max_attack}` AND `failed_domain_count = 4` AND `gate_mode = mean`. Documented in `PHASE_1_1_PRIMARY_RUN_RESOLUTION.md`.

---

## 1. Phase 2.D scope

This stage is a **mechanism replication** of the locked PRIMARY B1/B2 endpoints on the same ELARA-Bench-LA dataset (`real_domain_fusion_inputs.csv`, scorer_train_fraction=0.05, k-of-D corruption harness), with the new Phase 2.B prediction-archive contract enabled. The goal is to confirm:

1. **Direction stability** — B1/B2 Δ-signs match Phase-1 (positive).
2. **Magnitude stability** — Δ-magnitudes within the Phase-1 95% CI.
3. **Bootstrap CI** — paired sample bootstrap CI on the seed-averaged ensemble Δ; the CI excluding zero is the stronger evidence (cf. seed-level paired bootstrap was rejected by Phase 0.5 / 1.D).
4. **Gate activation** — mean-gate at τ=0.66 fires at k=4 (per T3 prediction) and stays closed at k<4.

This is mechanism replication on the same benchmark family — **not** cross-domain confirmation.

## 2. Execution contract (frozen)

| Item | Value |
|---|---|
| Cell id | B-MECH-1 |
| Benchmark | ELARA-Bench-LA |
| Inputs | `experiments/fusion/real_domain_fusion_inputs.csv` |
| Config | `configs/attention_real_fusion.yaml` (with `k_domain_corruption_values: [0,1,2,3,4]`) |
| Scorer train fraction | 0.05 (per `real_domain_fusion_metadata.json`) |
| τ (gate threshold) | locked at 0.66 (`clean_gate_threshold: 0.66`) |
| k | 4 (all-domain coherent corruption) |
| Gate mode | mean (the locked PRIMARY gate) |
| Method | base RGA (static + reliability-gated attention path) |
| Seeds | 30 (target); ≥5 if compute-constrained → downgrade to "pilot mechanism replication" |
| Prediction archive | required (Phase 2.B contract) |
| Primary statistic | seed-ensemble Δ ROC-AUC + paired-sample-bootstrap 95% CI (10 000 iterations) |
| Secondary | per-seed AUROC mean ± SD, sign-consistency count, gate activation rate, mean / min reliability |
| Multiplicity | Holm-Bonferroni within `B-MECH-K2` (B1 + B2 only) |
| Allowed claim | "Mechanism replication reproduces / does not reproduce the locked PRIMARY B1/B2 endpoints" |
| Forbidden claim | "confirmatory", "cross-domain generalization", "broad superiority" |

## 3. Execution driver

Driver: `src/scripts/run_phase2_family_b_mechanism_replication.py` (built in this task; see file). Reuses the existing k-of-D corruption harness in `run_breakthrough_experiment.py:_evaluate_k_domain_corruption` and adds the Phase 2.B prediction-archive call for each seed.

## 4. Decision labels (locked before evaluation)

After execution the report will classify the outcome as exactly one of:

- **Reproduced**: Phase 2 Δ direction agrees with Phase 1 AND the paired-sample bootstrap 95% CI excludes zero.
- **Directionally supported**: direction agrees but uncertainty CI includes zero.
- **Not reproduced**: sign reverses, or the expected endpoint is not observed.
- **Inconclusive**: artifact / protocol failure (e.g., insufficient seeds, archive validation failure).

## 5. Status in this task

**pending_compute.** The Phase 2 task scope (user-confirmed: contracts + infra + 1-cell pilot only) does not include B-MECH-1 execution. The contract is frozen here so a future compute-budgeted session can run the driver without protocol drift.
