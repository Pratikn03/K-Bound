# Phase 2 Family-B — Final Artifact Manifest

**Phase:** 2.2B.2 / Step 7

## Code (executable)

| Path | Purpose |
|---|---|
| `src/scripts/run_phase2_mechanism_replication.py` | B-MECH-1 driver (k=4 corruption, 30 seeds, archives static + RGA predictions) |
| `src/scripts/run_phase2_b_mech_1_inference.py` | B-MECH-1 seed-ensemble DeLong + paired bootstrap + Holm K=2 |
| `src/scripts/run_phase2_b_mech_1_clean_arm.py` | **NEW** B-MECH-1 clean k=0 arm driver (30 seeds; uncorrupted test fold) |
| `src/scripts/run_phase2_rga_v2_gate_sweep.py` | B-MECH-2 driver (15 seeds × 4 gates × validation-only τ selection) |
| `src/scripts/run_phase2_mixture_shift.py` | B-MECH-3S domain-composition driver (5 seeds × 10 mixtures) |
| `src/scripts/run_phase2_ks_power_sweep.py` | B-MECH-4 KS window sweep driver (5 seeds × 5 windows × 3 degradations) |
| `src/scripts/run_phase2_certificate_audit.py` | B-CERT-1 v1 (fired-subset certificate only) |
| `src/scripts/run_phase2_b_cert_1_v2.py` | **NEW** B-CERT-1 v2 (clean+degraded ⇒ q₀, q₁, Δ₀, Δ₁, π*) |

## Code (model layer)

| Path | Notable additions |
|---|---|
| `src/uais/fusion/attention/reliability_estimator.py` | G3 top-q gate + `ks_window_size` parameter |
| `src/elara/family_b/corruption.py` | `inject_corruption()`, `validation_fold_corruption_grid()` |
| `src/elara/family_b/mixture_shift.py` | `pure_mixture_shift_resample()` with KS invariance check |
| `src/elara/family_b/ks_window.py` | locked `KS_WINDOW_GRID = (32, 64, 128, 256, 512)` |
| `src/elara/certification/risk_dominance.py` | `estimate_risk_dominance()` (q₀, q₁, Δ₀, Δ₁, π*) |
| `src/elara/certification/switching_certificate.py` | `fired_subset_certificate()` (paired-bootstrap LCB) |

## Data — primary

| Path | Rows | Description |
|---|---:|---|
| `experiments/phase2/mechanism/b_mech_1_prediction_archives/B-MECH-1__.../static_attention__zero_attack_k4/test/seed_NN.parquet` | 30 | B1 degraded arm (static) |
| `..._rga_mean_gate_tau66__zero_attack_k4/test/seed_NN.parquet` | 30 | B1 degraded arm (RGA) |
| `..._static_attention__max_attack_k4/test/seed_NN.parquet` | 30 | B2 degraded arm (static) |
| `..._rga_mean_gate_tau66__max_attack_k4/test/seed_NN.parquet` | 30 | B2 degraded arm (RGA) |
| `..._static_attention__clean_k0/test/seed_NN.parquet` | 30 | **clean k=0 arm (static)** [NEW] |
| `..._rga_mean_gate_tau66__clean_k0/test/seed_NN.parquet` | 30 | **clean k=0 arm (RGA)** [NEW] |
| `family_b_primary_replication_seed_metrics.csv` | 60 | per-seed × scenario metrics |
| `family_b_primary_replication_inference.csv` | 2 | seed-ensemble inference + Holm K=2 |
| `family_b_primary_replication_holm_k2.csv` | 2 | Holm summary |
| `b2_phase1_vs_phase2_comparability.csv` | 4 | Phase-1 vs Phase-2 side-by-side |

## Data — RGA-v2 sweep

| Path | Rows | Description |
|---|---:|---|
| `experiments/phase2/mechanism/rga_v2_threshold_selection.csv` | 60 | 15 seeds × 4 gates val-fold τ selection log |
| `rga_v2_clean_false_fire.csv` | 60 | per (seed, gate) clean activation rate |
| `rga_v2_failure_surface_metrics.csv` | ~2 880 | per (seed, gate, attack, k, subset) static/RGA AUC |
| `rga_v2_failure_surface_inference.csv` | 4 | per-gate promotion decision |
| `rga_v2_prediction_archives/B-MECH-2__.../**/seed_*.parquet` | many | per-(gate, attack, k) archived predictions |

## Data — B-MECH-3S and B-MECH-4

| Path | Rows | Description |
|---|---:|---|
| `experiments/phase2/mechanism/domain_composition_shift_metrics.csv` | 50 | 5 seeds × 10 mixtures |
| `ks_window_size_power.csv` | 25 | 5 windows × 5 seeds |
| `ks_true_degradation_power.csv` | 75 | 5 windows × 5 seeds × 3 degradation types |

## Data — certification (v1 + v2)

| Path | Rows | Description |
|---|---:|---|
| `experiments/phase2/certification/switching_certificates.csv` | 2 | v1 (G0 fired-subset certificate; pre-clean-arm) |
| `experiments/phase2/certification/risk_dominance_terms.csv` | 2 | v1 (inadmissibility note) |
| `experiments/phase2/certification/switching_certificates_v2.csv` | 2 | **v2 (G0; populated after clean arm)** [NEW] |
| `experiments/phase2/certification/risk_dominance_terms_v2.csv` | 2 | **v2 (G0; full q₀, q₁, Δ₀, Δ₁, π*)** [NEW] |

## Documents

| Path | Stage |
|---|---|
| `PHASE_2_LOCO_PAIRING_STRENGTH_AUDIT.md` | Step 1 |
| `PHASE_2_B1_B2_INTEGRATION_POLICY.md` | Step 2 |
| `RGA_V2_SELECTION_PROVENANCE_RECONCILIATION.md` | Step 3 |
| `RGA_V2_SEED_COUNT_DECISION.md` | Step 4 |
| `RISK_DOMINANCE_AND_CERTIFICATE_REPORT_FINAL.md` | Step 5 |
| `RGA_V2_CERTIFICATE_EXTENSION_DECISION.md` | Step 6 |
| `PHASE_2_FAMILY_B_FINAL_HOSTILE_REVIEW_REPORT.md` | Step 7 |
| `PHASE_2_FAMILY_B_FINAL_DECISION.md` | Step 7 (decision) |
| `PHASE_2_FAMILY_B_FINAL_ARTIFACT_MANIFEST.md` | Step 7 (this file) |

## Tests added in Phase 2.2B.2

| File | Cases | Coverage |
|---|---:|---|
| `test_phase2_loco_pairing_strength_verified.py` | 2 | A-POWERED-3 = derived_view_proxy |
| `test_phase2_b2_dual_number_policy.py` | 3 | B2 dual-number wording |
| `test_phase2_rga_v2_selection_provenance.py` | 3 | 15 seeds × 4 gates in all CSVs |
| `test_phase2_rga_v2_seed_count_decision.py` | 3 | YAML minimum_for_inference=15 + C1 fail |
| `test_phase2_risk_dominance_clean_arm.py` | 3 | clean arm + v2 CSVs |
| `test_phase2_rga_v2_certificate_extension_boundary.py` | 2 | v2 cert CSV contains only G0 rows |
| `test_phase2_family_b_final_decision.py` | 3 | decision label invariants |
| `test_phase2_master_status_consistency.py` | 3 | master refresh consistency |
| `test_family_d_v2_candidate_eligibility.py` | 3 | VisA excluded etc. |
| `test_family_d_v2_test_not_executed.py` | 2 | v2 freeze never carries executed=true |
| `test_phase2_manuscripts_unchanged_during_closure.py` | 1 | paper/thesis untouched |

## Pre-execution commit hash

`dbf8dca` (Phase 2.2B.1)

## Closure commit hash (filled after commit lands)

`<TO BE FILLED ON COMMIT>`
