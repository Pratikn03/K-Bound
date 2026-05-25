# Phase 2 — Artifact Manifest

**Freeze point HEAD (pre-this-commit):** `6299c3f6fd2ab810ea97ccf2d5a84580f183fdd4`
**Manifest generated:** Phase 2 in-session pilot wrap-up.

## 1. Contract files (frozen — SHA256 anchors)

| Path | SHA256 |
|---|---|
| `docs/research/phase2/PHASE_2_RESEARCH_CONTRACT.md` | `3eb849f35aabc4220c2ed7777f58c8b56cd9094c475062b45b7337e344637733` |
| `docs/research/phase2/PHASE_2_STATISTICAL_POLICY.md` | `34b36c2c8bc8245c414affbe7e4d916cd9ebc7b30855dd89e47275735cd04ee4` |
| `docs/research/phase2/FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md` | `44b5812ca349ffb73df4f6cfc927870bc3e3dd2e74932a1dcc977b2ab639f1fb` |
| `docs/research/phase2/FAMILY_D_DATASET_INVENTORY.md` | `321d8007a566998370cccc437a82461b696b0773839c900ca3311fd2112b7c95` |
| `docs/research/phase2/FAMILY_D_HYPOTHESES.csv` | `8a835bb578c55be6f5f5c52691e74b8e4c7aacd9dd8c3e02fad7ba7467435131` |
| `docs/research/phase2/FAMILY_D_PARTITION_MANIFEST.json` | `b7d0f8843ef9cdc91d99283c23ba63717b618820e45bf9933ab944214df975be` |
| `docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md` | `9d3a67522868fba3c6255fac07c9725dab4fc22dd8b34586e432c0422df69a42` |
| `docs/research/phase2/FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md` | `e4fe5a0c0b2991fcb20b2d696bce415c7eee3d9aad156d75e06b505b5abc5dd8` |
| `configs/phase2/rga_v2_gate_contract.yaml` | `b2f59eaa3b5eda90d33740d0a7df3451fdffd28fcfc88db3c04d608b608d06f0` |

Re-verify any time with the shasum block in [PHASE_2_REPRODUCTION_COMMANDS.md](./PHASE_2_REPRODUCTION_COMMANDS.md) §F.

## 2. Code artefacts added in Phase 2

| Path | Purpose |
|---|---|
| `src/elara/__init__.py` | namespace export |
| `src/elara/evaluation/__init__.py` | namespace export |
| `src/elara/evaluation/prediction_archive.py` | 28-column Parquet schema, immutable `rerun_N` suffix, index management |
| `src/elara/evaluation/ensemble_inference.py` | fast-DeLong, paired-sample bootstrap (10 000 iter), Holm, audited-analysis driver |
| `src/elara/certification/__init__.py` | namespace export |
| `src/elara/certification/risk_dominance.py` | (q0, q1, Δ0, Δ1, π\*) risk-dominance terms |
| `src/elara/certification/switching_certificate.py` | paired-bootstrap LCB switching certificate |
| `src/scripts/run_phase2_powered_audited_pilot.py` | 30-seed pilot driver for A-POWERED-1 |
| `src/scripts/run_phase2_powered_audited_analysis.py` | audited inference driver on the prediction archive |
| `src/scripts/validate_phase2_prediction_archives.py` | SHA256 + schema + no-leakage integrity check |
| `tests/test_phase2_prediction_archive_schema.py` | 4 tests |
| `tests/test_phase2_prediction_archive_no_leakage.py` | 2 tests |
| `tests/test_phase2_validation_only_selection.py` | 5 tests |
| `tests/test_phase2_certification.py` | 4 tests |

Total Phase-2 tests passing: 15.

## 3. Data artefacts produced this session

| Path | Description | Rows |
|---|---|---|
| `experiments/phase2/predictions/A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired/` | 12 methods × 2 splits × 30 seeds Parquet files | 720 files |
| `experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv` | archive index | 721 lines (720 + header) |
| `experiments/phase2/statistics/family_a_powered_seed_metrics.csv` | per-seed RGA+ / static / CRAF / router / boost AUCs | 31 lines |
| `experiments/phase2/statistics/family_a_selection_log.csv` | per-seed selection trail (router vs boost) | 311 lines |
| `experiments/phase2/statistics/family_a_powered_ensemble_inference.csv` | seed-ensemble DeLong + bootstrap + Holm per comparator | 11 lines |
| `experiments/phase2/statistics/family_a_powered_holm_results.csv` | Holm-significance + CI-excludes-zero summary | 11 lines |

## 4. Scaffolded `pending_compute` artefacts (no execution this session)

- `experiments/phase2/mechanism/family_b_primary_replication_inference.csv` — B-MECH-1 scaffold.
- `experiments/phase2/mechanism/family_b_primary_replication_metrics.csv` — B-MECH-1 scaffold.
- `experiments/phase2/mechanism/rga_v2_failure_surface_metrics.csv` — B-MECH-2 scaffold.
- `experiments/phase2/mechanism/rga_v2_clean_false_fire.csv` — B-MECH-2 scaffold.
- `experiments/phase2/mechanism/rga_v2_threshold_selection.csv` — B-MECH-2 scaffold.
- `experiments/phase2/mechanism/ks_mixture_shift_control.csv` — B-MECH-3 scaffold.
- `experiments/phase2/mechanism/ks_true_degradation_power.csv` — B-MECH-4 scaffold.
- `experiments/phase2/mechanism/ks_window_size_power.csv` — B-MECH-4 scaffold.
- `experiments/phase2/certification/risk_dominance_terms.csv` — B-CERT-1 scaffold.
- `experiments/phase2/certification/switching_certificates.csv` — B-CERT-1 scaffold.

## 5. Markdown reports added in Phase 2

- `docs/research/phase2/PHASE_2_RESEARCH_CONTRACT.md`
- `docs/research/phase2/PHASE_2_STATISTICAL_POLICY.md`
- `docs/research/phase2/PHASE_2_COMPUTE_PLAN.md`
- `docs/research/phase2/PHASE_2_EXPERIMENT_REGISTRY.csv`
- `docs/research/phase2/PHASE_2_CLAIM_MATRIX.csv`
- `docs/research/phase2/FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md` (this session's headline result)
- `docs/research/phase2/FAMILY_B_PRIMARY_MECHANISM_REPLICATION_REPORT.md` (scaffold)
- `docs/research/phase2/RGA_V2_PARTIAL_FAILURE_REPORT.md` (scaffold)
- `docs/research/phase2/KS_REFERENCE_AND_POWER_REPORT.md` (scaffold)
- `docs/research/phase2/RISK_DOMINANCE_AND_CERTIFICATE_REPORT.md` (scaffold)
- `docs/research/phase2/FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md` (frozen)
- `docs/research/phase2/FAMILY_D_DATASET_INVENTORY.md` (frozen)
- `docs/research/phase2/FAMILY_D_HYPOTHESES.csv` (frozen)
- `docs/research/phase2/FAMILY_D_PARTITION_MANIFEST.json` (frozen)
- `docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md` (frozen)
- `docs/research/phase2/FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md` (frozen)
- `docs/research/phase2/PHASE_2_INTERIM_REPORT.md`
- `docs/research/phase2/PHASE_2_HOSTILE_REVIEW_REPORT.md`
- `docs/research/phase2/PHASE_2_REPRODUCTION_COMMANDS.md`
- `docs/research/phase2/PHASE_2_ARTIFACT_MANIFEST.md` (this file)
- `docs/research/phase2/PHASE_2_REMAINING_OPEN_GAPS.md`
