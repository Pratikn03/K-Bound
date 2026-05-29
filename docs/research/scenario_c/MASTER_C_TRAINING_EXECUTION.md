# Master Scenario C — Training Execution Map

This document maps your **T0–T12** checklist to this repository.

**Live progress:** run `src/scripts/scenario_c/audit_checklist_progress.py` →
`elara_master_c/audits/MASTER_C_CHECKLIST_STATUS.md` (currently **~62%** scaffolding
done; confirmatory training **not** complete).

## Governance (T0) — PASS when validator green

| Task | Repo artifact | Status |
|------|---------------|--------|
| Freeze manuscript baseline | `research_lock/BASELINE_STATE_v1.md` | exists |
| Archive Family A/B/D | `docs/research/audit/EXPERIMENT_REGISTRY.csv`, `research_lock/family_d_failure_record.md` | exists |
| Eyecandies Policy B | `research_lock/DECISIONS_v1.md` D1, `dataset_registry_v2.yaml` | **ratified** |
| New untouched M2 | `frozen_test_sets_v2.yaml` → `m2_new_untouched_transfer` | **OPEN (D3)** |
| Non-vision M3 | D4 in `DECISIONS_v1.md` | **OPEN** |
| Primary endpoints | `research_lock/primary_endpoints_v1.yaml` | frozen |
| Statistical policy | `research_lock/statistical_policy_v1.md` | frozen |
| Training critical fixes | `audits/training_truth_audit/12_training_critical_fixes.md` | applied |
| T0 validator | `src/scripts/scenario_c/validate_master_c_governance.py` | run before training |

```bash
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/validate_master_c_governance.py
```

## Staged training (do not merge stages)

| Stage | Train? | Entry point |
|-------|--------|-------------|
| T1 splits / experts | yes | `run_training_stage.py --stage T1` |
| T2 calibration | yes | `monitor_calibration.py` |
| T3 static + baselines | yes | `run_breakthrough_experiment.py`, harness |
| T4 base RGA | yes | `run_phase2_mechanism_replication.py` |
| T5 RGA+ | yes | `run_phase2_powered_audited_pilot.py` |
| T6 monitor | yes | `audit_gate_decision_rule_e2e.py` |
| T7 confirmatory | **no** | blocked until D3 + Phase-5 baseline freeze |

Registry: `elara_master_c/configs/training_stage_registry.yaml`

## Locked attention baseline (Static-Attention-v1)

| Parameter | Value | Config |
|-----------|------:|--------|
| Domains (ELARA-Bench-LA) | 4 | `configs/attention_real_fusion.yaml` |
| Hidden dim | 48 | `embed_dim` |
| Heads / layers | 4 / 1 | |
| Epochs | 25 | |
| Domain dropout | 0.15 | |
| Seeds | 42–46 | harness |
| Early stop | **val PR-AUC** | `early_stopping_metric: pr_auc` |
| Checkpoint | **restore best val** | `restore_best_weights: true` |

Legacy reproduction (pre-fix locked JSON): `hyperparameter_search_space_v1.yaml` → `legacy_reproduction`.

## Dataset roles (effective after v2)

| Dataset | Role | Final claim? |
|---------|------|--------------|
| ELARA-Bench-LA | M0 development | mechanism only |
| MVTec 3D-AD | M1 development | bounded / diagnostic |
| Eyecandies | M2 **development** (D1) | no — use new M2 for P4 |
| Visa / LOCO / UNSW | development | bounded |
| **New M2** | final audit | **required for transfer** |

## Immediate order (§19 of your plan)

1. ✅ Repo alignment — `elara_master_c/`, stage runner, v2 registries  
2. Prediction logging — `src/elara/evaluation/prediction_archive.py` (use for all new runs)  
3. ELARA-Bench-LA as M0 only — `natural_pairing: false` in metadata  
4. MVTec RGB+depth experts — `prepare_mvtec3d_fusion_benchmark.py`  
5. Static + strongest baselines — breakthrough + Phase 5 (pending)  
6. Base RGA partial failures — mechanism replication  
7. RGA+ vs strongest frozen baseline — **P2 open**  
8. Eyecandies development diagnostics — allowed post-D1  
9. Select + seal new M2 — **D3**  
10. Non-vision M3 — **D4**  
11. Monitor / abstention — GDR audit path  
12. One-shot T7 — after freeze  

## Pass/fail gates (summary)

| Gate | Blocker if fail |
|------|-----------------|
| A — experts | Do not train headline fusion |
| B — baselines | No ELARA improvement claim |
| C — base RGA | Mechanism-only |
| D — RGA+ | No superiority claim |
| E — transfer | No transferable reliability (needs new M2) |
| F — Scenario C | Not flagship until all pillars |

Full contract: `research_lock/SCENARIO_C_CLAIM_CONTRACT.md`
