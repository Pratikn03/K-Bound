# What remains for 100% Master Scenario C

**Current automated status:** run `audit_checklist_progress.py` (typically **~88%** development items).

## Cannot complete without you (external)

| Item | Action |
|------|--------|
| **D3 / M2 final transfer** | Download and register a **new untouched** RGB+depth dataset; add to `dataset_registry_v3.yaml`; never tune on its test fold. See `research_lock/M2_FINAL_AUDIT_PENDING_v1.yaml`. |
| **T7 confirmatory** | After all models frozen: **one-shot** eval on sealed M2 + Holm-corrected statistics. |
| **Gate E (transfer)** | Positive bootstrap CI on untouched M2 only (Eyecandies is development per D1). |
| **Gate F (flagship)** | All pillars P1–P6 + partial-failure sweep + temporal monitoring. |

## Recommended next commands (repo-local)

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8

# Full development refresh
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/complete_master_c_checklist.py

# 5-seed fusion retrain + archives (slow)
for s in 42 43 44 45 46; do
  PYTHONPATH=src .venv/bin/python src/scripts/run_breakthrough_experiment.py \
    --config configs/attention_real_fusion.yaml \
    --output experiments/fusion/master_c_real_domain_seed${s}.json \
    --archive-root elara_master_c/predictions/development \
    --seed $s
done
```

## Ratify

- **D4:** Healthcare GridPulse sealed as M3 **development** candidate (`M3_SEALED_CANDIDATE_v1.yaml`) — confirm in `DECISIONS_v1.md` if you agree.
