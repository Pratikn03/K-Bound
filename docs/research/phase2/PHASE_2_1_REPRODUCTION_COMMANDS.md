# Phase 2.1 — Reproduction Commands

All commands assume the project venv is active and the working directory is the repo root.

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
source .venv/bin/activate
```

## A. Verify the Phase-2.1 repair is in place (no compute)

```bash
# A.1 — Regenerate v2 registries deterministically
PYTHONPATH=src .venv/bin/python src/scripts/emit_phase2_registries_v2.py

# A.2 — Run the 8 Phase-2.1 contract-integrity test files
PYTHONPATH=src .venv/bin/python -m pytest \
    tests/test_phase2_registry_csv_schema.py \
    tests/test_phase2_registry_family_alignment.py \
    tests/test_phase2_report_registry_consistency.py \
    tests/test_family_d_v1_never_executable.py \
    tests/test_family_d_v2_no_placeholders_before_freeze.py \
    tests/test_family_d_no_previously_touched_dataset.py \
    tests/test_family_d_claim_boundary.py \
    tests/test_phase2_no_forbidden_claims_in_manuscripts.py \
    --no-header --tb=short -q

# A.3 — Full test suite (expected: 431 passed / 7 skipped)
PYTHONPATH=src .venv/bin/python -m pytest tests/ \
    --no-header --tb=no -p no:warnings | tail -3
```

## B. (Future) — full Phase-2 audited compute under v2 policy

**These commands run actual training and inference. Phase 2.1 does NOT
execute them. They are documented here so that the next compute window
can run the full Phase-2 evidence sweep under the v2 policy.**

### B.1 — Family A: A-POWERED-1 primary-surface recompute (no retraining; uses existing archive)

```bash
# Recompute the primary Family-A surface for A-POWERED-1 against
# static_attention only (K=5 family, not K=10 within-cell).
# This is a pure CPU recompute against the existing prediction
# archive. It does not retrain any model.
PYTHONPATH=src .venv/bin/python -c "
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, 'src')
from elara.evaluation.ensemble_inference import audited_analysis

archive = pathlib.Path('experiments/phase2/predictions/A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired')
sm = pd.read_csv('experiments/phase2/statistics/family_a_powered_seed_metrics.csv').drop_duplicates(subset=['seed'])
chosen = dict(zip(sm['seed'].astype(int), sm['chosen_head']))

def load(method):
    out = {}
    for p in sorted((archive/method/'test').glob('seed_*.parquet')):
        s = int(p.stem.replace('seed_',''))
        df = pd.read_parquet(p)
        out[s] = (df['sample_id'].to_numpy(), df['label'].to_numpy().astype(int), df['raw_score'].to_numpy().astype(float))
    return out

router = load('rga_meta_router'); boost = load('rga_boosted_fusion'); static = load('static_attention')
seeds = sorted(chosen)
ids, lbl, _ = router[seeds[0]]
rga = {s: (router[s][2] if chosen[s]=='router' else boost[s][2]) for s in seeds}
stat = {s: static[s][2] for s in seeds}

res = audited_analysis(
    cell_id='A-POWERED-1__PRIMARY',
    benchmark='MVTec 3D-AD', protocol='PatchCore supervised-paired',
    rga_method='rga_plus_validation_frozen',
    comparator_method='static_attention',
    sample_ids=ids, labels=lbl,
    per_seed_rga_scores=rga, per_seed_comp_scores=stat,
)
print(f'PRIMARY_FAMILY_A_CELL_LEVEL A-POWERED-1:')
print(f'  RGA+ ensemble AUC = {res.ensemble_rga_auc:.4f}')
print(f'  static_attention ensemble AUC = {res.ensemble_comparator_auc:.4f}')
print(f'  Delta = {res.ensemble_delta_auc:+.4f}')
print(f'  DeLong p (raw, no K=5 Holm yet) = {res.delong_p_value:.4g}')
print(f'  Bootstrap 95% CI = [{res.bootstrap_ci_low:+.4f}, {res.bootstrap_ci_high:+.4f}]')
print(f'  Effect band = {res.practical_effect_band}')
print(f'  K=5 Holm-adjusted p = pending_full_family (need A-POWERED-2..5)')
"
```

### B.2 — Family A: A-POWERED-2..5 (full 30-seed runs, requires real benchmarks)

**Each command is the 30-seed pilot driver re-pointed at the relevant
benchmark. The driver currently hard-codes MVTec 3D-AD PatchCore SP;
it must be parameterized first by adding `--benchmark` / `--protocol`
flags. That parameterization is a follow-up coding task; it is NOT
done in Phase 2.1.**

```bash
# Pattern (NOT directly runnable — requires the parameterization above):
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --experiment-id A-POWERED-2 --benchmark "MVTec 3D-AD" --protocol "PatchCore held-out category" \
    --seeds 30 --seed-start 42

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --experiment-id A-POWERED-3 --benchmark "MVTec LOCO-AD" --protocol "PatchCore supervised-paired" \
    --seeds 30 --seed-start 42

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --experiment-id A-POWERED-4 --benchmark "VisA" --protocol "RGB+edge supervised-paired" \
    --seeds 30 --seed-start 42

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --experiment-id A-POWERED-5 --benchmark "UNSW-NB15" --protocol "flow/conn/context" \
    --seeds 30 --seed-start 42
```

### B.3 — Family-A K = 5 Holm correction (only after all 5 cells exist)

```bash
# After all 5 Family-A archives are populated:
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_analysis.py \
    --family A --primary-comparator static_attention --multiplicity K5
# (The --family / --multiplicity flags are a follow-up parameterization
# of the existing audited-analysis driver.)
```

### B.4 — Family B (B-MECH-1..4 + B-CERT-1)

```bash
# (Follow-up implementations; not in Phase 2.1.)
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_mechanism_replication.py \
    --cell B-MECH-1 --seeds 30 --seed-start 42

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_rga_v2_gate_sweep.py \
    --cell B-MECH-2 --seeds 30 --seed-start 42 \
    --contract configs/phase2/rga_v2_gate_contract.yaml

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_ks_power_sweep.py \
    --cell B-MECH-3 --seeds 5 --seed-start 42
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_ks_power_sweep.py \
    --cell B-MECH-4 --seeds 5 --seed-start 42

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_certificate_audit.py \
    --cell B-CERT-1
```

## C. (Forbidden) — Family-D execution

Family-D execution is **forbidden** in Phase 2.1 and forbidden under
the current v1 contract. Family-D execution may be initiated only
after:

1. The Family-D v2 eligibility review is closed
   ([FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md](./FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md)).
2. A complete v2 contract is written and frozen with NO placeholders
   ([FAMILY_D_V2_DESIGN_STATUS.md](./FAMILY_D_V2_DESIGN_STATUS.md) §2).
3. An independent external review of the v2 contract has signed off.

No reproduction command for Family-D execution is given here on
purpose.

## D. Status snapshot

After running A.1 + A.2 + A.3, the expected state is:

- v2 registries valid; 16 + 17 = 33 rows total, schema-clean.
- 25 Phase-2.1 tests pass (5 skipped: v2 placeholder guards against
  files that don't exist yet, which is the correct state).
- Full suite: 431 passed, 7 skipped.
- A-POWERED-1 secondary-pilot-audit numbers remain RGA+ AUC = 0.7420
  with 5/10 comparators Holm-significant within-cell.
- A-POWERED-2..5 are `pending_compute`.
- Family-D v1 is `INVALID_FOR_EXECUTION`; v2 is `V2_DESIGN_PENDING`.
- Final Phase-2.1 decision: **PASS FOR CONTINUED FAMILY-A/B COMPUTE ONLY**.
