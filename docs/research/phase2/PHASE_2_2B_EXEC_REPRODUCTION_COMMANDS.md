# Phase 2.2B.exec — Reproduction Commands

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
source .venv/bin/activate
```

## 0. Pre-execution check

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ --no-header --tb=no -p no:warnings | tail -3
# expected: 535 passed / 11 skipped (or 536 / 10 after B-MECH-1 archives exist)
```

## 1. B-MECH-1 primary mechanism replication (~6 min, 30 seeds on M-series Mac)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_mechanism_replication.py \
    --experiment-id B-MECH-1 --seeds 30 --seed-start 42

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_b_mech_1_inference.py
# expected output: B1 + B2 both REPRODUCED
```

## 2. B-CERT-1 certificate audit (~10 s, post-hoc on archives)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_certificate_audit.py \
    --experiment-id B-CERT-1
# expected: max_attack_k4 CERTIFIED; zero_attack_k4 NOT_CERTIFIED
```

## 3. B-MECH-2 / B-MECH-3 / B-MECH-4 (scaffold drivers — no real compute)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_rga_v2_gate_sweep.py \
    --experiment-id B-MECH-2 --seeds 30 --seed-start 42 --gates G0,G1,G2,G3

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_mixture_shift.py \
    --experiment-id B-MECH-3 --seeds 5 --seed-start 42 --mixture-shifts 10

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_ks_power_sweep.py \
    --experiment-id B-MECH-4 --seeds 5 --seed-start 42
```

Each prints `[b-mech-N B-MECH-N] ... reserved for a future compute window` and exits 0. Closing these gaps requires the driver-loop implementation listed in [PHASE_2_2B_EXEC_REMAINING_GAPS.md](./PHASE_2_2B_EXEC_REMAINING_GAPS.md).

## 4. Forbidden

- Do NOT execute any Family-D driver. Family-D v1 is `INVALID_FOR_EXECUTION`; v2 is `V2_DESIGN_PENDING`.
- Do NOT edit the paper / thesis based on B1/B2 v2 numbers without going through a separate revision task.
- Do NOT promote RGA-v2 — the C1..C6 evidence does not exist.
