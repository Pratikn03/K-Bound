# Phase 2.2B.exec — Pre-Execution Check

**Status:** PASS for B-MECH-1 and B-CERT-1 execution. **PARTIAL FAIL** for B-MECH-2, B-MECH-3, B-MECH-4 because their driver `main()` loops are scaffolds, not full training+evaluation implementations.

## 1. Branch and commit

| Item | Value |
|---|---|
| Branch | `exp/elara-phase2-mechanism-and-replication` |
| Pre-execution commit | `2719d8111405a4fcc75e288678cd5a18d37134c5` |
| Commit message | `Implement ELARA Phase 2.2B Family-B execution infrastructure` |

## 2. SHA256 anchors

| Path | SHA256 |
|---|---|
| `configs/phase2/rga_v2_gate_contract.yaml` | `b2f59eaa3b5eda90d33740d0a7df3451fdffd28fcfc88db3c04d608b608d06f0` |
| `docs/research/phase2/PHASE_2_EXPERIMENT_REGISTRY_v2.csv` | `24d027c7485a082f139f7a6b391cf0de73232bf7e0a4ff86ea9c272142c1c192` |
| `src/scripts/run_phase2_mechanism_replication.py` | `e4744070d336daec030ec34c679ed33c0d5ca7f91e10123895c4fa2e59e1c663` |
| `src/scripts/run_phase2_rga_v2_gate_sweep.py` | `b90cf87f83be8e51fed80710624735686eedce089eac0201ae7157cabfae535e` |
| `src/scripts/run_phase2_mixture_shift.py` | `628b2d5f6b34541e3856ed4256f6deffa49f31327741caae68466a96c5e381c6` |
| `src/scripts/run_phase2_ks_power_sweep.py` | `9b430562567f2c325fefc01db27c88dd82481cb476456669a23a0bfe7f9d7500` |
| `src/scripts/run_phase2_certificate_audit.py` | `531b1f4949263ced9055e52dea556a67dd71bf75065d7e36f1a00325c095617f` |

## 3. Test suite state

`PYTHONPATH=src .venv/bin/python -m pytest tests/ --no-header --tb=no -p no:warnings`:

```
535 passed, 11 skipped
```

Meets requirement (≥ 535 / 11).

## 4. Family-D protection state

- `docs/research/phase2/FAMILY_D_V1_INVALIDATION_NOTICE.md` — contains `INVALID_FOR_EXECUTION` ✓
- `docs/research/phase2/FAMILY_D_V2_DESIGN_STATUS.md` — contains `V2_DESIGN_PENDING` ✓
- No Family-D code path is called by any Phase 2.2B.exec driver.

## 5. Driver scaffold disclosure (CRITICAL)

The Phase 2.2B infrastructure-completion report claimed `READY FOR FULL FAMILY-B COMPUTE`. That claim was **overstated**. Honest audit of the five driver `main()` functions:

| Driver | `main()` state | What happens at execution time |
|---|---|---|
| `run_phase2_mechanism_replication.py` (B-MECH-1) | **FULL TRAINING LOOP** | Trains model, injects k=4 corruption, runs static/RGA predictions, archives per-sample predictions. Produces real B1/B2 replication evidence. |
| `run_phase2_rga_v2_gate_sweep.py` (B-MECH-2) | **SCAFFOLD** | Validates inputs, prints "implementation hook is ready; reserved for a future compute window", exits 0. **Does not produce result rows.** |
| `run_phase2_mixture_shift.py` (B-MECH-3) | **SCAFFOLD** | Same pattern. Mixture-shift sampler is implemented + tested, but driver does not invoke the model training loop. |
| `run_phase2_ks_power_sweep.py` (B-MECH-4) | **SCAFFOLD** | Same pattern. KS window-size parameter is wired into `ReliabilityEstimator` but driver does not invoke the sweep loop. |
| `run_phase2_certificate_audit.py` (B-CERT-1) | **SCAFFOLD (pairing logic not implemented)** | Scans archive directory and prints inventory. Does not yet pair clean/degraded archives or compute certificates. |

This pre-check is therefore a transparent correction: only **B-MECH-1** will produce real results in this Phase 2.2B.exec task. **B-CERT-1** can be brought to executable state with a small follow-up implementation patch (pairing B-MECH-1 archives by attack × k); the pairing patch is in scope for this task. **B-MECH-2, B-MECH-3, B-MECH-4** remain scaffolds and will be reported as `EXECUTION_BLOCKED_DRIVER_SCAFFOLD` with the specific gap documented.

The honest expected final decision is **`PASS FOR MECHANISM REPLICATION ONLY`** (if B-MECH-1 reproduces B1/B2 and B-CERT-1 produces valid certificates) **or** **`FAIL`** (if either fails).

## 6. Gate decision

- B-MECH-1 execution: **PROCEED**.
- B-CERT-1: **IMPLEMENT PAIRING LOOP, THEN PROCEED**.
- B-MECH-2 / B-MECH-3 / B-MECH-4: **RUN SCAFFOLDED DRIVER FOR THE RECORD; DO NOT CLAIM RESULTS.**

This pre-check itself blocks any "PASS FOR RGA-v2 METHOD ADVANCEMENT" decision in advance — that decision requires a fully executed B-MECH-2 plus all C1..C6, neither of which exist.
