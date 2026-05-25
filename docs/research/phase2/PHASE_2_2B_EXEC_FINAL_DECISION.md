# Phase 2.2B.exec — Final Decision

## Decision: **`PASS FOR MECHANISM REPLICATION ONLY`**

B1 and B2 primary mechanism endpoints reproduce under the v2 Phase-2 archived-prediction pipeline. RGA-v2 / theory closure are **not** achieved because the B-MECH-2 / B-MECH-3 / B-MECH-4 driver `main()` functions are scaffolds — they validate inputs, document the remaining loop-implementation work, and exit without producing result rows. This was disclosed up front in [PHASE_2_2B_EXECUTION_PRECHECK.md](./PHASE_2_2B_EXECUTION_PRECHECK.md) §5.

## Why not a higher decision

- **PASS FOR THEORY-CLOSURE WITH NO RGA-v2 PROMOTION** requires KS analysis + certificate analysis to have **final outcomes** on the full partial-failure surface. B-CERT-1 produced a partial result (k=4 only; risk-dominance terms inadmissible). B-MECH-3 / B-MECH-4 produced no result. Theory closure is therefore not achieved.
- **PASS FOR RGA-v2 METHOD ADVANCEMENT** requires C1..C6 to pass. B-MECH-2 didn't execute, so C1..C6 cannot be evaluated. No promotion.
- **READY TO RETURN TO FAMILY-D v2 DESIGN** requires all five Family-B cells to have final outcomes. Three (B-MECH-2/3/4) don't.

## Why not a lower decision

- **FAIL** would imply a validity violation. None occurred: every archived row carries `selection_used_test_metrics=False`; no Family-D file was touched; no overlooked archive issues; full test suite green at 535 / 11; B1/B2 reproduced with positive Δ + Holm-significant p + CI excluding zero.

## Final-decision evidence summary

| Cell | Status | Evidence |
|---|---|---|
| B-MECH-1 | **REPRODUCED (both endpoints)** | Δ B1=+0.0507 CI [+0.036, +0.065]; Δ B2=+0.0939 CI [+0.074, +0.115]; Holm K=2 p both < 0.001 |
| B-MECH-2 | **EXECUTION_BLOCKED_DRIVER_SCAFFOLD** | driver main() is a scaffold; documented |
| B-MECH-3S | **EXECUTION_BLOCKED_DRIVER_SCAFFOLD** | protocol locked (committed at 204775b); driver main() is a scaffold |
| B-MECH-4 | **EXECUTION_BLOCKED_DRIVER_SCAFFOLD** | driver main() is a scaffold |
| B-CERT-1 | **PARTIAL** | max_attack k=4 CERTIFIED (LCB=+0.0085); zero_attack k=4 NOT_CERTIFIED (LCB=-0.0050); risk-dominance terms inadmissible without paired clean arm |

## What this decision unlocks

- The manuscript may say: "Under the Phase-2 v2 archived-prediction pipeline, the B1/B2 mechanism endpoints from Phase-1 reproduce (B1 Δ AUC = +0.0507, B2 Δ AUC = +0.0939; both Holm K=2 p < 0.001)."
- The manuscript may say: "A retrospective fired-subset paired-loss certificate is positive for max_attack k=4 (LCB = +0.0085) under the defined stress protocol; it is negative for zero_attack k=4 (LCB = -0.0050). The fired-subset paired-loss certificate measures local per-sample benefit and is not equivalent to the global ROC-AUC improvement."

## What this decision does NOT unlock

- No RGA-v2 promotion claim.
- No partial-failure (k=1,2,3) improvement claim.
- No KS power / mixture-shift claim.
- No theory-closure claim.
- No Family-D activation.
- No paper / thesis edits.
- No Phase 3 / ELARA-Universal / ORIUS work.

## Stop boundary honoured

- No Family-D file modified.
- No paper / thesis edit.
- No Phase 3 / ELARA-Universal / ORIUS work.
- No Phase 2.2A Family-A regression (CSVs untouched; tests green).
- No gate tuning on test outcomes.
- No B1/B2 endpoint redefinition (Phase-1 targets reported as-is alongside Phase-2 estimates).
- No RGA-v2 promotion without C1..C6.

## Test suite state

- 535 passed / 11 skipped at end of Phase 2.2B.exec (unchanged from start; no Family-B test regression).

## Pre-execution commit anchor

`2719d8111405a4fcc75e288678cd5a18d37134c5` (Phase 2.2B infrastructure) + `204775b...` (Phase 2.2B.exec protocol lock).
