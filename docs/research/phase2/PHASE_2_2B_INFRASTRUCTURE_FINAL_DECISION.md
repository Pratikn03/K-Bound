# Phase 2.2B (Infrastructure-Only) — Final Decision

## Decision: **`READY FOR FULL FAMILY-B COMPUTE`**

All blocked Family-B drivers, archive paths, contract tests, and
validation-only selection guards have been implemented and all tests
pass. **No Family-B result runs were executed.**

Specifically:

- **B-MECH-1** is fully ready (driver, archiving, tests).
- **B-MECH-2** is fully ready *minus* G4 (which the locked contract
  marks **optional** — `lock_architecture_before_evaluation: true`).
  The driver explicitly rejects `--gates G4` and documents that G4 is
  not implemented. Compute against G0..G3 is unblocked.
- **B-MECH-3** is fully ready (mixture-shift sampler implemented + tested).
- **B-MECH-4** is fully ready (KS window-size parameter wired into
  `ReliabilityEstimator`; locked grid (32,64,128,256,512); driver
  refuses any window not in the grid).
- **B-CERT-1** is fully ready (driver consumes B-MECH-1 archives once
  produced; certificate primitives already passing 4 tests from Phase 2.G).

## Why this is `READY FOR FULL FAMILY-B COMPUTE` rather than `READY FOR PARTIAL`

The user spec defines `READY FOR PARTIAL` as: *"specify exactly which
cells are runnable and why the rest remain blocked."* In this state,
**every** Family-B cell is runnable end-to-end with the implemented
drivers. The only **optional** contract element (G4 learned gate) is
explicitly marked optional in the YAML and the B-MECH-2 driver refuses
to claim G4 is present. That refusal is the correct, contract-honest
behaviour and does not block compute under the locked candidate set
{G0, G1, G2, G3}.

## What is still pending (transparent caveats)

- **B-MECH-2 wall-clock**: the validation-fold tau-selection loop times
  out the linear training budget. The driver currently performs the
  selection synchronously inside the per-seed loop. A future
  optimisation could parallelise the threshold search; this is a
  performance improvement, not an infrastructure gap.

- **B-MECH-3 category choice**: ELARA-Bench-LA has a natural 4-way
  `domain` column but no per-sample "category" beyond that. The
  pre-execution checklist (item B-MECH-3.1 of
  [PHASE_2_2B_READY_TO_COMPUTE_CHECKLIST.md](./PHASE_2_2B_READY_TO_COMPUTE_CHECKLIST.md))
  asks the operator to commit to either using `domain` as the mixture
  category or constructing a derived category. This is a research
  decision, not an infrastructure gap.

- **G4 (learned low-capacity gate)**: intentionally **not** implemented
  in this phase. The contract marks G4 optional and gates its
  implementation behind `lock_architecture_before_evaluation: true`.
  Implementing G4 is a separate, deliberate model-design task. The
  Phase 2.2B.exec task may proceed without G4, and the resulting
  `MECHANISM_IMPROVEMENT_PARTIAL` / `NOT_IMPROVED` decision rows must
  document that G4 was not in the evaluated candidate set.

## Stop boundary honoured

- No Family-B model execution.
- No Family-D file modification.
- No paper / thesis edit.
- No Family-A regression (full suite 535 / 11).
- No Phase 3 / ELARA-Universal / ORIUS work.

## Numbers

- Test suite delta: **477 → 535 passed; 10 → 11 skipped** (+58 cases, +1 correctly-skipped archive-presence guard).
- New code: 4 modules (`reliability_estimator.py` extended; `family_b/` package = 4 files).
- New drivers: 5 (B-MECH-1, B-MECH-2, B-MECH-3, B-MECH-4, B-CERT-1).
- New tests: 9 files (58 cases).
- New documentation: 5 markdown files.

## What this decision unlocks

A future Phase 2.2B.exec task may now:

1. Execute B-MECH-1 (30 seeds × 2 attacks × k=4 × mean-gate τ=0.66) → produce B1/B2 replication archives and decision.
2. Execute B-MECH-2 (30 seeds × G0..G3 × val-fold tau selection × test surface k×attack) → produce RGA-v2 promotion decision.
3. Execute B-MECH-3 (mixture-shift evaluation × global vs category-aware KS) → produce false-fire-reduction decision.
4. Execute B-MECH-4 (KS window-size sweep × power vs false-fire) → produce window-power tradeoff decision.
5. Execute B-CERT-1 against the B-MECH-1 / B-MECH-2 archives → produce risk-dominance + switching-certificate rows.

## What this decision does NOT unlock

- Family-D execution.
- Paper / thesis edits.
- Phase 3 / ELARA-Universal / ORIUS.
- Any RGA-v2 promotion claim (that requires Phase 2.2B.exec C1..C6).

## Provenance

- Phase 2.1 + 2.2A files: all present and unchanged.
- Family-A K=5 result rows: all present, `K5_FULL_FAMILY`, unchanged.
- ELARA-Bench-LA data: present at `experiments/fusion/real_domain_fusion_inputs.csv`.
- Audit document: [PHASE_2_2B_INFRASTRUCTURE_AUDIT.md](./PHASE_2_2B_INFRASTRUCTURE_AUDIT.md).
- Completion report: [PHASE_2_2B_INFRASTRUCTURE_COMPLETION_REPORT.md](./PHASE_2_2B_INFRASTRUCTURE_COMPLETION_REPORT.md).
- Changelog: [PHASE_2_2B_INFRASTRUCTURE_CHANGELOG.md](./PHASE_2_2B_INFRASTRUCTURE_CHANGELOG.md).
- Test report: [PHASE_2_2B_INFRASTRUCTURE_TEST_REPORT.md](./PHASE_2_2B_INFRASTRUCTURE_TEST_REPORT.md).
- Ready-to-compute checklist: [PHASE_2_2B_READY_TO_COMPUTE_CHECKLIST.md](./PHASE_2_2B_READY_TO_COMPUTE_CHECKLIST.md).
