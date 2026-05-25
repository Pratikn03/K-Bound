# Phase 2.2B.exec — Remaining Open Gaps

Each gap is named, sized, and tied to the test that prevents silent regression.

## G1 — B-MECH-2 driver `main()` is a scaffold

- **State:** the driver validates inputs and refuses unregistered IDs / G4 gate, but does not invoke the per-seed training + gate-sweep loop.
- **Closing this gap entitles:** an RGA-v2 promotion / non-promotion decision under the locked C1..C6 criteria.
- **Effort:** ~150 lines of training-loop code wrapping `_select_tau_on_validation_only()`; plus wall-clock ~6–12 hours for 30 seeds × 4 gates × full fault surface.
- **Test guard:** the scaffold message is preserved; once implemented, `tests/test_phase2_family_b_prediction_archive_complete` will assert archive presence.

## G2 — B-MECH-3 driver `main()` is a scaffold

- **State:** mixture-shift sampler implemented + tested; driver does not invoke model training × mixture iteration.
- **Closing this gap entitles:** a `DOMAIN_COMPOSITION_FALSE_FIRE_REDUCED` / `NOT_REDUCED` / `INCONCLUSIVE` decision (the protocol is locked in [MIXTURE_SHIFT_PROTOCOL.md](./MIXTURE_SHIFT_PROTOCOL.md)).
- **Effort:** ~100 lines; wall-clock ~100–300 minutes.

## G3 — B-MECH-4 driver `main()` is a scaffold

- **State:** `ks_window_size` parameter wired into estimator; driver does not invoke the window-size sweep.
- **Closing this gap entitles:** a `TRADEOFF_IMPROVED` / `FALSE_FIRE_REDUCED_POWER_LOST` / `NO_MEANINGFUL_CHANGE` / `INCONCLUSIVE` decision.
- **Effort:** ~80 lines; wall-clock ~125–375 minutes.

## G4 — B-MECH-1 driver writes `gate_fired = False` for all samples

- **State:** the driver consumes the runner's batch-level `adapted` flag, which under-reports per-sample firing. The empirical AUC delta confirms RGA did effectively fire, but the per-sample column is uniformly False.
- **Workaround:** B-CERT-1 defines the fired subset empirically as samples where the seed-averaged RGA prediction differs from static by > 1e-6. This recovers a meaningful certificate input.
- **Closing this gap entitles:** B-CERT-1 to use the contract-spec definition of fired (per-sample `gate_fired` written by the model code path), which matches the formal mechanism statement.
- **Effort:** ~10 lines in the B-MECH-1 driver; re-archive run on existing trained models is unnecessary if the existing predictions are kept.

## G5 — B-CERT-1 risk-dominance terms (q₀, q₁, Δ₀, Δ₁, π*) are inadmissible

- **State:** the B-MECH-1 archive captures only the degraded (k=4) arm per scenario; the clean (k=0) arm was not archived. Risk-dominance terms require both arms.
- **Closing this gap entitles:** the full (q₀, q₁, Δ₀, Δ₁, π*) table per scenario.
- **Effort:** add a clean-arm archive step in B-MECH-1; ~20 lines. Wall-clock: re-run B-MECH-1 (~6 min) or amend a new "k=0 baseline" cell.

## G6 — B-CERT-1 produced a split result (zero_attack NOT_CERTIFIED)

- **State:** AUC delta is positive on both scenarios; per-sample paired-loss certificate is positive only on max_attack. This is a real distinction, not a bug.
- **Closing this gap entitles:** nothing extra — the split is reported honestly. A future investigation could ask **why** zero_attack improves AUC while not improving per-sample loss (likely answer: rank-only improvement on borderline samples without absolute-loss reduction).
- **Effort:** investigation; not a code fix.

## G7 — Phase-2 B2 estimate (+0.0939) sits above the Phase-1 target CI (+0.0319 [0.0050, 0.0617])

- **State:** both numbers are reported as-is. Neither is retroactively redefined.
- **Closing this gap entitles:** nothing — the manuscript records both observations side by side. A future audit could investigate whether the discrepancy is a seed-count effect (5 vs 30), a corruption-injection PRNG offset effect, or a third explanation. None of those investigations is in scope for Phase 2.2B.exec.

## G8 — Family-D v2 design still pending

- **State:** unchanged from Phase 2.1 / 2.2A. Family-D v2 design pending; v1 invalid for execution.
- **Phase 2.2B.exec does not touch this** (verified by [tests/test_phase2_family_d_untouched_during_family_b.py](../../../tests/test_phase2_family_d_untouched_during_family_b.py)).
