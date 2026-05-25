# Phase 2 Family-B — Final Decision

## Decision: **`FAMILY_B_COMPLETE_WITH_NEGATIVE_RGA_V2_AND_BOUNDED_THEORY_EVIDENCE`**

Family-B has produced complete, validity-clean evidence across all five cells. The RGA-v2 promotion hypothesis is **rejected** by negative C1 results; the B-MECH-3S and B-MECH-4 results are bounded exploratory / locked-grid results. B-MECH-1 reproduces the B1 endpoint cleanly and the B2 endpoint under an estimator change. B-CERT-1 yields one positive and one negative certificate plus full risk-dominance terms.

## Why not a higher decision

- **`FAMILY_B_COMPLETE_WITH_RGA_V2_METHOD_ADVANCEMENT`** requires at least one candidate to pass all C1..C6. None did. (G1/G2/G3 all fail C1 at clean false-fire = 1.000.)

## Why not a lower decision

- **`FAMILY_B_COMPLETE_MECHANISM_REPLICATION_ONLY`** would understate the work: B-MECH-2/3S/4 + B-CERT-1 v2 (with risk-dominance) all produced final, bounded outcomes — even though some outcomes are negative.
- **`FAIL_FAMILY_B_VALIDITY`** would require an actual validity / archive violation. None occurred.

## Evidence summary

| Cell | Status | Final result |
|---|---|---|
| B-MECH-1 (B1, B2) | EXECUTED + INFERRED | B1 `VERIFIED_REPRODUCED`; B2 `COMPARABLE_BUT_ESTIMATOR_CHANGED_POSITIVE_RESULT` |
| B-MECH-2 (RGA-v2 sweep) | EXECUTED at 15 seeds (contract minimum) | G0 `BASELINE_REFERENCE`; G1/G2/G3 `NOT_IMPROVED` (C1 fail) |
| B-MECH-3S (domain composition) | EXECUTED 5 seeds × 10 mixtures | `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED` |
| B-MECH-4 (KS window sweep) | EXECUTED 5 seeds × 5 windows × 3 degradations | `TRADEOFF_IMPROVED` on locked grid |
| B-CERT-1 (clean-arm + degraded) | EXECUTED with v2 risk-dominance | max_attack k=4 `CERTIFIED`; zero_attack k=4 `NOT_CERTIFIED`; (q₀, q₁, Δ₀, Δ₁, π*) admissible |

## What this decision unlocks

A manuscript-update phase may now incorporate the seven manuscript-permitted claims listed in the hostile-review report §Q14.

## What this decision does NOT unlock

- Family-D execution.
- Paper / thesis edits.
- Phase 3 / ELARA-Universal / ORIUS.
- Any RGA-v2 promotion claim.
- Any extrapolation beyond the locked KS window grid + degradation types in B-MECH-4.
- Any general category/cohort-mixture theorem closure claim.

## Provenance

- Pre-execution commit: `dbf8dca` (Phase 2.2B.1).
- Clean-arm + B-CERT-1 v2 run in this phase; commit hash recorded in the final artifact manifest after commit lands.
- All test invariants pass (refer to terminal output).
