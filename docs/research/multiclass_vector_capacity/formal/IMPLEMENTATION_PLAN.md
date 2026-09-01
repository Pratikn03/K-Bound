# Multiclass Vector-Capacity Finite Foundation Implementation Plan

> **For agentic workers:** Execute this approved, isolated slice task-by-task. The parent owns the research specification, oracle, and theorem ledger; this subtree owns only the Lean foundation and its verification.

**Goal:** Kernel-check the finite cost-benefit identity and sound realized-fiber decisions, including a nonpoint-identified but uniformly positive benefit example.

**Architecture:** A standalone `MulticlassVectorCapacity` library has finite model definitions, the algebraic benefit identity, linear observable fibers, realized-fiber decision results, and exact witnesses. The audit imports only actual proofs and prints their axioms. Unproved T2/T3/T5 generalizations are not declared as theorems.

**Tech Stack:** Lean 4.29.1; Mathlib commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6`.

**Spec:** `/Users/pratik_n/.codex/attachments/713b862e-738e-4035-bd3a-536839cc7f00/pasted-text.txt`, read in full before implementation.

## Global Constraints

- Modify only this new formal subtree and scoped temporary diagnostics.
- No changes to existing K-Bound formal files, manuscript, empirical authorities, or release files.
- Use finite strata, classes, candidates; normalized nonnegative probabilities; costs in `[0,1]`.
- Positive benefit is frozen expected cost minus candidate expected cost.
- Empty fibers are protocol failure, never a certificate.
- No proof gaps, custom axioms, target-benefit-oracle assumptions, or unsupported claim promotion.
- Reuse resident pinned dependency artifacts without changing their source or outputs; no downloads.
- A fresh-checkout build remains `NOT_RUN` unless it is actually performed.

## Task 1: Pinned project and finite identity

Files: `lakefile.lean`, `lean-toolchain`, `lake-manifest.json`, `MulticlassVectorCapacity/Basic.lean`, `Benefit.lean`, `Regression.lean`.

- [ ] Write a Lean regression that demands `expectedCost frozen - expectedCost candidate = benefit`, and confirm it fails while the module is absent.
- [ ] Define `Model.q`, its probability proofs, bounded `Model.cost`, and normalized `ConditionalLabels.prob`.
- [ ] Define finite expected cost and cost contrast; prove the exact sum identity by finite distributivity.
- [ ] Prove expected-cost and benefit bounds using the declared probability and cost assumptions.
- [ ] Recompile the regression through the pinned kernel.

## Task 2: Realized fibers and strict decisions

Files: `ObservableFiber.lean`, `SignCapacity.lean`, `Examples.lean`, `Regression.lean`.

- [ ] Define a finite linear observable map and its equality fiber over `ConditionalLabels`.
- [ ] Define `PointIdentified`, explicitly nonempty `StrictAdapt`/`StrictFreeze`, and positive uniform margin predicates.
- [ ] Prove that a positive lower bound implies candidate cost is lower for every feasible world; prove the negative upper-bound dual.
- [ ] Prove the equivalent strict-decision tests when actual minimum/maximum witnesses are provided, without asserting those extrema exist by assumption on a headline theorem.
- [ ] Construct the exact three-class fiber `eta₀=1/5`, with frozen costs `(0,1/2,1)` and candidate costs `(1,0,0)`.
- [ ] Prove its benefit interval is `[1/5,3/5]`, endpoints are feasible, and the same fiber is not point identified but is strictly positive.
- [ ] Prove the null variation `(0,1,-1)` changes the benefit while no negative world exists in this fiber; this refutes the zero-crossing inference in T5 as stated, not every qualified impossibility theorem.

## Task 3: Audited build

Files: `Audit.lean`, `MulticlassVectorCapacity.lean`, and local build receipts.

- [ ] Build all maintained modules using Lean 4.29.1 and the pinned resident Mathlib artifacts.
- [ ] Print axioms for every proof marked verified; allow only `propext`, `Classical.choice`, and `Quot.sound`.
- [ ] Run exact Lean regression examples and deliberately false orientation/strict-boundary probes in temporary files; require compiler rejection.
- [ ] Record source, toolchain, dependency, output, and axiom-report hashes, with clean-checkout portability `NOT_RUN`.
- [ ] Report exact declaration names, assumptions, verification commands, and excluded claims to the parent. Do not commit or promote this slice into K-Bound.
