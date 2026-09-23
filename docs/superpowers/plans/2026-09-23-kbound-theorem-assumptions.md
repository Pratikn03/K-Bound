# K-Bound theorem and assumption audit implementation plan

**Goal:** Improve the coverage and precision of the maintained mathematical claims, and prevent diagnostic checks from being mistaken for evidence that statistical assumptions hold.

**Architecture:** Start from frozen artifact commit `69548cf44bcbede86053ea3ef6572ef57b2a1615` on a separate revision branch. Inventory the active manuscript, audit its arguments independently, add a missing formal probability result, and correct any assumption-reporting defects. Source bindings check correspondence and freshness, not mathematical truth.

**Tech Stack:** LaTeX, pinned Lean 4/Mathlib, Python, pytest, JSON.

**Spec:** The user's request in this task: work on correctness of every maintained theorem and validity of statistical assumptions. This does not authorize asserting universal validity, reopening stopped targets, or replacing the frozen publication.

## Global constraints

- Preserve tag `kbound-frozen-2026-09-23`, its assets, and historical receipts.
- Work only in the new `codex/kbound-theorem-assumptions-20260923` revision.
- Keep the refuted historical orbit-only extension excluded.
- Separate full mechanization, component-level support, and reviewed LaTeX arguments.
- Keep assumptions external where data/design cannot establish them; never promote a diagnostic non-rejection into a coverage guarantee.
- No new target scoring, retraining, paid compute, or camera deployment.

## Review focus

- Random and infinite radius: infinity and zero-touching intervals must abstain.
- Named result missing from the active-source register: fail the correspondence check.
- Source or proof changes after review: fail stale byte bindings.
- Diagnostic fails to detect concept shift: coverage must remain unresolved.
- Dependent cells, candidate selection, and repeated decisions: do not inherit unsupported population, conditional, or anytime guarantees.

## Tasks

- [x] **1. Inventory and review.** Enumerate every theorem, lemma, proposition, and corollary in both maintained driver include closures. Write `docs/research/kbound/theorem_assumption_revision/theorem_register.json`, `theorem_review.md`, `assumption_register.json`, and `statistical_review.md`. Record exact scope and actionable findings.
- [x] **2. Close the random-radius formal gap.** Add an extended-nonnegative random-radius certificate to the pinned Lean project, including both directional errors and their union. Compile, inspect transitive axioms, and add only successfully checked declarations to the registry. Update manuscript correspondence and current count while retaining the historical 150-name receipt.
- [x] **3. Correct assumption reporting.** Reproduce and fix the legacy promotion from `not_falsified` to `guarantee applies`. Add regressions showing a quiet diagnostic and matched evidence cannot authorize adaptation or coverage. Preserve historical output and document the correction.
- [x] **4. Bind the review to active source.** Add `docs/research/kbound/scripts/check_theorem_assumption_revision.py` and `tests/test_theorem_assumption_revision.py`. Test missing, duplicate, stale, and unresolved input handling. Check exact statement inventory and references; label this a correspondence check, not a theorem verifier.
- [x] **5. Integrate and verify.** Clarify sample-count and fixed-sampling premises, run relevant Python checks and a fresh strict Lean audit, record exclusions, and compile the affected manuscript into a new revision output directory. Review the final changes independently and write a report with explicit remaining proof and empirical obligations.

## Execution decisions

- The user already requested execution. Planning is recorded here and work proceeds without a redundant approval step.
- The native worktree tool could not resolve the frozen commit in the task's different checkout. An explicit `git worktree add` from the authoritative repository created this revision. Existing Git pack warnings are recorded; the checkout completed successfully. No object store repairs were attempted.
- Existing pinned dependencies are reused via a copy-on-write copy of the Lean cache. No dependency updates or experiment execution are needed.
- Baseline and final verification results are recorded in the revision report. A successful correspondence check or Lean build will not be represented as proof that empirical assumptions hold.

## Completion record

The bounded revision is complete. See `docs/research/kbound/theorem_assumption_revision/verification.json`: 639 targeted tests passed, 162 strict Lean declarations passed, source correspondence passed for 13 named results and 20 assumptions, and the separate 69-page compact PDF was compiled and visually reviewed. Full mechanization and empirical assumption validity remain explicitly scoped in the revision README; the frozen release was preserved.
