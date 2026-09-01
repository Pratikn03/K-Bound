# Independent formal-scope review

Date: 2026-08-31. Read-only semantic review of the six requested Lean files, followed by a documentation-only recheck; the current versions contain 535 lines. No Lake invocation, compiler claim, proof edits, ledger changes, or experiments were performed. Compilation and imported-axiom verification remain the separate build receipt's responsibility.

## Verdict

No blocking orientation, non-vacuity, or conditionally complete-order misuse was identified in the reviewed statements. The code supports finite-stratum population-cost identities, scalar fixed-candidate strict-decision soundness, an exact nontrivial three-class counterexample, and the stated edge cases. It does not establish the whole proposed T1–T9 program.

Follow-up: item 1 is resolved by the clarified ObservableFiber header. Replacing only that comment in a read-only stream reproduces the previously reviewed hash; the other five source files are byte-identical. All six current source fingerprints below match the source hashes in the final [theorem inventory](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/multiclass_vector_capacity/formal/verification/theorem_inventory.json). The no-general-attainment, scalar fixed-candidate, and noncomputable limits remain unchanged.

## Actionable scope findings

1. **RESOLVED — finite-domain documentation clarified.** [ObservableFiber.lean:9](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/multiclass_vector_capacity/formal/MulticlassVectorCapacity/ObservableFiber.lean:9) now correctly describes linear moments over finite strata/classes, indexed by R, with finite rows as a specialization. S and Y are finite in the results; R and J remain unrestricted. This documentation-only correction does not add finite row counts, rank/supplement minimality, or finite-candidate multiplicity control to the proved statements.
2. **T4 is fixed-candidate soundness and uniform-margin equivalence, not a fully formalized sharp decision frontier.** [SignCapacity.lean:54](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/multiclass_vector_capacity/formal/MulticlassVectorCapacity/SignCapacity.lean:54) proves infimum/supremum sign equivalence to a positive uniform margin. The action equivalences unfold the defined rule. The general converse from pointwise strict superiority in every feasible world to a certificate is not proved here; neither are general compactness/attainment, convexity, or a zero-world characterization of abstention. Such consequences require additional lemmas. The exact interval theorem currently concerns the concrete example only. The source header already labels these results conservatively; preserve that scope in external claims.
3. **No vector selection, statistical certification, or executable solver.** [ObservableFiber.lean:24](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/multiclass_vector_capacity/formal/MulticlassVectorCapacity/ObservableFiber.lean:24) defines a scalar image for one j. [SignCapacity.lean:29](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/multiclass_vector_capacity/formal/MulticlassVectorCapacity/SignCapacity.lean:29) is noncomputable and uses classical real comparisons and feasible-set existence. It is a mathematical decision function, not an implemented LP algorithm. There is no candidate-versus-candidate winner certificate, confidence event, query process, delay model, sample-complexity bound, or randomized two-world error lower bound in these files. T2/T3 rank claims and T6–T9 remain outside this artifact.
4. **Availability and provenance remain external assumptions.** A and b may encode arbitrary moments; no theorem prevents an answer-oracle row or proves a row observable without target labels. Costs are supplied policy-indexed columns rather than a separately modeled common action-cost function composed with frozen predictors. These abstractions are sufficient for the stated algebra, but operational claims require those bridges. [Examples.lean:7](/Users/pratik_n/Documents/AutoML_Flagship_V8/docs/research/multiclass_vector_capacity/formal/MulticlassVectorCapacity/Examples.lean:7) correctly discloses eta_0 = 1/5 as hypothetical structural knowledge, not an unlabeled inference. Preserve that qualification.

## Checks supporting the narrow verdict

- **Orientation:** frozen policy is none; candidate j is some j. Benefit is frozen-minus-candidate cost. Benefit.lean:22,50,55 and SignCapacity.lean:115,125 consistently turn positive benefit into lower candidate cost and negative benefit into lower frozen cost.
- **Probability constraints:** nonnegative normalized stratum masses and conditional rows, with costs in [0,1], imply expected costs in [0,1] and benefits in [-1,1]. Empty S cannot admit Model; a Model together with ConditionalLabels cannot have empty Y. The generic theorems do not require three labels, but validly include that setting; the witness explicitly uses Fin 3. Zero-mass strata are permitted, with no active-query attainability claim.
- **Infimum/supremum:** ObservableFiber.lean:44,50 supplies genuine lower/upper bounds. SignCapacity.lean:38,43 uses a feasible member, which itself witnesses nonemptiness. The reverse uniform-margin directions explicitly assume nonempty fibers. The lower-versus-upper comparison also requires nonemptiness. The controller checks nonemptiness before branching; pointIdentified includes it, so an inconsistent fiber is not vacuously certified.
- **Strictness and edge cases:** zero-cost equality yields some abstain; an inconsistent zero operator with b=1 yields none. Negative and unnormalized conditional coordinates are rejected. No <=0 or >=0 commitment replaces the strict tests.
- **Exact T5 witness:** costs (0,1/2,1) and (1,0,0) give contrast (-1,1/2,1). The feasible family is (1/5,t,4/5-t), 0<=t<=4/5, and benefit is 3/5-t/2. Examples.lean:78 proves the full attained interval [1/5,3/5], not just outer bounds. Distinct feasible endpoints prove non-point-identification; null variation (0,1,-1) preserves normalization and observables while changing benefit by -1/2. The nonempty fiber remains strictly positive, and the candidate is worse on class 0. This genuinely refutes the unqualified T5 implication, without asserting an opposite-sign world or a statistical lower bound.

## Reviewed source fingerprints

SHA-256 of the read-back source versions:

| File | SHA-256 |
| --- | --- |
| Basic.lean | 505733f578cc636148d729cba15e0baf639ba704633a02eb5bf75506d588a8f6 |
| Benefit.lean | f8676365c237cd71558a0acfe62173d8fdda002db1e4160a2955eb464cb25b81 |
| ObservableFiber.lean | f359a93279c9608316ed52760e56650194478871f2312e2ec410f7073f48f5fe |
| SignCapacity.lean | ab3a71f5e1b3e64049122c139c500c448fafdee4adb6b31ee0615275fa341a01 |
| Examples.lean | 2539ef4d246d6b4a0199f4de7bd99879379cf687e6f48c4aab466d74db7f13d5 |
| EdgeCases.lean | 7749f9c4a162e2ec25d51626116664d620a2cff4b45182a23282e252e185fa89 |
