# Isolated multiclass vector-capacity finite foundation

This is a separate Lean project. It does not modify or promote claims into K-Bound.

The initial verified slice consists of T1, fixed-candidate realized-fiber guard
soundness, exact interval witnesses, and boundary counterexamples. It is not the
complete T1–T5 foundation or the T1–T9 research program.

## Mathematical scope

`Model` records nonnegative stratum masses summing to one and known costs in
`[0,1]`. `ConditionalLabels` records one nonnegative, normalized label vector per
stratum. `none` is the frozen policy and `some j` is candidate `j`.

The finite-sum theorems require finite strata and finite classes. Candidate indices
and the index of equality restrictions need not themselves be finite for these
pointwise results. In particular, the specified finite multiclass setting is a
specialization, not an extra assumption needed by the algebraic identity.

`cost_benefit_identity` proves exactly:

```
expectedCost frozen − expectedCost candidate = benefit.
```

`fiberDecision` is a **noncomputable mathematical guard for one fixed candidate**.
It uses the actual infimum and supremum of the feasible benefit image. Empty
fibers return `none` (protocol failure); otherwise positive infimum means ADAPT,
negative supremum means FREEZE, and the remaining cases mean ABSTAIN. It is not
an implemented LP solver, multi-candidate selection policy, or T9 statistical
false-commit guarantee.

The exact three-class example uses the declared moment `eta[0] = 1/5`, frozen
costs `(0, 1/2, 1)`, and candidate costs `(1, 0, 0)`. Lean proves that its entire
identified benefit set is `[1/5, 3/5]`. Point identification fails, but strict
ADAPT is sound. The candidate is worse on class 0, so the example does not rely
on pointwise candidate dominance. The observable/simplex-null vector `(0,1,-1)`
changes benefit by `−1/2` while every feasible benefit remains positive. This
refutes the inference from a surviving null contrast to opposite strict signs
without an additional zero-crossing assumption. The moment is hypothetical
structural knowledge, not claimed to be learned from unlabeled data.

## Verification

The project pins Lean 4.29.1 and Mathlib commit
`5e932f97dd25535344f80f9dd8da3aab83df0fe6`.

With those dependencies already resident:

```sh
lake --no-cache --wfail build
python3 -B -m unittest discover -s tests -v
python3 -B export_inventory.py
```

The exporter refuses absent or revision-mismatched dependencies; it does not
download them. A local `.lake/packages` link may point at an existing dependency
cache, while this project's build/configuration outputs stay in its own `.lake`.
The successful narrow-import build does not require modifying the old K-Bound
cache. The output inventory records exact printed types, transitive axioms,
source and `.olean` hashes, and hashed build/axiom logs. `Audit.lean` also inspects
every compiled declaration in the namespace, rejecting custom axioms, unsafe or
partial declarations, and transitive axioms outside `propext`, `Classical.choice`,
and `Quot.sound`.

Clean-checkout portability is **NOT_RUN**. Cached-dependency compilation does not
establish the whole clean-checkout promotion gate, novelty, admissibility of a
scientific observable moment, T2/T3, a general T5 error bound, or T6–T9 label
complexity/controller claims. These remain separate research tasks.
