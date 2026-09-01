# Multiclass vector-capacity: independent research track

This is the isolated theory program requested on 2026-08-31. It is **not a new
K-Bound result**, a submission release, or a completed T1–T9 theorem family.
The existing short PDF, long PDF, Word document, and frozen empirical authorities
are unchanged by this track.

## Current result

The first exact gate has found useful negative results:

- A surviving observable-null direction does not force opposite decisions (T5).
  One checked benefit interval is `[1/5,3/5]`: nonpoint-identified, but ADAPT has
  lower cost throughout the fiber.
- Rank increment one can require two admissible primitive class-probability
  moments (T3). A rank calculation cannot establish scientific availability.
- Point identification on a realized simplex face differs from uniform affine
  identification (T2).
- Wrong-commitment control alone allows zero-label, always-ABSTAIN behavior;
  positive label-complexity claims need an explicit useful-commitment task.

The bounded rational grid contains 67 finite problems. It emits 103
frozen-versus-candidate interval certificates, each with exact primal and dual
witnesses for both endpoints. A separately implemented checker verifies all 206
extrema. Twelve deliberately inconsistent problems are rejected for each of their
two candidates. These checks certify concrete instances, not general theorems.

The local Lean build passed with 46 named proofs and an axiom/dependency check of
all 180 compiled declarations in the namespace. The only transitive axioms are
the documented standard foundations `propext`, `Classical.choice`, and
`Quot.sound`. This covers T1, a scalar fixed-candidate T4 guard, the concrete T5
refutation, and supporting bounds/edge cases. It does not complete T2–T9.

The Lean inventory and theorem ledger report the exact compiled initial scope.
Do not equate the number of declarations or passing tests with new scientific
content. T2/T3 general identification and supplement results, corrected T5
probabilistic lower bounds, and T6–T9 statistical results are not established by
the exact oracle. See the ledger for any subsequent narrower compiled results.

Novelty is **unresolved, with partial literature collisions**. No claim is
promoted into K-Bound. A fresh-checkout gate, complete theorem family, prospective
statistical suite, favorable novelty decision, and original paper release
closure are not supplied by this milestone.

## Files

| File or directory | Role |
| --- | --- |
| `specification/design_specification.txt` | Supplied design, preserved with a terminal newline |
| `specification/scope_corrections.md` | Exact-counterexample-driven narrowing of the design |
| `theorem_ledger.json` | Claim IDs, actual formal scope, evidence, and unpassed gates |
| `formal/` | Pinned Lean project and audited initial proofs; no empirical processing |
| `discovery/` | Exact rational oracle, independent checker, cases, and gate generator |
| `protocols/initial_exact_suite.json` | Deterministic exploratory grid and explicitly uncovered gates |
| `artifacts/initial_exact_certificates.json` | Regenerable primal/dual interval witnesses |
| `tests/` | Exact arithmetic, malformed-input, and deliberate-corruption tests |
| `novelty/2026-08-31_initial_novelty_audit.md` | Primary-source, claim-level collision review |
| `reports/2026-08-31_mathematical_adversarial_review.md` | Counterexamples, corrected quantifiers, unproved rate directions |
| `reports/initial_exact_gate.json` | Deterministic discovery receipt and source bindings |
| `verify_track.py` | Fail-closed source, theorem-type, and statement-record binding checks |
| `reports/initial_local_snapshot.json` | Reviewed local fingerprint, explicitly not a clean-commit release seal |

## Reproduce the exact gate

From the repository root, use Python 3.10 or newer. Only the standard library is
needed; no GPU, downloaded dataset, external service, or numerical LP package is
used.

```sh
python3 -B docs/research/multiclass_vector_capacity/discovery/run_gate.py --check
python3 -B docs/research/multiclass_vector_capacity/discovery/certificate_check.py docs/research/multiclass_vector_capacity/artifacts/initial_exact_certificates.json
python3 -B -m unittest discover -s docs/research/multiclass_vector_capacity/tests -v
PYTHONPATH=docs/research/multiclass_vector_capacity/formal python3 -B -m unittest discover -s docs/research/multiclass_vector_capacity/formal/tests -v
python3 -B docs/research/multiclass_vector_capacity/verify_track.py
```

Omit `--check` only when intentionally regenerating the two discovery artifacts
after a reviewed source or protocol edit. A stale artifact must fail, not silently
rewrite its provenance during verification. Local generated hashes are not a
clean-source release seal or a signature proving who generated an artifact.

The formal subtree documents its separate pinned Lean verification. Its current
dependency-cache reuse is local verification, not a fresh dependency installation
or a clean-checkout portability test.

The statement guard binds reviewed claim wording and scope to exact compiler
types and source/artifact hashes. It rejects unreviewed quantifier, inequality,
constant, orientation, and assumption changes. It is not an automated proof that
arbitrary English and Lean statements mean the same thing; the independent
semantic review remains necessary. Explicit snapshot recording is reserved for a
reviewed update, never a way to turn an unproved conjecture into a theorem.

## Do not infer

No natural-shift experiment, training, dataset deletion, Git repair, blanket
cleanup, commit, push, or old-paper claim promotion is performed by this track.
No "9.5", publication, or breakthrough outcome is guaranteed. The substantive
research question is what correct, non-vacuous result survives the mathematical
and literature gates—not how many checks can be accumulated.
