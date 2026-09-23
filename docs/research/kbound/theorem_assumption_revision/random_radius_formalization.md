# Random extended-radius coverage-to-action

This revision closes the specific mismatch between the manuscript's random
radius in `[0,∞]` and the earlier fixed-real-radius named measure theorems.
It does not prove that any empirical calibration procedure satisfies coverage.

## Mathematical correspondence

`KBound/Probability/RandomRadiusCertificate.lean` represents the benefit and
estimate as arbitrary finite real-valued maps on one probability space, and the
radius as an arbitrary map to `ENNReal`. It defines coverage by
`ENNReal.ofReal |Bhat - B| ≤ eps`. For finite `eps`, Lean proves this equivalent
to the ordinary real inequality against `eps.toReal`; for infinite `eps` it is
automatic.

`extendedCertificate` handles `eps = ∞` before applying `toReal`, so an infinite
radius always gives `ABSTAIN`. This ordering matters because Lean defines
`ENNReal.toReal ∞ = 0`. On the finite branch, the rule is exactly the existing
strict real-interval rule. Its lower-zero and upper-zero endpoints, including
the degenerate zero-radius interval at zero, are proved to abstain.

The module proves the following bounds from the supplied marginal coverage
premise and measurability of its event:

- The false-ADAPT probability is at most `alpha`.
- The false-FREEZE probability is at most `alpha`.
- The probability of their union is at most the same `alpha`, rather than
  `2 alpha`, because both are subsets of the same coverage-failure event.

The radius can depend on the estimate and other randomness. No independence of
these quantities is assumed or used. A convenience theorem derives the
coverage event's measurability from measurable estimate, benefit, and radius
maps. The general statement also bounds the outer measure of error sets when
only coverage-event measurability is supplied; this includes the manuscript's
measurable random-variable case. The `ENNReal` error budget is unrestricted in
the encoded theorem, so the manuscript's real `0 < alpha < 1` is a special case.

This is a direct formalization of the mathematical coverage-to-action
implication, not a weaker theorem with a fixed or finite-only radius. The
manuscript's evaluation-unit declaration and prohibition on using new-unit
labels remain protocol requirements. Neither those requirements nor the
coverage premise follow from this implication.

## Verification and scope

Twelve new registered declarations cover the measure bounds, pointwise
soundness, measurability bridge, and boundary cases. The current registry is
162 declarations (70 legacy and 92 foundation capstones). The frozen release's
150-declaration receipt remains historical and unchanged.

The full strict audit builds the pinned Lean/Mathlib project, checks all
registered names, rejects proof holes, and inspects each declaration's
transitive axioms. Results are recorded in
`formal_random_radius_strict_receipt.json` and
`formal_random_radius_strict.log`. Both the audit and the two focused Python
regression modules passed. `formal_random_radius_source_snapshot.json` binds
the receipt and log to 47 formal source/configuration files by SHA-256 and
records the immutable base commit separately from the modified working tree.
These are scoped formal checks, not a claim
that every manuscript extension or statistical assumption has been established.

The historical unrestricted orbit-selection claim remains refuted; the
full-foundations gate retains its disclosed blocker. Conditional-on-ADAPT
error, repeated use, simultaneous candidates, and empirical coverage transfer
remain outside the new result.

Reproduction from `docs/research/kbound/formal`:

```sh
lake env lean KBound/Probability/RandomRadiusCertificate.lean
python formal_audit.py --build --strict-core --json-out ../theorem_assumption_revision/formal_random_radius_strict_receipt.json
```

Focused Python regression checks from the repository root:

```sh
PYTHONPATH=. python -m pytest tests/test_kbound_formal_audit.py tests/test_release_formal_audit_invocation.py -q
```

Use the already pinned repository verification environment; do not update
Mathlib or replace the frozen release to run these checks.
