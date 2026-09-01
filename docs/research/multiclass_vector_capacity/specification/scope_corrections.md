# Initial scope corrections

Date: 2026-08-31. Status: research decisions, not a new theorem paper.

The supplied design is preserved in `design_specification.txt`; this companion
records the narrowing required by its own counterexample and novelty gates. No
current K-Bound statement, result, or submission artifact is promoted or changed.

## T2: uniform identification is not identification at one boundary fiber

Let `P` be the product simplex, `B` its normalization rows, `H=[B; A]`, and `G`
the **mass-weighted** cost-contrast operator. The intended uniform proof target is

`forall eta,eta' in P: A eta=A eta' implies G eta=G eta'`

if and only if `ker H` is contained in `ker G`. The row-space formulation needs
the corresponding finite-dimensional real linear-algebra proof. This is a
uniform-over-all-feasible-right-hand-sides statement for fixed `A,G`.

At a single nonempty realized fiber `F`, the relevant directions are
`span(F-F)`, not necessarily all of `ker H`. The three-class example
`A=(1,1,0), b=0` forces `F={(0,0,1)}`. The benefit `eta_0` is exactly zero there
although it varies on an affine null direction. Nonnegativity cannot be removed
from this argument. The adversarial review also supplies a non-singleton,
two-stratum boundary example.

Disposition: retain the uniform conjecture with explicit quantifiers; keep the
realized-face characterization separate. Do not promote either from rank tests.

## T3: distinguish algebraic dimension from admissible measurements

The unrestricted rank increment is an algebraic benchmark. It is not generally
the minimum for a fixed dictionary of independently justified measurements.

Exact three-class witness: `q=1`, frozen costs `(0,1/2,1)`, candidate costs
`(0,0,0)`, and no observable restrictions beyond normalization. The unrestricted
rank increment is one. With the admissible dictionary consisting of the three
primitive class probabilities, every single entry is insufficient, whereas two
entries suffice. The dictionary minimum is two.

A literal prohibition on all invertible encodings of the missing answer also
needs care: algebraically minimal sufficient rows can encode exactly the missing
contrast coordinates conditional on existing information. Calling such rows
"observable moments" does not make them label-free or scientifically available.

Disposition: withdraw the unspecified admissible-class equality. Keep separate:

1. An unrestricted linear-algebra benchmark, with no claim of scientific novelty
   or practical availability.
2. A declared primitive-moment dictionary problem, where the minimum can exceed
   the benchmark or be unattainable.
3. Finite-label estimation of those moments, which is not exact population-moment
   access and needs its own statistical guarantee.

The initial oracle searches only a supplied finite dictionary. It does not
certify that any dictionary is scientifically admissible. Its toy constraints
are explicitly labeled unvalidated outside their mathematical examples.

## T5: nonidentification does not imply an ambiguous decision

In the main exact witness, `K=3`, `q=1`, `eta_0=1/5`, frozen costs are
`(0,1/2,1)`, and candidate costs are `(1,0,0)`. The direction `(0,1,-1)` preserves
the observable and normalization, but changes benefit by `-1/2` per unit. Still,
the identified benefit interval is exactly `[1/5,3/5]`: every world strictly
favors ADAPT. The candidate is harmful on class 0, so this is not merely global
pointwise dominance.

Disposition: the proposed implication from a surviving null direction alone is
refuted. A replacement impossibility target needs **feasible strict sign
crossing**, for example two feasible worlds in the same fiber with benefits of
opposite sign. A zero-benefit center and a feasible two-sided null perturbation
are one possible sufficient construction, not an automatic consequence of rank.

The loss/commitment contract must also be explicit. On indistinguishable
opposite-sign worlds, a forced binary decision has the familiar two-point
obstruction. A ternary rule can always abstain with zero wrong commitments. Any
claimed positive lower bound must incorporate mandatory commitment, a power
requirement, or an explicit abstention penalty. The general probabilistic claim
is not promoted merely because these finite witnesses check.

## T6–T9: safety plus a useful decision, with an honest feedback budget

The next statistical theorem target must state both:

- A bound on wrong strict commitments over the permitted target family.
- A correct-commitment probability by a specified budget on a nonempty,
  margin-separated subclass. An always-ABSTAIN procedure must fail this condition.

Before statistical implementation, fix whether the budget is deterministic,
high-probability, or expected at a stopping time. State separate quantities for
unlabeled arrivals, requested labels, returned labels, pending labels, and
calendar time. A query may use only current permitted inputs and already
returned feedback. Informative delays, censoring, adaptive stopping, positivity,
and multiplicity cannot be left implicit.

The asymmetric rare-stratum two-world example is retained as a discovery
candidate. With rare mass `rho` and local perturbation `epsilon`, its benefit
margin is `3 rho epsilon / 2`. A proposed label advantage does not imply an
arrival-time advantage; a fixed positive population margin cannot be maintained
while rare mass tends to zero under fixed normalized costs. Candidate rate
expressions remain unproved and are recorded only in the adversarial report.

No passive/adaptive confidence procedure or simulation is marked implemented by
the initial exact-fiber suite. Query, importance-weight, confidence, multiplicity,
delay, and resource-accounting mutations remain required future gates.

## Novelty and promotion

The initial literature outcome is `UNRESOLVED / PARTIAL_COLLISION`, not a
favorable novelty certificate. Robust ambiguity-set decisions, active comparison
of a baseline and challenger, and selective-label multiclass model selection
already have close precedents. The dated novelty matrix records primary sources
and the unresolved distinctions.

The verified initial algebra/guard slice, if its Lean audit passes, is not the
user's "verified foundation" success level: that level requires the complete
T1–T5 family **and** novelty/nonvacuity. No standalone theory manuscript, natural
experiment, or integration into current K-Bound is authorized by this local
milestone alone. All seven original promotion conditions remain binding.
