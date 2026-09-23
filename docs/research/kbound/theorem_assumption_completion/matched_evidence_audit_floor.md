# Matched-evidence and audit-floor formal completion

Two new modules address previously unassembled probability statements without
changing their manuscript assumptions or the frozen release.

## Randomized matched-evidence abstention

`KBound/Probability/MatchedAbstention.lean` equips the existing three-action
type with its discrete measurable space. A decision rule is an arbitrary
measurable Markov kernel from evidence to actions. Identical observable
pushforward laws imply identical action laws after applying this same kernel.
The proof then converts the negative world's false-ADAPT control and positive
world's false-FREEZE control into action bounds on that common law. Measure
subadditivity and probability complementation give abstention at least
`1 - 2 alpha`. This is a theorem about genuine probability measures, not just
three unconnected numeric weights.

At zero population benefit, both strict commitments are directional errors.
The module proves the same probability bound there, transfers it along an
evidence fibre, and proves that pointwise strict soundness at zero is equivalent
to ABSTAIN. Two additional capstones obtain the zero-benefit world directly
from the existing measurable label-kernel construction for the full
correctness-field class. Thus the closed-band randomized and pointwise clauses
do not introduce an unproved abstract richness assumption. Those concrete
class capstones concern any measurable function of the input; the general
matched-law capstone permits arbitrary observable spaces, including a supplied
batch evidence law.

## Exact evidence-fibre audit floor

`KBound/Probability/AuditFloor.lean` proves the floor for an arbitrary nonempty
bounded set of attainable real residual magnitudes. The supremum need not be
attained. Lean constructs an increasing sequence of attainable magnitudes
converging to the supremum, identifies the limiting upper-tail event with a
decreasing intersection, and applies continuity of the probability measure
from above. There is no union-bound loss and no finite-world restriction.

The full fibre theorem takes probability laws indexed by the nonempty fibre,
equal observable laws, a common independent random seed, and a measurable
audit of the observable and seed. Product-measure pushforwards prove the audit
law is common. Worldwise residual coverage then gives coverage of the exact
fibre supremum under every world. The audit may be any finite real-valued map;
nonnegativity is unnecessary for the implication, so the printed nonnegative
real audit is included. The formal theorem explicitly assumes bounded residual
magnitudes, as supplied by the manuscript's bounded score/correctness setting.

The constant supremum audit has zero error, and a separate implication proves
that its coverage event precludes extra strict commitments by an audited
frontier. These are oracle comparisons, not algorithms estimating the unknown
fibre radius.

The supporting identity `Gamma(C_beta)=beta` for `0<=beta<=1/2` is also
mechanized. At nonnegative margin it constructs a constant correctness field
with residual `-beta`; at negative margin it constructs one with residual
`+beta`. Feasible margin and the radius bound guarantee the field remains in
`[0,1]`. Every admitted residual magnitude is at most beta, and the construction
attains beta, proving the supremum identity. The beta-zero and half-unit
endpoints are included. No equality beyond the stated radius range is claimed.

## Integration and verification

Imports required by the integrating agent:

```lean
import KBound.Probability.MatchedAbstention
import KBound.Probability.AuditFloor
```

Recommended registered declarations (8 and 7 respectively):

```text
decision_abstention_ge_of_directional_bounds
matched_evidence_randomized_action_law
randomized_matched_evidence_abstention
randomized_zero_benefit_abstention
randomized_matched_zero_boundary_abstention
zero_benefit_sound_iff_abstain
measurable_closed_band_randomized_abstention
measurable_closed_band_pointwise_abstention
audit_floor_of_uniform_coverage
constant_supremum_audit_valid
residual_supremum_nonneg
audited_frontier_no_extra_commitment
common_independent_seed_audit_law
evidence_fibre_audit_floor
full_correctness_residual_radius_exact
```

Both modules compile directly against the pinned dependencies. Component
kernel-axiom results are in `matched_audit_floor_receipt.json` and
`matched_audit_floor_axioms.log`; consult the receipt's actual status. The
receipt hashes the two new source modules. The integrating full audit must
also bind transitive formal sources and check the final shared registry.

These proofs do not establish that a deployment's evidence laws match, that its
audit has uniform coverage, that a narrower target class is rich, or that an
unknown fibre radius can be identified from a finite observation. They prove
the consequences of the explicit mathematical premises.
