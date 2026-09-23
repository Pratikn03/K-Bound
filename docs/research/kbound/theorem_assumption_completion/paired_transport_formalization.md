# Conditional paired-transport formalization

`KBound/Probability/PairedTransport.lean` now supplies the missing formal chain
from simultaneous probability-box containment to a true feasible table,
objective enclosure, and strict-action error bounds. It does not assume that
the true table is feasible as an input to the main result.

## What is constructed and proved

The finite model contains normalized nonnegative source conditional columns
`A`, target conditional columns `B`, and a target prior `pi`. Unit upper bounds
are derived from normalization. The true target marginal is `q_r=sum_y B_ry*pi_y`.
The LP witness is constructed explicitly as:

```text
t_ry = B_ry * pi_y
s_ry = A_ry * pi_y
v_ry = |t_ry - s_ry|
```

On the event that each source and target probability lies in its supplied box,
Lean proves every printed constraint: normalization of the prior, both column
sums, source and target boxes, both signed slack inequalities, and all unit
coordinate bounds. The slack budget equals the declared aggregate conditional
TV budget exactly, by factoring the nonnegative prior and exchanging finite
sums. Zero-prior classes are included; no division by their prior is used.
Thus exact nonemptiness follows from the true witness on the containment event,
rather than from a solver's residual tolerance.

The paired objective is proved equal to the difference of adapted and frozen
accuracies on that same true finite joint distribution. Every feasible
objective is bounded using its finite coefficients and unit coordinate bounds.
Consequently the exact infimum and supremum enclose the true objective whenever
the boxes contain their true probabilities and the external TV restriction
holds. The mathematical certificate does not require a supplied approximate
primal solution or a full-rank confusion matrix.

A finite union-bound theorem combines individual probability-box coverage
budgets. It does not require independence between multinomial coordinates or
between box events. Equal allocation over all declared boxes is proved to sum
to the requested total budget. The paired result then gives coverage of the
objective interval and bounds the union of both wrong strict directions by the
same total budget. A three-way decision function explicitly abstains on an
empty objective set and when zero lies in the interval.

The generic measure statements apply to outer measure when an optimization
event is not separately known measurable. An additional theorem proves that
the coverage and decision-error events are measurable on finite/countable
sample spaces with measurable singletons. This covers the count-vector sample
spaces used by the exact binomial-box construction. No measurability assertion
about a floating-point optimizer is used.

## Remaining boundaries

- **Individual binomial interval coverage is a premise.** The module proves
  its simultaneous union-bound consequences. It does not derive the
  Clopper--Pearson theorem from binomial sampling, prove that the actual samples
  satisfy that sampling model, or prove selection after observing evidence is
  valid.
- **The TV bound is external.** The input model must satisfy it. The proof does
  not infer that restriction from unlabeled target evidence.
- **Endpoints are exact infimum/supremum values.** These give the same coverage
  enclosure as the printed attained extrema. The finite-polytope compactness
  and attainment statement is not separately mechanized in this module.
- **Numerical code remains a separate layer.** Quantile enclosures, rational or
  floating-point coefficient conversion, solver output, directed rounding,
  exact numerical feasibility, and the loss-preserving construction of bins
  are not authenticated by this Lean result.
- **The fixed finite model is explicit.** Random model selection or conditional
  candidate fitting needs the corresponding conditional experiment and
  coverage argument. Repeated decisions and multiple candidates need their
  own valid simultaneous construction.

These limitations must remain visible in the formal correspondence register;
the module is a substantial conditional construction, not an end-to-end
formalization of every statistical and numerical statement in the appendix.

## Integration and checks

Required import:

```lean
import KBound.Probability.PairedTransport
```

The component audit checks these 18 declarations:

```text
FiniteProbabilityVector.le_one
paired_true_target_unit
paired_true_source_unit
paired_true_slack_tv
paired_true_point_feasible
paired_objective_is_benefit
paired_objective_abs_bound
paired_true_benefit_in_objective_interval
finite_boxes_miss_le
uniform_probability_box_budget
paired_box_event_implies_objective_containment
paired_transport_coverage_from_boxes
paired_transport_either_error_le
paired_transport_rule_empty
paired_transport_rule_zero_in_interval
paired_rule_error_implies_endpoint_error
paired_transport_rule_either_error_le
paired_count_sample_events_measurable
```

The module compiled without warnings against the pinned Lean/Mathlib checkout.
The component kernel-axiom receipt and log are
`paired_transport_receipt.json` and `paired_transport_axioms.log`; consult the
receipt's actual status. It binds this module's source hash. The integrating
full audit must bind all transitive sources and the final shared registry.

No shared registry, manuscript, previous receipt, or frozen release artifact
was modified by this component task.
