import MulticlassVectorCapacity.Benefit
import MulticlassVectorCapacity.SignCapacity
import MulticlassVectorCapacity.Examples
import MulticlassVectorCapacity.EdgeCases

namespace MulticlassVectorCapacity.Regression

variable {S Y J : Type*} [Fintype S] [Fintype Y] [Fintype J]

-- Reversing the public benefit orientation must break this type-level contract.
example (M : Model S Y J) (η : ConditionalLabels S Y) (j : J) :
    expectedCost M η none - expectedCost M η (some j) = benefit M η j :=
  cost_benefit_identity M η j

-- Removing probability/cost assumptions must not turn this into an unconditional bound.
example (M : Model S Y J) (η : ConditionalLabels S Y) (a : Option J) :
    0 ≤ expectedCost M η a ∧ expectedCost M η a ≤ 1 :=
  expectedCost_mem_unitInterval M η a

-- An empty equality fiber is not a valid strict certificate.
example {R : Type*} (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) (h : ¬ (fiber A b).Nonempty) :
    fiberDecision M A b j = none :=
  empty_fiber_no_certificate M A b j h

-- This non-dominated candidate is uniformly beneficial without point identification.
example :
    identifiedBenefits ThreeClass.model ThreeClass.observable ThreeClass.observedValue 0 =
      Set.Icc (1 / 5 : ℝ) (3 / 5 : ℝ) :=
  ThreeClass.identified_interval

example :
    ¬ pointIdentified ThreeClass.model ThreeClass.observable ThreeClass.observedValue 0 :=
  ThreeClass.not_point_identified

-- Equality at zero is ABSTAIN, never a strict direction.
example : fiberDecision EdgeCases.zeroCostModel ThreeClass.observable ThreeClass.observedValue 0 =
    some .abstain := EdgeCases.zero_cost_boundary_abstains

end MulticlassVectorCapacity.Regression
