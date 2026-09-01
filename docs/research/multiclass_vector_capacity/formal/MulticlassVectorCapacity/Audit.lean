import MulticlassVectorCapacity.Regression
import Lean.Util.CollectAxioms
import Lean.Data.Json.Printer

/-! Exact public theorem types and transitive axiom reports for the verified finite slice. -/

set_option pp.all true

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let names := env.constants.fold (init := #[]) fun acc name _ =>
    if (`MulticlassVectorCapacity).isPrefixOf name then acc.push name else acc
  if names.isEmpty then
    throwError "No compiled declarations were found in the formal namespace"
  for name in names do
    let info ← getConstInfo name
    if info.isAxiom || info.isUnsafe || info.isPartial then
      throwError "Unapproved compiled declaration: {name}"
    let axioms ← collectAxioms name
    for ax in axioms do
      unless #[`propext, `Classical.choice, `Quot.sound].contains ax do
        throwError "Unapproved transitive axiom {ax} used by {name}"
    let used := info.type.getUsedConstants ++
      (info.value?.map Expr.getUsedConstants).getD #[]
    let localDeps := ((used.toList.filter fun n =>
      (`MulticlassVectorCapacity).isPrefixOf n).eraseDups).toArray
    let encoded := Json.arr (localDeps.map fun n => Json.str n.toString)
    logInfo m!"MVC_DIRECT_DEPENDENCIES {name} {encoded.compress}"
  logInfo m!"MVC_NAMESPACE_AUDIT_PASS {names.size}"

-- Basic
#check @MulticlassVectorCapacity.stratum_mass_le_one
#print axioms MulticlassVectorCapacity.stratum_mass_le_one
#check @MulticlassVectorCapacity.conditional_probability_le_one
#print axioms MulticlassVectorCapacity.conditional_probability_le_one

-- Benefit
#check @MulticlassVectorCapacity.cost_benefit_identity
#print axioms MulticlassVectorCapacity.cost_benefit_identity
#check @MulticlassVectorCapacity.expectedCost_mem_unitInterval
#print axioms MulticlassVectorCapacity.expectedCost_mem_unitInterval
#check @MulticlassVectorCapacity.benefit_mem_Icc
#print axioms MulticlassVectorCapacity.benefit_mem_Icc
#check @MulticlassVectorCapacity.positive_benefit_iff_lower_cost
#print axioms MulticlassVectorCapacity.positive_benefit_iff_lower_cost
#check @MulticlassVectorCapacity.negative_benefit_iff_higher_cost
#print axioms MulticlassVectorCapacity.negative_benefit_iff_higher_cost

-- ObservableFiber
#check @MulticlassVectorCapacity.same_fiber_observably_equivalent
#print axioms MulticlassVectorCapacity.same_fiber_observably_equivalent
#check @MulticlassVectorCapacity.identifiedBenefits_nonempty
#print axioms MulticlassVectorCapacity.identifiedBenefits_nonempty
#check @MulticlassVectorCapacity.identifiedBenefits_bddBelow
#print axioms MulticlassVectorCapacity.identifiedBenefits_bddBelow
#check @MulticlassVectorCapacity.identifiedBenefits_bddAbove
#print axioms MulticlassVectorCapacity.identifiedBenefits_bddAbove

-- SignCapacity
#check @MulticlassVectorCapacity.lowerBenefit_le_benefit
#print axioms MulticlassVectorCapacity.lowerBenefit_le_benefit
#check @MulticlassVectorCapacity.benefit_le_upperBenefit
#print axioms MulticlassVectorCapacity.benefit_le_upperBenefit
#check @MulticlassVectorCapacity.lowerBenefit_le_upperBenefit
#print axioms MulticlassVectorCapacity.lowerBenefit_le_upperBenefit
#check @MulticlassVectorCapacity.lowerBenefit_pos_iff_uniform_margin
#print axioms MulticlassVectorCapacity.lowerBenefit_pos_iff_uniform_margin
#check @MulticlassVectorCapacity.upperBenefit_neg_iff_uniform_margin
#print axioms MulticlassVectorCapacity.upperBenefit_neg_iff_uniform_margin
#check @MulticlassVectorCapacity.empty_fiber_no_certificate
#print axioms MulticlassVectorCapacity.empty_fiber_no_certificate
#check @MulticlassVectorCapacity.adapt_decision_iff
#print axioms MulticlassVectorCapacity.adapt_decision_iff
#check @MulticlassVectorCapacity.freeze_decision_iff
#print axioms MulticlassVectorCapacity.freeze_decision_iff
#check @MulticlassVectorCapacity.adapt_decision_sound
#print axioms MulticlassVectorCapacity.adapt_decision_sound
#check @MulticlassVectorCapacity.freeze_decision_sound
#print axioms MulticlassVectorCapacity.freeze_decision_sound
#check @MulticlassVectorCapacity.abstain_decision_iff
#print axioms MulticlassVectorCapacity.abstain_decision_iff

-- Examples
#check @MulticlassVectorCapacity.ThreeClass.mem_fiber_iff
#print axioms MulticlassVectorCapacity.ThreeClass.mem_fiber_iff
#check @MulticlassVectorCapacity.ThreeClass.labels_mem_fiber
#print axioms MulticlassVectorCapacity.ThreeClass.labels_mem_fiber
#check @MulticlassVectorCapacity.ThreeClass.benefit_formula
#print axioms MulticlassVectorCapacity.ThreeClass.benefit_formula
#check @MulticlassVectorCapacity.ThreeClass.labels_benefit
#print axioms MulticlassVectorCapacity.ThreeClass.labels_benefit
#check @MulticlassVectorCapacity.ThreeClass.fiber_benefit_bounds
#print axioms MulticlassVectorCapacity.ThreeClass.fiber_benefit_bounds
#check @MulticlassVectorCapacity.ThreeClass.identified_interval
#print axioms MulticlassVectorCapacity.ThreeClass.identified_interval
#check @MulticlassVectorCapacity.ThreeClass.fiber_nonempty
#print axioms MulticlassVectorCapacity.ThreeClass.fiber_nonempty
#check @MulticlassVectorCapacity.ThreeClass.lowerBenefit_eq
#print axioms MulticlassVectorCapacity.ThreeClass.lowerBenefit_eq
#check @MulticlassVectorCapacity.ThreeClass.upperBenefit_eq
#print axioms MulticlassVectorCapacity.ThreeClass.upperBenefit_eq
#check @MulticlassVectorCapacity.ThreeClass.not_point_identified
#print axioms MulticlassVectorCapacity.ThreeClass.not_point_identified
#check @MulticlassVectorCapacity.ThreeClass.strict_adapt
#print axioms MulticlassVectorCapacity.ThreeClass.strict_adapt
#check @MulticlassVectorCapacity.ThreeClass.candidate_not_pointwise_dominant
#print axioms MulticlassVectorCapacity.ThreeClass.candidate_not_pointwise_dominant
#check @MulticlassVectorCapacity.ThreeClass.nullVariation_simplex_tangent
#print axioms MulticlassVectorCapacity.ThreeClass.nullVariation_simplex_tangent
#check @MulticlassVectorCapacity.ThreeClass.nullVariation_observable_zero
#print axioms MulticlassVectorCapacity.ThreeClass.nullVariation_observable_zero
#check @MulticlassVectorCapacity.ThreeClass.nullVariation_changes_benefit
#print axioms MulticlassVectorCapacity.ThreeClass.nullVariation_changes_benefit
#check @MulticlassVectorCapacity.ThreeClass.no_negative_world
#print axioms MulticlassVectorCapacity.ThreeClass.no_negative_world
#check @MulticlassVectorCapacity.ThreeClass.surviving_null_contrast_without_sign_ambiguity
#print axioms MulticlassVectorCapacity.ThreeClass.surviving_null_contrast_without_sign_ambiguity

-- EdgeCases
#check @MulticlassVectorCapacity.EdgeCases.zero_cost_benefit
#print axioms MulticlassVectorCapacity.EdgeCases.zero_cost_benefit
#check @MulticlassVectorCapacity.EdgeCases.zero_cost_identified_set
#print axioms MulticlassVectorCapacity.EdgeCases.zero_cost_identified_set
#check @MulticlassVectorCapacity.EdgeCases.zero_cost_boundary_abstains
#print axioms MulticlassVectorCapacity.EdgeCases.zero_cost_boundary_abstains
#check @MulticlassVectorCapacity.EdgeCases.inconsistent_fiber_empty
#print axioms MulticlassVectorCapacity.EdgeCases.inconsistent_fiber_empty
#check @MulticlassVectorCapacity.EdgeCases.inconsistent_fiber_rejected
#print axioms MulticlassVectorCapacity.EdgeCases.inconsistent_fiber_rejected
#check @MulticlassVectorCapacity.EdgeCases.negative_conditional_coordinate_rejected
#print axioms MulticlassVectorCapacity.EdgeCases.negative_conditional_coordinate_rejected
#check @MulticlassVectorCapacity.EdgeCases.unnormalized_simplex_rejected
#print axioms MulticlassVectorCapacity.EdgeCases.unnormalized_simplex_rejected
