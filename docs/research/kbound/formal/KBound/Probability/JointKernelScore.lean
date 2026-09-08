import KBound.Probability.JointTargetReduction
import KBound.Probability.MeasureTarget
import Mathlib.Probability.Kernel.Disintegration.StandardBorel
import Mathlib.Data.Real.Sign

/-!
# Actual joint-law correctness and the score residual

The input measurable space is arbitrary. Binary labels have a standard Borel
space, so the pinned disintegration theorem supplies a Markov conditional label
kernel for every finite joint law. We evaluate that kernel at the actual
candidate prediction and identify its disagreement integral with the joint
correctness event. No conditional-mean identity or integrability is postulated.
-/

namespace KBound.JointKernelScore

open MeasureTheory ProbabilityTheory Set JointTargetReduction
open scoped ENNReal ProbabilityTheory

variable {X : Type*} [MeasurableSpace X]

/-- The actual candidate-correctness field of a binary Markov kernel. -/
noncomputable def kernelCorrectness (κ : Kernel X Bool) [IsMarkovKernel κ]
    (fa : X → Bool) (ha : Measurable fa) : CorrectnessField X where
  value x := κ x {fa x}
  measurable_value := by
    classical
    have h : Measurable (fun x => if fa x = true then κ x {true} else κ x {false}) :=
      (κ.measurable_coe (measurableSet_singleton true)).piecewise
        (measurableSet_eq_fun ha (measurable_const (a := true)))
        (κ.measurable_coe (measurableSet_singleton false))
    convert h using 1
    funext x
    cases fa x <;> simp
  value_le_one _ := prob_le_one

/-- Disintegration of an arbitrary joint law, not a supplied model kernel. -/
noncomputable def jointCorrectness (P : Measure (X × Bool)) [IsFiniteMeasure P]
    (fa : X → Bool) (ha : Measurable fa) : CorrectnessField X :=
  kernelCorrectness P.condKernel fa ha

/-- The input disagreement mass is exactly the joint disagreement-event mass. -/
theorem joint_disagreement_mass (P : Measure (X × Bool))
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa) :
    P.fst.real {x | f0 x ≠ fa x} = P.real (disagreementEvent f0 fa) := by
  unfold Measure.real
  rw [Measure.fst_apply (measurable_disagreement f0 fa h0 ha)]
  rfl

/-- The integral of kernel correctness equals the actual joint event mass. -/
theorem kernel_correctness_event (μ : Measure X) [IsProbabilityMeasure μ]
    (κ : Kernel X Bool) [IsMarkovKernel κ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa) :
    (μ ⊗ₘ κ).real (correctOnDisagreement f0 fa fa) =
      ∫ x in {x | f0 x ≠ fa x}, ((kernelCorrectness κ fa ha).value x).toReal ∂μ := by
  classical
  have hD := measurable_disagreement f0 fa h0 ha
  have hA := measurableSet_correctOnDisagreement f0 fa fa h0 ha ha
  have he : (μ ⊗ₘ κ) (correctOnDisagreement f0 fa fa) =
      ∫⁻ x in {x | f0 x ≠ fa x}, (kernelCorrectness κ fa ha).value x ∂μ := by
    rw [Measure.compProd_apply hA, ← lintegral_indicator hD]
    apply lintegral_congr
    intro x
    by_cases hx : f0 x ≠ fa x
    · have hp : Prod.mk x ⁻¹' correctOnDisagreement f0 fa fa = {fa x} := by
        ext y
        simp [correctOnDisagreement, hx]
      simp [hp, hx, kernelCorrectness]
    · have hp : Prod.mk x ⁻¹' correctOnDisagreement f0 fa fa = ∅ := by
        ext y
        simp [correctOnDisagreement, hx]
      simp [hp, hx]
  unfold Measure.real
  rw [he]
  symm
  apply integral_toReal (kernelCorrectness κ fa ha).measurable_value.aemeasurable
  exact Filter.Eventually.of_forall fun x =>
    lt_of_le_of_lt ((kernelCorrectness κ fa ha).value_le_one x) ENNReal.one_lt_top

/-- Arbitrary-joint version, with the library-provided conditional kernel. -/
theorem joint_correctness_event (P : Measure (X × Bool)) [IsProbabilityMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa) :
    P.real (correctOnDisagreement f0 fa fa) =
      ∫ x in {x | f0 x ≠ fa x}, ((jointCorrectness P fa ha).value x).toReal ∂P.fst := by
  have h := kernel_correctness_event P.fst P.condKernel f0 fa h0 ha
  rw [Measure.disintegrate P P.condKernel] at h
  exact h

/-- The conditional accuracy in the manuscript equals the conditional-kernel mean. -/
theorem joint_conditional_accuracy (P : Measure (X × Bool)) [IsProbabilityMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa) :
    P.real (correctOnDisagreement f0 fa fa) / P.real (disagreementEvent f0 fa) =
      disagreementMean P.fst {x | f0 x ≠ fa x} (jointCorrectness P fa ha) := by
  rw [joint_correctness_event P f0 fa h0 ha, ← joint_disagreement_mass P f0 fa h0 ha]
  rfl

noncomputable def scoreMargin (μ : Measure X) (D : Set X) (score : CorrectnessField X) : ℝ :=
  disagreementMean μ D score - 1 / 2

noncomputable def scoreResidual (P : Measure (X × Bool)) [IsFiniteMeasure P]
    (f0 fa : X → Bool) (ha : Measurable fa) (score : CorrectnessField X) : ℝ :=
  (∫ x in {x | f0 x ≠ fa x},
    (((jointCorrectness P fa ha).value x).toReal - (score.value x).toReal) ∂P.fst) /
      P.fst.real {x | f0 x ≠ fa x}

/-- Bounded measurable scores supply integrability automatically; the residual
is the actual conditional integral, not an independent scalar parameter. -/
theorem joint_score_identity (P : Measure (X × Bool)) [IsProbabilityMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (score : CorrectnessField X) :
    scoreMargin P.fst {x | f0 x ≠ fa x} score + scoreResidual P f0 fa ha score =
      P.real (correctOnDisagreement f0 fa fa) / P.real (disagreementEvent f0 fa) - 1 / 2 := by
  rw [joint_conditional_accuracy P f0 fa h0 ha]
  exact conditional_score_residual_identity P.fst _ _ _
    (correctnessField_integrable _ score) (correctnessField_integrable _ (jointCorrectness P fa ha))

/-- Full arbitrary-joint-law benefit identity with the manuscript's score and residual. -/
theorem joint_score_benefit (P : Measure (X × Bool)) [IsProbabilityMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (score : CorrectnessField X) (hD : 0 < P.fst.real {x | f0 x ≠ fa x}) :
    populationBenefit f0 fa P = 2 * P.fst.real {x | f0 x ≠ fa x} *
      (scoreMargin P.fst {x | f0 x ≠ fa x} score + scoreResidual P f0 fa ha score) := by
  rw [joint_score_identity P f0 fa h0 ha score,
    joint_disagreement_mass P f0 fa h0 ha] at *
  exact binary_population_reduction P f0 fa h0 ha hD

/-- Margin feasibility follows from the actual bounded score. -/
theorem scoreMargin_bounds (μ : Measure X) [IsProbabilityMeasure μ]
    (D : Set X) (score : CorrectnessField X) (hD : 0 < μ.real D) :
    -1 / 2 ≤ scoreMargin μ D score ∧ scoreMargin μ D score ≤ 1 / 2 := by
  have h := disagreementMean_bounds μ D hD score
  unfold scoreMargin
  constructor <;> linarith [h.1, h.2]

/-- The score/residual sign reduction needs no residual-budget assumption. -/
theorem joint_score_sign (P : Measure (X × Bool)) [IsProbabilityMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (score : CorrectnessField X) (hD : 0 < P.fst.real {x | f0 x ≠ fa x}) :
    Real.sign (populationBenefit f0 fa P) =
      Real.sign (scoreMargin P.fst {x | f0 x ≠ fa x} score + scoreResidual P f0 fa ha score) := by
  rw [joint_score_benefit P f0 fa h0 ha score hD]
  have hc : 0 < 2 * P.fst.real {x | f0 x ≠ fa x} := mul_pos (by norm_num) hD
  have hn (r : ℝ) : 2 * P.fst.real {x | f0 x ≠ fa x} * r < 0 ↔ r < 0 := by
    simpa only [mul_zero] using (mul_lt_mul_iff_right₀ hc : _ * r < _ * 0 ↔ r < 0)
  simp only [Real.sign, hn, mul_pos_iff_of_pos_left hc]

end KBound.JointKernelScore

#print axioms KBound.JointKernelScore.joint_disagreement_mass
#print axioms KBound.JointKernelScore.kernel_correctness_event
#print axioms KBound.JointKernelScore.joint_correctness_event
#print axioms KBound.JointKernelScore.joint_conditional_accuracy
#print axioms KBound.JointKernelScore.joint_score_identity
#print axioms KBound.JointKernelScore.joint_score_benefit
#print axioms KBound.JointKernelScore.scoreMargin_bounds
#print axioms KBound.JointKernelScore.joint_score_sign
