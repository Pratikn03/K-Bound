import KBound.Probability.MeasureSwap
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Disagreement reduction for arbitrary joint target laws

This module does not restrict the target to a constructed correctness kernel.
For any measurable fixed predictors and any finite joint law, actual zero-one
population benefit is the difference of the two correct-on-disagreement event
masses. Positive disagreement mass gives the multiclass conditional-accuracy
identity. Binary labels additionally give the factor-two reduction. Conditional
probabilities here are event-mass ratios, so no regular conditional distribution
existence assumption is introduced. A separate integral identity connects a
bounded-score residual to the same conditional mean.
-/

namespace KBound.JointTargetReduction

open MeasureTheory Set

variable {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y] [MeasurableEq Y]

def disagreementEvent (f0 fa : X → Y) : Set (X × Y) :=
  {xy | f0 xy.1 ≠ fa xy.1}

def correctOnDisagreement (f0 fa f : X → Y) : Set (X × Y) :=
  {xy | f0 xy.1 ≠ fa xy.1 ∧ xy.2 = f xy.1}

lemma measurableSet_disagreementEvent (f0 fa : X → Y)
    (h0 : Measurable f0) (ha : Measurable fa) :
    MeasurableSet (disagreementEvent f0 fa) :=
  (measurableSet_eq_fun (h0.comp measurable_fst) (ha.comp measurable_fst)).compl

lemma measurableSet_correctOnDisagreement (f0 fa f : X → Y)
    (h0 : Measurable f0) (ha : Measurable fa) (hf : Measurable f) :
    MeasurableSet (correctOnDisagreement f0 fa f) :=
  (measurableSet_disagreementEvent f0 fa h0 ha).inter
    (measurableSet_eq_fun measurable_snd (hf.comp measurable_fst))

omit [MeasurableSpace X] [MeasurableSpace Y] [MeasurableEq Y] in
lemma zeroOneBenefit_eq_indicator_sub (f0 fa : X → Y) (xy : X × Y) :
    zeroOneBenefit f0 fa xy =
      (correctOnDisagreement f0 fa fa).indicator (fun _ => (1 : ℝ)) xy -
      (correctOnDisagreement f0 fa f0).indicator (fun _ => (1 : ℝ)) xy := by
  classical
  by_cases hd : f0 xy.1 = fa xy.1
  · simp [zeroOneBenefit, correctOnDisagreement, hd]
  · by_cases ha : xy.2 = fa xy.1 <;> by_cases h0 : xy.2 = f0 xy.1 <;>
      simp_all [zeroOneBenefit, correctOnDisagreement]

/-- The population loss-difference identity for arbitrary multiclass joint laws,
including targets with positive mass on labels predicted by neither model. -/
theorem population_benefit_event_difference (P : Measure (X × Y)) [IsFiniteMeasure P]
    (f0 fa : X → Y) (h0 : Measurable f0) (ha : Measurable fa) :
    populationBenefit f0 fa P =
      P.real (correctOnDisagreement f0 fa fa) -
      P.real (correctOnDisagreement f0 fa f0) := by
  have hA := measurableSet_correctOnDisagreement f0 fa fa h0 ha ha
  have hB := measurableSet_correctOnDisagreement f0 fa f0 h0 ha h0
  unfold populationBenefit
  simp_rw [zeroOneBenefit_eq_indicator_sub]
  rw [integral_sub ((integrable_const (1 : ℝ)).indicator hA)
    ((integrable_const (1 : ℝ)).indicator hB)]
  simp only [integral_indicator_const _ hA, integral_indicator_const _ hB,
    smul_eq_mul, mul_one]

/-- The appendix multiclass formula with the two conditional accuracies written
as their defining event ratios. Positive disagreement mass is the sole division
premise; no binary complementarity is assumed. -/
theorem multiclass_population_reduction (P : Measure (X × Y)) [IsFiniteMeasure P]
    (f0 fa : X → Y) (h0 : Measurable f0) (ha : Measurable fa)
    (hD : 0 < P.real (disagreementEvent f0 fa)) :
    populationBenefit f0 fa P = P.real (disagreementEvent f0 fa) *
      (P.real (correctOnDisagreement f0 fa fa) / P.real (disagreementEvent f0 fa) -
       P.real (correctOnDisagreement f0 fa f0) / P.real (disagreementEvent f0 fa)) := by
  rw [population_benefit_event_difference P f0 fa h0 ha]
  field_simp

omit [MeasurableSpace X] in
lemma binary_benefit_eq_indicators (f0 fa : X → Bool) (xy : X × Bool) :
    zeroOneBenefit f0 fa xy =
      2 * (correctOnDisagreement f0 fa fa).indicator (fun _ => (1 : ℝ)) xy -
      (disagreementEvent f0 fa).indicator (fun _ => (1 : ℝ)) xy := by
  classical
  cases h0 : f0 xy.1 <;> cases ha : fa xy.1 <;> cases hy : xy.2 <;>
    norm_num [zeroOneBenefit, correctOnDisagreement, disagreementEvent, h0, ha, hy]

/-- Binary complementarity, integrated against an arbitrary joint target law. -/
theorem binary_population_event_reduction (P : Measure (X × Bool)) [IsFiniteMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa) :
    populationBenefit f0 fa P =
      2 * P.real (correctOnDisagreement f0 fa fa) - P.real (disagreementEvent f0 fa) := by
  have hA := measurableSet_correctOnDisagreement f0 fa fa h0 ha ha
  have hD := measurableSet_disagreementEvent f0 fa h0 ha
  unfold populationBenefit
  simp_rw [binary_benefit_eq_indicators]
  rw [integral_sub (((integrable_const (1 : ℝ)).indicator hA).const_mul 2)
    ((integrable_const (1 : ℝ)).indicator hD)]
  rw [integral_const_mul]
  simp only [integral_indicator_const _ hA, integral_indicator_const _ hD,
    smul_eq_mul, mul_one]

/-- The paper's arbitrary-target-law binary reduction, expressed directly in
terms of joint probabilities. It includes atomic and nonatomic input laws. -/
theorem binary_population_reduction (P : Measure (X × Bool)) [IsFiniteMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (hD : 0 < P.real (disagreementEvent f0 fa)) :
    populationBenefit f0 fa P = 2 * P.real (disagreementEvent f0 fa) *
      (P.real (correctOnDisagreement f0 fa fa) / P.real (disagreementEvent f0 fa) - 1 / 2) := by
  rw [binary_population_event_reduction P f0 fa h0 ha]
  field_simp

/-- The actual conditional-integral margin/residual identity, rather than only
an algebraic identity for supplied mean values. Integrability follows for the
paper's measurable [0,1]-valued score/correctness fields on a probability space. -/
theorem conditional_score_residual_identity (mu : Measure X) (D : Set X)
    (score eta : X → ℝ) (hs : IntegrableOn score D mu) (he : IntegrableOn eta D mu) :
    ((∫ x in D, score x ∂mu) / mu.real D - 1 / 2) +
      (∫ x in D, (eta x - score x) ∂mu) / mu.real D =
      (∫ x in D, eta x ∂mu) / mu.real D - 1 / 2 := by
  rw [integral_sub he hs]
  ring

end KBound.JointTargetReduction

#print axioms KBound.JointTargetReduction.population_benefit_event_difference
#print axioms KBound.JointTargetReduction.multiclass_population_reduction
#print axioms KBound.JointTargetReduction.binary_population_event_reduction
#print axioms KBound.JointTargetReduction.binary_population_reduction
#print axioms KBound.JointTargetReduction.conditional_score_residual_identity
