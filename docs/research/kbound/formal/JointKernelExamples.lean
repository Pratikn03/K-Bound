import KBound

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal ProbabilityTheory

open KBound KBound.JointKernelScore KBound.JointTargetReduction

-- These three names failed against the unchanged 238 root before implementation.
#check KBound.JointKernelScore.joint_correctness_event
#check KBound.JointKernelScore.joint_score_benefit
#check KBound.JointKernelScore.joint_score_sign

-- Arbitrary measurable input space: no StandardBorelSpace X premise.
example {X : Type*} [MeasurableSpace X] (P : Measure (X × Bool))
    [IsProbabilityMeasure P] (f0 fa : X → Bool) (h0 : Measurable f0)
    (ha : Measurable fa) (score : CorrectnessField X)
    (hD : 0 < P.fst.real {x | f0 x ≠ fa x}) :
    populationBenefit f0 fa P = 2 * P.fst.real {x | f0 x ≠ fa x} *
      (scoreMargin P.fst {x | f0 x ≠ fa x} score + scoreResidual P f0 fa ha score) :=
  joint_score_benefit P f0 fa h0 ha score hD

-- Candidate-always-correct endpoint, with a nonzero actual residual.
example : scoreResidual (Measure.dirac ((), true))
    (fun _ : Unit => false) (fun _ => true) measurable_const
    (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num)) = 1 / 2 := by
  have h := joint_score_identity (Measure.dirac ((), true))
    (fun _ : Unit => false) (fun _ => true) measurable_const measurable_const
    (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num))
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant,
    Measure.real, disagreementEvent, correctOnDisagreement] at h
  exact h

-- Candidate-always-wrong endpoint, without any budget assumption.
example : scoreResidual (Measure.dirac ((), false))
    (fun _ : Unit => false) (fun _ => true) measurable_const
    (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num)) = -1 / 2 := by
  have h := joint_score_identity (Measure.dirac ((), false))
    (fun _ : Unit => false) (fun _ => true) measurable_const measurable_const
    (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num))
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant,
    Measure.real, disagreementEvent, correctOnDisagreement] at h
  simpa only [neg_div] using h

-- Agreement-only targets have zero correctness-on-disagreement mass; no
-- positive-mass hypothesis is smuggled into the event identity.
example (P : Measure (Unit × Bool)) [IsProbabilityMeasure P] :
    (∫ x in {_x : Unit | (false : Bool) ≠ false},
      ((jointCorrectness P (fun _ => false) measurable_const).value x).toReal ∂P.fst) = 0 := by
  rw [← joint_correctness_event P (fun _ => false) (fun _ => false)
    measurable_const measurable_const]
  simp [correctOnDisagreement]

-- Nonconstant prediction, a proper disagreement set, and arbitrary dependence
-- of the input and label remain covered by the same event/integral identity.
example (P : Measure (Bool × Bool)) [IsProbabilityMeasure P] :
    P.real {(x, y) | x = true ∧ y = true} =
      ∫ x in {true}, ((jointCorrectness P id measurable_id).value x).toReal ∂P.fst := by
  have h := joint_correctness_event P (fun _ => false) id measurable_const measurable_id
  have hd : {x : Bool | false ≠ id x} = {true} := by
    ext x
    cases x <;> simp
  have he : correctOnDisagreement (fun _ : Bool => false) id id =
      {(x, y) | x = true ∧ y = true} := by
    ext ⟨x, y⟩
    cases x <;> cases y <;> simp [correctOnDisagreement]
  rw [hd, he] at h
  exact h
