import KBound

open KBound MeasureTheory ProbabilityTheory
open KBound.JointKernelScore KBound.ActualWorldFrontier KBound.ActualWorldEvidence
open scoped ENNReal ProbabilityTheory

example {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1 / 2) :
    fibreRadius {P | admissible μ f0 fa ha score beta P}
      (fun P => scoreResidual (P : Measure (X × Bool)) f0 fa ha score) = beta :=
  KBound.ActualFibreRadius.actual_class_radius μ f0 fa h0 ha κ0 score hD beta hb0 hb1

example {X Z : Type*} [MeasurableSpace X] [MeasurableSpace Z]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : (Q : Measure (X × Bool)).fst = μ)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1 / 2) :
    fibreRadius {P | admissible μ f0 fa ha score beta P ∧
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q}
      (fun P => scoreResidual (P : Measure (X × Bool)) f0 fa ha score) = beta :=
  KBound.ActualFibreRadius.actual_augmented_fibre_radius μ f0 fa h0 ha κ0 score
    observation Q hQ hD beta hb0 hb1

example {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) :
    fibreRadius {P | admissible μ f0 fa ha score 0 P}
      (fun P => scoreResidual (P : Measure (X × Bool)) f0 fa ha score) = 0 := by
  exact KBound.ActualFibreRadius.actual_class_radius μ f0 fa h0 ha κ0 score hD 0
    (by norm_num) (by norm_num)

-- With M=0, beta=1 exceeds the attainable radius. The small-budget premise
-- cannot be dropped from the equality theorem, even for the full actual class.
example {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x})
    (hm : scoreMargin μ {x | f0 x ≠ fa x} score = 0) :
    fibreRadius {P | admissible μ f0 fa ha score 1 P}
      (fun P => scoreResidual (P : Measure (X × Bool)) f0 fa ha score) = 1 / 2 := by
  obtain ⟨P, hP, he⟩ := KBound.ActualFibreRadius.actual_abs_residual_attained
    μ f0 fa h0 ha κ0 score hD (1 / 2) (by norm_num) le_rfl
  apply fibreRadius_eq_of_bound_and_witness {Q | admissible μ f0 fa ha score 1 Q}
    (fun Q => scoreResidual (Q : Measure (X × Bool)) f0 fa ha score) (1 / 2) ?_
    ⟨⟨P, hP.1, hP.2.trans (by norm_num)⟩, he⟩
  intro Q
  have hi := joint_score_identity (Q.1 : Measure (X × Bool)) f0 fa h0 ha score
  rw [joint_conditional_accuracy (Q.1 : Measure (X × Bool)) f0 fa h0 ha,
    Q.property.1, hm] at hi
  have bounds := disagreementMean_bounds μ {x | f0 x ≠ fa x} hD
    (jointCorrectness (Q.1 : Measure (X × Bool)) fa ha)
  exact abs_le.mpr ⟨by linarith [bounds.1], by linarith [bounds.2]⟩

example {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) :
    fibreRadius {P | admissible μ f0 fa ha score (1 / 2) P}
      (fun P => scoreResidual (P : Measure (X × Bool)) f0 fa ha score) = 1 / 2 := by
  exact KBound.ActualFibreRadius.actual_class_radius μ f0 fa h0 ha κ0 score hD (1 / 2)
    (by norm_num) le_rfl

example :
    let μ : Measure Unit := Measure.dirac ()
    let f0 : Unit → Bool := fun _ => false
    let fa : Unit → Bool := fun _ => true
    let score : CorrectnessField Unit := CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num)
    fibreRadius {P | admissible μ f0 fa measurable_const score (1 / 4) P}
      (fun P => scoreResidual (P : Measure (Unit × Bool)) f0 fa measurable_const score) = 1 / 4 := by
  dsimp only
  exact KBound.ActualFibreRadius.actual_class_radius (Measure.dirac ())
    (fun _ : Unit => false) (fun _ => true) measurable_const measurable_const
    (Kernel.const Unit (Measure.dirac false)) _ (by norm_num [Measure.real]) (1 / 4)
    (by norm_num) (by norm_num)
