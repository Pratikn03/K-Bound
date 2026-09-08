import KBound

open MeasureTheory ProbabilityTheory Set KBound KBound.JointKernelScore
open KBound.ActualWorldFrontier KBound.ActualWorldEvidence
open scoped ENNReal ProbabilityTheory

#check KBound.ActualWorldSubclass.full_class_tilt_witnesses
#check KBound.ActualWorldSubclass.subclass_augmented_fibre_strict_direction_iff
#check KBound.ActualWorldSubclass.subclass_closed_band_abstention

-- Paper closure admits interior pairs without imposing the zero interior world.
example {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : KBound.ActualWorldSubclass.InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : KBound.ActualWorldSubclass.BoundaryClosed μ f0 fa h0 ha κ0 score beta W) :
    ((∀ P : ProbabilityMeasure (X × Bool), W P →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∨
      (∀ P : ProbabilityMeasure (X × Bool), W P →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0)) ↔
      beta < |scoreMargin μ {x | f0 x ≠ fa x} score| :=
  KBound.ActualWorldSubclass.subclass_strict_direction_iff μ f0 fa h0 ha κ0 score
    hD beta hb W hsub hi he

-- Individual direction iff explicitly supplies an actually realized member.
example {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : KBound.ActualWorldSubclass.InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : KBound.ActualWorldSubclass.BoundaryClosed μ f0 fa h0 ha κ0 score beta W)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : W Q) :
    (∀ P : ProbabilityMeasure (X × Bool), W P →
      0 < populationBenefit f0 fa (P : Measure (X × Bool))) ↔
      beta < scoreMargin μ {x | f0 x ≠ fa x} score :=
  KBound.ActualWorldSubclass.subclass_adapt_iff_of_nonempty μ f0 fa h0 ha κ0 score
    hD beta hb W hsub hi he ⟨Q, hQ⟩

example {X : Type*} [MeasurableSpace X] (f0 fa : X → Bool) :
    (∀ P : ProbabilityMeasure (X × Bool), False →
      0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∧
    (∀ P : ProbabilityMeasure (X × Bool), False →
      populationBenefit f0 fa (P : Measure (X × Bool)) < 0) := by simp

local notation "mu" => (Measure.dirac () : Measure Unit)
local notation "f0" => (fun _ : Unit => false)
local notation "fa" => (fun _ : Unit => true)
local notation "kernel0" => (Kernel.const Unit (Measure.dirac false))
local notation "score" => (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num) : CorrectnessField Unit)

-- Exactly two actual worlds, with benefits +/-1/4, satisfy the paper closure
-- at M=0,beta=1/4 but exclude the zero world. This rejects the superseded premise.
example : ∃ W : ProbabilityMeasure (Unit × Bool) → Prop,
    (∀ P, W P → admissible mu f0 fa measurable_const score (1 / 4) P) ∧
    KBound.ActualWorldSubclass.InteriorClosed mu f0 fa measurable_const measurable_const
      kernel0 score (1 / 4) W ∧
    KBound.ActualWorldSubclass.BoundaryClosed mu f0 fa measurable_const measurable_const
      kernel0 score (1 / 4) W ∧
    ¬ W (KBound.ActualWorldSubclass.zeroWorld mu f0 fa measurable_const measurable_const kernel0) := by
  let δ : KBound.ActualWorldSubclass.Tilt := ⟨1 / 8, by norm_num⟩
  let Ppos := KBound.ActualWorldSubclass.tiltWorld mu f0 fa measurable_const measurable_const kernel0 δ true
  let Pneg := KBound.ActualWorldSubclass.tiltWorld mu f0 fa measurable_const measurable_const kernel0 δ false
  let W := fun P : ProbabilityMeasure (Unit × Bool) => P = Ppos ∨ P = Pneg
  have hD : 0 < (mu).real {x | f0 x ≠ fa x} := by norm_num [Measure.real]
  have hg : δ.val < (1 / 4 : ℝ) - |scoreMargin mu {x | f0 x ≠ fa x} score| := by
    norm_num [δ, scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]
  refine ⟨W, ?_, ?_, ?_, ?_⟩
  · intro P hP
    rcases hP with rfl | rfl
    · exact KBound.ActualWorldSubclass.tiltWorld_admissible mu f0 fa measurable_const measurable_const
        kernel0 score hD (1 / 4) δ true hg
    · exact KBound.ActualWorldSubclass.tiltWorld_admissible mu f0 fa measurable_const measurable_const
        kernel0 score hD (1 / 4) δ false hg
  · intro _
    exact ⟨δ, hg, Or.inl rfl, Or.inr rfl⟩
  · intro hboundary
    norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real] at hboundary
  · intro hz
    have hzero := KBound.ActualWorldSubclass.zeroWorld_benefit mu f0 fa
      measurable_const measurable_const kernel0 hD
    rcases hz with hz | hz
    · change _ = Ppos at hz
      rw [hz] at hzero
      change populationBenefit f0 fa (KBound.ActualWorldSubclass.tiltWorld mu f0 fa
        measurable_const measurable_const kernel0 δ true : Measure (Unit × Bool)) = 0 at hzero
      rw [KBound.ActualWorldSubclass.tiltWorld_benefit mu f0 fa measurable_const measurable_const
        kernel0 hD δ true] at hzero
      norm_num [δ, Measure.real] at hzero
    · change _ = Pneg at hz
      rw [hz] at hzero
      change populationBenefit f0 fa (KBound.ActualWorldSubclass.tiltWorld mu f0 fa
        measurable_const measurable_const kernel0 δ false : Measure (Unit × Bool)) = 0 at hzero
      rw [KBound.ActualWorldSubclass.tiltWorld_benefit mu f0 fa measurable_const measurable_const
        kernel0 hD δ false] at hzero
      norm_num [δ, Measure.real] at hzero

example : admissible mu f0 fa measurable_const score 0
    (KBound.ActualWorldSubclass.zeroWorld mu f0 fa measurable_const measurable_const kernel0) := by
  apply (KBound.ActualWorldSubclass.zeroWorld_admissible_iff mu f0 fa measurable_const measurable_const
    kernel0 score (by norm_num [Measure.real]) 0).mpr
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Both boundary residuals use the same actual zero-benefit construction.
example : scoreResidual
    (KBound.ActualWorldSubclass.zeroWorld mu f0 fa measurable_const measurable_const kernel0 :
      Measure (Unit × Bool)) f0 fa measurable_const
        (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) = -(1 / 4 : ℝ) := by
  rw [KBound.ActualWorldSubclass.zeroWorld_residual mu f0 fa measurable_const measurable_const
    kernel0 _ (by norm_num [Measure.real])]
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

example : scoreResidual
    (KBound.ActualWorldSubclass.zeroWorld mu f0 fa measurable_const measurable_const kernel0 :
      Measure (Unit × Bool)) f0 fa measurable_const
        (CorrectnessField.constant (1 / 4) (by norm_num) (by norm_num)) = (1 / 4 : ℝ) := by
  rw [KBound.ActualWorldSubclass.zeroWorld_residual mu f0 fa measurable_const measurable_const
    kernel0 _ (by norm_num [Measure.real])]
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Class inclusion alone does not imply the paper's interior closure.
example : ∃ Q : ProbabilityMeasure (Unit × Bool),
    admissible mu f0 fa measurable_const score 1 Q ∧
    (∀ P : ProbabilityMeasure (Unit × Bool), P = Q →
      0 < populationBenefit f0 fa (P : Measure (Unit × Bool))) ∧
    ¬ KBound.ActualWorldSubclass.InteriorClosed mu f0 fa measurable_const measurable_const
      kernel0 score 1 (fun P => P = Q) := by
  let Q := fieldWorld mu f0 fa measurable_const measurable_const kernel0
    (CorrectnessField.constant 1 (by norm_num) (by norm_num))
  have hD : 0 < (mu).real {x | f0 x ≠ fa x} := by norm_num [Measure.real]
  have hq : populationBenefit f0 fa (Q : Measure (Unit × Bool)) = 1 := by
    change populationBenefit f0 fa (correctnessFieldTarget mu f0 fa
      measurable_const measurable_const kernel0 _) = 1
    rw [correctnessFieldTarget_benefit mu f0 fa measurable_const measurable_const kernel0 hD]
    norm_num [disagreementMean, CorrectnessField.constant, Measure.real]
  refine ⟨Q, ⟨fieldWorld_fst _ _ _ _ _ _ _, ?_⟩, ?_, ?_⟩
  · change |scoreResidual (fieldWorld mu f0 fa measurable_const measurable_const kernel0
      (CorrectnessField.constant 1 (by norm_num) (by norm_num)) : Measure (Unit × Bool))
        f0 fa measurable_const score| ≤ 1
    rw [fieldWorld_residual mu f0 fa measurable_const measurable_const kernel0 _ _ hD]
    norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]
  · intro P heq
    rw [heq, hq]
    norm_num
  · intro hi
    obtain ⟨δ, _, _, hn⟩ := hi (by
      norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real])
    have hb := KBound.ActualWorldSubclass.tiltWorld_benefit mu f0 fa
      measurable_const measurable_const kernel0 hD δ false
    rw [hn, hq] at hb
    norm_num [Measure.real] at hb
    linarith [δ.property.1]

-- Individual negative direction also requires an actually realized member.
example {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (g0 ga : X → Bool) (h0 : Measurable g0) (ha : Measurable ga)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (s : CorrectnessField X)
    (hD : 0 < μ.real {x | g0 x ≠ ga x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ g0 ga ha s beta P)
    (hi : KBound.ActualWorldSubclass.InteriorClosed μ g0 ga h0 ha κ0 s beta W)
    (he : KBound.ActualWorldSubclass.BoundaryClosed μ g0 ga h0 ha κ0 s beta W)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : W Q) :
    (∀ P : ProbabilityMeasure (X × Bool), W P →
      populationBenefit g0 ga (P : Measure (X × Bool)) < 0) ↔
      scoreMargin μ {x | g0 x ≠ ga x} s < -beta :=
  KBound.ActualWorldSubclass.subclass_freeze_iff_of_nonempty μ g0 ga h0 ha κ0 s
    hD beta hb W hsub hi he ⟨Q, hQ⟩

-- The actual event bound is checked through the public import, with error
-- control only over W and without a zero-interior membership premise.
example {X Z : Type*} [MeasurableSpace X] [MeasurableSpace Z]
    [MeasurableSpace Decision] [MeasurableSingletonClass Decision]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (g0 ga : X → Bool) (h0 : Measurable g0) (ha : Measurable ga)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (s : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (policy : Kernel (Z × ℝ) Decision) [IsMarkovKernel policy]
    (hD : 0 < μ.real {x | g0 x ≠ ga x}) (beta : ℝ)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ g0 ga ha s beta P)
    (hi : KBound.ActualWorldSubclass.InteriorClosed μ g0 ga h0 ha κ0 s beta W)
    (he : KBound.ActualWorldSubclass.BoundaryClosed μ g0 ga h0 ha κ0 s beta W)
    (hband : |scoreMargin μ {x | g0 x ≠ ga x} s| ≤ beta) (alpha : ENNReal)
    (hcontrol : ∀ P : ProbabilityMeasure (X × Bool), W P →
      (policy ∘ₘ (augmentedLaw observation g0 ga s P : Measure (Z × ℝ)))
        {a | a = Decision.adapt ∧ populationBenefit g0 ga (P : Measure (X × Bool)) ≤ 0} ≤ alpha ∧
      (policy ∘ₘ (augmentedLaw observation g0 ga s P : Measure (Z × ℝ)))
        {a | a = Decision.freeze ∧ 0 ≤ populationBenefit g0 ga (P : Measure (X × Bool))} ≤ alpha)
    (P : ProbabilityMeasure (X × Bool)) (hP : W P) :
    1 - 2 * alpha ≤ (policy ∘ₘ
      (augmentedLaw observation g0 ga s P : Measure (Z × ℝ))) {Decision.abstain} :=
  KBound.ActualWorldSubclass.subclass_closed_band_abstention μ g0 ga h0 ha κ0 s
    observation policy hD beta W hsub hi he hband alpha hcontrol P hP
