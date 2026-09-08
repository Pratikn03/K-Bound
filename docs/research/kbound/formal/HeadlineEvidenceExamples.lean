import KBound

open MeasureTheory ProbabilityTheory Set KBound KBound.JointKernelScore
open KBound.ActualWorldFrontier
open scoped ENNReal ProbabilityTheory

-- RED: the actual-class/evidence assembly is absent from the current public root.
#check KBound.ActualWorldEvidence.augmentedLaw_iid
#check KBound.ActualWorldEvidence.actual_open_band_matched_targets
#check KBound.ActualWorldEvidence.actual_closed_band_abstention
#check KBound.ActualWorldEvidence.actual_augmented_fibre_strict_direction_iff

-- The final iff must use compatibility with a fixed actual augmented law.
example {X Z : Type*} [MeasurableSpace X] [MeasurableSpace Z]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : (Q : Measure (X × Bool)).fst = μ) :
    ((∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P ∧
        KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score P =
          KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score Q →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∨
      (∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P ∧
        KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score P =
          KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score Q →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0)) ↔
      beta < |scoreMargin μ {x | f0 x ≠ fa x} score| :=
  KBound.ActualWorldEvidence.actual_augmented_fibre_strict_direction_iff μ f0 fa h0 ha κ0
    score observation hD beta hb Q hQ

-- This catches replacing genuine score-class witnesses by free signed numbers.
example {X Z : Type*} [MeasurableSpace X] [MeasurableSpace Z]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hband : |scoreMargin μ {x | f0 x ≠ fa x} score| < beta) :
    ∃ Ppos Pneg : ProbabilityMeasure (X × Bool),
      admissible μ f0 fa ha score beta Ppos ∧
      admissible μ f0 fa ha score beta Pneg ∧
      0 < populationBenefit f0 fa (Ppos : Measure (X × Bool)) ∧
      populationBenefit f0 fa (Pneg : Measure (X × Bool)) < 0 ∧
      KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score Ppos =
        KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score Pneg :=
  KBound.ActualWorldEvidence.actual_open_band_matched_targets μ f0 fa h0 ha κ0
    score observation hD beta hband

-- The zero-benefit world must force both error bounds, even at beta = M = 0.
example {X Z : Type*} [MeasurableSpace X] [MeasurableSpace Z]
    [MeasurableSpace Decision] [MeasurableSingletonClass Decision]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (policy : Kernel (Z × ℝ) Decision) [IsMarkovKernel policy]
    (hD : 0 < μ.real {x | f0 x ≠ fa x})
    (hm : scoreMargin μ {x | f0 x ≠ fa x} score = 0) (alpha : ENNReal)
    (hcontrol : ∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score 0 P →
      (policy ∘ₘ (KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score P :
        Measure (Z × ℝ))) {a | a = Decision.adapt ∧
          populationBenefit f0 fa (P : Measure (X × Bool)) ≤ 0} ≤ alpha ∧
      (policy ∘ₘ (KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score P :
        Measure (Z × ℝ))) {a | a = Decision.freeze ∧
          0 ≤ populationBenefit f0 fa (P : Measure (X × Bool))} ≤ alpha)
    (P : ProbabilityMeasure (X × Bool)) (hP : admissible μ f0 fa ha score 0 P) :
    1 - 2 * alpha ≤ (policy ∘ₘ
      (KBound.ActualWorldEvidence.augmentedLaw observation f0 fa score P :
        Measure (Z × ℝ))) {Decision.abstain} := by
  apply KBound.ActualWorldEvidence.actual_closed_band_abstention μ f0 fa h0 ha κ0
    score observation policy hD 0 (by rw [hm]; norm_num) alpha hcontrol P hP

-- The concrete iid bridge must concern the actual joint sample, not a free law.
example {X : Type*} [MeasurableSpace X]
    (f0 fa : X → Bool) (score : CorrectnessField X)
    (P : ProbabilityMeasure (X × Bool)) :
    (KBound.ActualWorldEvidence.augmentedLaw
      (KBound.ActualWorldEvidence.iidObservation
        (fun b : Fin 3 → X => (b 0, b 2))
        ((measurable_pi_apply 0).prodMk (measurable_pi_apply 2)))
      f0 fa score P : Measure ((X × X) × ℝ)) =
      (Measure.pi (fun _ : Fin 3 => (P : Measure (X × Bool)))).map
        (fun b => (((b 0).1, (b 2).1),
          scoreMargin (P : Measure (X × Bool)).fst {x | f0 x ≠ fa x} score)) :=
  KBound.ActualWorldEvidence.augmentedLaw_iid _ _ f0 fa score P

-- A specified perfectly dependent two-input experiment is allowed: draw X once
-- and return (X,X). This test must not require iid coordinate independence.
example {X : Type*} [MeasurableSpace X]
    (f0 fa : X → Bool) (score : CorrectnessField X)
    (P Q : ProbabilityMeasure (X × Bool))
    (h : (P : Measure (X × Bool)).fst = (Q : Measure (X × Bool)).fst) :
    KBound.ActualWorldEvidence.augmentedLaw
      (fun ν : ProbabilityMeasure X =>
        ν.map (measurable_id.prodMk measurable_id).aemeasurable) f0 fa score P =
    KBound.ActualWorldEvidence.augmentedLaw
      (fun ν : ProbabilityMeasure X =>
        ν.map (measurable_id.prodMk measurable_id).aemeasurable) f0 fa score Q :=
  KBound.ActualWorldEvidence.augmentedLaw_eq _ f0 fa score h

-- A proper disagreement region must not overwrite the supplied outside kernel.
example (μ : Measure Bool) [IsProbabilityMeasure μ]
    (κ0 : Kernel Bool Bool) [IsMarkovKernel κ0] (score : CorrectnessField Bool)
    (hD : 0 < μ.real {x | false ≠ x}) (beta : ℝ)
    (hb : |scoreMargin μ {x | false ≠ x} score| < beta) :
    ∃ ηpos ηneg : CorrectnessField Bool,
      admissible μ (fun _ => false) id measurable_id score beta
        (fieldWorld μ (fun _ => false) id measurable_const measurable_id κ0 ηpos) ∧
      admissible μ (fun _ => false) id measurable_id score beta
        (fieldWorld μ (fun _ => false) id measurable_const measurable_id κ0 ηneg) ∧
      targetLabelKernel (fun _ => false) id measurable_const measurable_id
        κ0 ηpos.value ηpos.measurable_value false = κ0 false ∧
      targetLabelKernel (fun _ => false) id measurable_const measurable_id
        κ0 ηneg.value ηneg.measurable_value false = κ0 false := by
  obtain ⟨ηpos, ηneg, hp, hn, _, _, hoffp, hoffn⟩ :=
    KBound.ActualWorldEvidence.actual_open_band_field_witnesses μ (fun _ => false) id
      measurable_const measurable_id κ0 score hD beta hb
  exact ⟨ηpos, ηneg, hp, hn, hoffp false rfl, hoffn false rfl⟩

local notation "mu" => (Measure.dirac () : Measure Unit)
local notation "f0" => (fun _ : Unit => false)
local notation "fa" => (fun _ : Unit => true)
local notation "kernel0" => (Kernel.const Unit (Measure.dirac false))

-- Positive boundary: matched zero, not two strict opposite signs.
example {Z : Type*} [MeasurableSpace Z]
    (observation : ProbabilityMeasure Unit → ProbabilityMeasure Z) :
    ∃ Pzero : ProbabilityMeasure (Unit × Bool),
      admissible mu f0 fa measurable_const
        (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) (1 / 4) Pzero ∧
      populationBenefit f0 fa (Pzero : Measure (Unit × Bool)) = 0 ∧
      ∀ P : ProbabilityMeasure (Unit × Bool),
        admissible mu f0 fa measurable_const
          (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) (1 / 4) P →
        KBound.ActualWorldEvidence.augmentedLaw observation f0 fa
          (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) P =
        KBound.ActualWorldEvidence.augmentedLaw observation f0 fa
          (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) Pzero := by
  apply KBound.ActualWorldEvidence.actual_closed_band_matched_zero mu f0 fa
    measurable_const measurable_const kernel0 _ observation (by norm_num [Measure.real])
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Negative boundary is included with the opposite sign of M.
example {Z : Type*} [MeasurableSpace Z]
    (observation : ProbabilityMeasure Unit → ProbabilityMeasure Z) :
    ∃ Pzero : ProbabilityMeasure (Unit × Bool),
      admissible mu f0 fa measurable_const
        (CorrectnessField.constant (1 / 4) (by norm_num) (by norm_num)) (1 / 4) Pzero ∧
      populationBenefit f0 fa (Pzero : Measure (Unit × Bool)) = 0 ∧
      ∀ P : ProbabilityMeasure (Unit × Bool),
        admissible mu f0 fa measurable_const
          (CorrectnessField.constant (1 / 4) (by norm_num) (by norm_num)) (1 / 4) P →
        KBound.ActualWorldEvidence.augmentedLaw observation f0 fa
          (CorrectnessField.constant (1 / 4) (by norm_num) (by norm_num)) P =
        KBound.ActualWorldEvidence.augmentedLaw observation f0 fa
          (CorrectnessField.constant (1 / 4) (by norm_num) (by norm_num)) Pzero := by
  apply KBound.ActualWorldEvidence.actual_closed_band_matched_zero mu f0 fa
    measurable_const measurable_const kernel0 _ observation (by norm_num [Measure.real])
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Large budgets still use actual probabilities, including endpoint scores.
example {Z : Type*} [MeasurableSpace Z]
    (observation : ProbabilityMeasure Unit → ProbabilityMeasure Z) :
    ∃ Ppos Pneg : ProbabilityMeasure (Unit × Bool),
      admissible mu f0 fa measurable_const
        (CorrectnessField.constant 1 (by norm_num) (by norm_num)) 100 Ppos ∧
      admissible mu f0 fa measurable_const
        (CorrectnessField.constant 1 (by norm_num) (by norm_num)) 100 Pneg ∧
      0 < populationBenefit f0 fa (Ppos : Measure (Unit × Bool)) ∧
      populationBenefit f0 fa (Pneg : Measure (Unit × Bool)) < 0 ∧
      KBound.ActualWorldEvidence.augmentedLaw observation f0 fa
        (CorrectnessField.constant 1 (by norm_num) (by norm_num)) Ppos =
      KBound.ActualWorldEvidence.augmentedLaw observation f0 fa
        (CorrectnessField.constant 1 (by norm_num) (by norm_num)) Pneg := by
  apply KBound.ActualWorldEvidence.actual_open_band_matched_targets mu f0 fa
    measurable_const measurable_const kernel0 _ observation (by norm_num [Measure.real])
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]
