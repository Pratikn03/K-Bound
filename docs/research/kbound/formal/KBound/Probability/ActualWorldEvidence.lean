import KBound.Probability.ActualWorldFrontier
import KBound.Probability.EvidenceTransport

/-!
# Actual score-class witnesses and augmented label-free evidence

A fixed observation experiment maps the input probability law to an evidence
probability law. This factorization is explicit: it does not follow merely from
equal one-input marginals for arbitrary world-dependent batch couplings. Finite
iid input-batch observations have a concrete realization as the pushforward of
the target's iid joint sample, proved below. Other fixed marginal-based sampling
designs are allowed without imposing iid on the generic impossibility theorem.

The augmentation is the actual margin of the target input marginal. The rule
is one fixed Markov kernel on this augmented evidence space, and the error
premises measure its actual directional events in every admissible target law.
-/

namespace KBound.ActualWorldEvidence

open MeasureTheory ProbabilityTheory Set JointKernelScore ActualWorldFrontier
open scoped ENNReal ProbabilityTheory

variable {X Z : Type*} [MeasurableSpace X] [MeasurableSpace Z]

noncomputable def inputLaw (P : ProbabilityMeasure (X × Bool)) : ProbabilityMeasure X :=
  P.map measurable_fst.aemeasurable

noncomputable def augmentedLaw
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (f0 fa : X → Bool) (score : CorrectnessField X)
    (P : ProbabilityMeasure (X × Bool)) : ProbabilityMeasure (Z × ℝ) :=
  (observation (inputLaw P)).map
    (measurable_id.prodMk (measurable_const (a :=
      scoreMargin (P : Measure (X × Bool)).fst {x | f0 x ≠ fa x} score))).aemeasurable

/-- Equality includes the actual margin, not an independently supplied M. -/
theorem augmentedLaw_eq (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (f0 fa : X → Bool) (score : CorrectnessField X)
    {P Q : ProbabilityMeasure (X × Bool)}
    (hinput : (P : Measure (X × Bool)).fst = (Q : Measure (X × Bool)).fst) :
    augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q := by
  have hi : inputLaw P = inputLaw Q := Subtype.ext hinput
  simp only [augmentedLaw, hi, hinput]

noncomputable def iidObservation {ι : Type*} [Fintype ι]
    (evidence : (ι → X) → Z) (he : Measurable evidence)
    (μ : ProbabilityMeasure X) : ProbabilityMeasure Z :=
  ProbabilityMeasure.map
    (⟨Measure.pi (fun _ : ι => (μ : Measure X)), inferInstance⟩ :
      ProbabilityMeasure (ι → X)) he.aemeasurable

/-- The observation model is realized by an actual iid sample from P, including
arbitrary input/label dependence within each observation. -/
theorem iidObservation_realization {ι : Type*} [Fintype ι]
    (evidence : (ι → X) → Z) (he : Measurable evidence)
    (P : ProbabilityMeasure (X × Bool)) :
    (iidObservation evidence he (inputLaw P) : Measure Z) =
      (Measure.pi (fun _ : ι => (P : Measure (X × Bool)))).map
        (fun batch => evidence (fun i => (batch i).1)) := by
  have hb : Measurable (fun batch : ι → X × Bool => fun i => (batch i).1) :=
    measurable_pi_lambda _ fun i => measurable_fst.comp (measurable_pi_apply i)
  letI : IsProbabilityMeasure ((P : Measure (X × Bool)).map Prod.fst) :=
    Measure.isProbabilityMeasure_map measurable_fst.aemeasurable
  have hp := Measure.pi_map_pi (μ := fun _ : ι => (P : Measure (X × Bool)))
    (fun _ => measurable_fst.aemeasurable)
  change (Measure.pi (fun _ : ι => (P : Measure (X × Bool)).map Prod.fst)).map evidence = _
  rw [← hp, Measure.map_map he hb]
  rfl

theorem augmentedLaw_iid {ι : Type*} [Fintype ι]
    (evidence : (ι → X) → Z) (he : Measurable evidence)
    (f0 fa : X → Bool) (score : CorrectnessField X)
    (P : ProbabilityMeasure (X × Bool)) :
    (augmentedLaw (iidObservation evidence he) f0 fa score P : Measure (Z × ℝ)) =
      (Measure.pi (fun _ : ι => (P : Measure (X × Bool)))).map
        (fun batch => (evidence (fun i => (batch i).1),
          scoreMargin (P : Measure (X × Bool)).fst {x | f0 x ≠ fa x} score)) := by
  have hb : Measurable (fun batch : ι → X × Bool => fun i => (batch i).1) :=
    measurable_pi_lambda _ fun i => measurable_fst.comp (measurable_pi_apply i)
  change (iidObservation evidence he (inputLaw P) : Measure Z).map _ = _
  rw [iidObservation_realization]
  exact Measure.map_map
    (measurable_id.prodMk (measurable_const (a :=
      scoreMargin (P : Measure (X × Bool)).fst {x | f0 x ≠ fa x} score))) (he.comp hb)

/-- Interior witnesses belong to the actual residual-defined class and retain
the same specified label kernel off disagreement. Probability clipping is kept. -/
theorem actual_open_band_field_witnesses (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hband : |scoreMargin μ {x | f0 x ≠ fa x} score| < beta) :
    ∃ ηpos ηneg : CorrectnessField X,
      admissible μ f0 fa ha score beta (fieldWorld μ f0 fa h0 ha κ0 ηpos) ∧
      admissible μ f0 fa ha score beta (fieldWorld μ f0 fa h0 ha κ0 ηneg) ∧
      0 < populationBenefit f0 fa (fieldWorld μ f0 fa h0 ha κ0 ηpos : Measure (X × Bool)) ∧
      populationBenefit f0 fa (fieldWorld μ f0 fa h0 ha κ0 ηneg : Measure (X × Bool)) < 0 ∧
      (∀ x, f0 x = fa x →
        targetLabelKernel f0 fa h0 ha κ0 ηpos.value ηpos.measurable_value x = κ0 x) ∧
      (∀ x, f0 x = fa x →
        targetLabelKernel f0 fa h0 ha κ0 ηneg.value ηneg.measurable_value x = κ0 x) := by
  obtain ⟨ηpos, ηneg, _, _, hbp, hbn, hp, hn, hoffp, hoffn, hfp, hfn, _⟩ :=
    measurable_open_band_opposite_targets (E := Unit) μ f0 fa h0 ha κ0 hD hband
  refine ⟨ηpos, ηneg, ⟨hfp, ?_⟩, ⟨hfn, ?_⟩, hp, hn, hoffp, hoffn⟩
  · rw [fieldWorld_residual μ f0 fa h0 ha κ0 score ηpos hD]
    exact hbp
  · rw [fieldWorld_residual μ f0 fa h0 ha κ0 score ηneg hD]
    exact hbn

/-- Actual opposite-sign target laws with identical augmented evidence. -/
theorem actual_open_band_matched_targets (μ : Measure X) [IsProbabilityMeasure μ]
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
      augmentedLaw observation f0 fa score Ppos = augmentedLaw observation f0 fa score Pneg := by
  obtain ⟨ηpos, ηneg, hpos, hneg, hp, hn, _, _⟩ :=
    actual_open_band_field_witnesses μ f0 fa h0 ha κ0 score hD beta hband
  exact ⟨fieldWorld μ f0 fa h0 ha κ0 ηpos, fieldWorld μ f0 fa h0 ha κ0 ηneg,
    hpos, hneg, hp, hn, augmentedLaw_eq observation f0 fa score (hpos.1.trans hneg.1.symm)⟩

/-- A zero target throughout the closed band shares the evidence of every
admissible target, including the two boundaries and beta = M = 0. -/
theorem actual_closed_band_matched_zero (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hband : |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta) :
    ∃ Pzero : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta Pzero ∧
      populationBenefit f0 fa (Pzero : Measure (X × Bool)) = 0 ∧
      ∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Pzero := by
  obtain ⟨Pzero, hzero, hz⟩ := actual_closed_band_zero_target μ f0 fa h0 ha κ0 score hD beta hband
  exact ⟨Pzero, hzero, hz, fun P hP =>
    augmentedLaw_eq observation f0 fa score (hP.1.trans hzero.1.symm)⟩

/-- The strict-direction iff on the actual fixed augmented-evidence fibre.
The representative Q supplies a realized input law, not a freely chosen action law. -/
theorem actual_augmented_fibre_strict_direction_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : (Q : Measure (X × Bool)).fst = μ) :
    ((∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P ∧
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∨
      (∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P ∧
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0)) ↔
      beta < |scoreMargin μ {x | f0 x ≠ fa x} score| := by
  have heq (P : ProbabilityMeasure (X × Bool)) (hP : admissible μ f0 fa ha score beta P) :=
    augmentedLaw_eq observation f0 fa score (hP.1.trans hQ.symm)
  rw [← actual_strict_direction_iff μ f0 fa h0 ha κ0 score hD beta hb]
  constructor
  · rintro (hp | hn)
    · exact Or.inl (fun P hP => hp P ⟨hP, heq P hP⟩)
    · exact Or.inr (fun P hP => hn P ⟨hP, heq P hP⟩)
  · rintro (hp | hn)
    · exact Or.inl (fun P hP => hp P hP.1)
    · exact Or.inr (fun P hP => hn P hP.1)

section Decisions

variable [MeasurableSpace Decision] [MeasurableSingletonClass Decision]

/-- Uniform actual directional-error bounds force abstention under the shared
augmented-evidence law, for every world in the actual admissible class. -/
theorem actual_closed_band_abstention (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (policy : Kernel (Z × ℝ) Decision) [IsMarkovKernel policy]
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hband : |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta) (alpha : ENNReal)
    (hcontrol : ∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
      (policy ∘ₘ (augmentedLaw observation f0 fa score P : Measure (Z × ℝ)))
        {a | a = Decision.adapt ∧ populationBenefit f0 fa (P : Measure (X × Bool)) ≤ 0} ≤ alpha ∧
      (policy ∘ₘ (augmentedLaw observation f0 fa score P : Measure (Z × ℝ)))
        {a | a = Decision.freeze ∧ 0 ≤ populationBenefit f0 fa (P : Measure (X × Bool))} ≤ alpha)
    (P : ProbabilityMeasure (X × Bool)) (hP : admissible μ f0 fa ha score beta P) :
    1 - 2 * alpha ≤ (policy ∘ₘ
      (augmentedLaw observation f0 fa score P : Measure (Z × ℝ))) {Decision.abstain} := by
  obtain ⟨Pzero, hzero, hz, heq⟩ :=
    actual_closed_band_matched_zero μ f0 fa h0 ha κ0 score observation hD beta hband
  obtain ⟨hpa, hpf⟩ := hcontrol Pzero hzero
  simp only [hz, le_refl, and_true, setOf_eq_eq_singleton] at hpa hpf
  rw [heq P hP]
  exact RandomizedActionLaw.abstention_lower_bound hpa hpf

end Decisions
end KBound.ActualWorldEvidence

#print axioms KBound.ActualWorldEvidence.augmentedLaw_eq
#print axioms KBound.ActualWorldEvidence.iidObservation_realization
#print axioms KBound.ActualWorldEvidence.augmentedLaw_iid
#print axioms KBound.ActualWorldEvidence.actual_open_band_field_witnesses
#print axioms KBound.ActualWorldEvidence.actual_open_band_matched_targets
#print axioms KBound.ActualWorldEvidence.actual_closed_band_matched_zero
#print axioms KBound.ActualWorldEvidence.actual_augmented_fibre_strict_direction_iff
#print axioms KBound.ActualWorldEvidence.actual_closed_band_abstention
