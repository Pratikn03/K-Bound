import KBound.Probability.CompoundCoverage
import KBound.Probability.MeasureSwap

/-! A fixed predictor pair applied to genuinely independent fresh labeled inputs
supplies the sampling conditions of CompoundCoverage. The signed sample score is
the paired accuracy difference, and its mean is the actual population risk gain. -/
namespace KBound
open MeasureTheory ProbabilityTheory Set
open scoped ENNReal ProbabilityTheory

variable {Ω X Y ι : Type*} [MeasurableSpace Ω] [MeasurableSpace X]
  [MeasurableSpace Y] [MeasurableEq Y] [DecidableEq Y]

omit [MeasurableSpace Ω] [MeasurableSpace X] [MeasurableSpace Y] [MeasurableEq Y] in
/-- Accuracy improvement and reduction of zero-one loss are exactly the same
paired score, including agreement and both-correct/both-incorrect cases. -/
theorem zeroOneBenefit_eq_paired_accuracy (f₀ fₐ : X → Y) (xy : X × Y) :
    zeroOneBenefit f₀ fₐ xy =
      (if fₐ xy.1 = xy.2 then 1 else 0) - (if f₀ xy.1 = xy.2 then 1 else 0) := by
  classical
  by_cases ha : xy.2 = fₐ xy.1
  · by_cases h0 : xy.2 = f₀ xy.1
    · have h : f₀ xy.1 = fₐ xy.1 := h0.symm.trans ha
      simp [zeroOneBenefit, ha, h0, h, eq_comm]
    · have h : ¬ f₀ xy.1 = fₐ xy.1 := by
        intro h
        exact h0 (h.symm ▸ ha)
      simp [zeroOneBenefit, ha, h0, h, eq_comm]
  · by_cases h0 : xy.2 = f₀ xy.1
    · have h : ¬ f₀ xy.1 = fₐ xy.1 := by
        intro h
        exact ha (h0.trans h)
      simp [zeroOneBenefit, ha, h0, h, eq_comm]
    · by_cases h : f₀ xy.1 = fₐ xy.1 <;>
        simp [zeroOneBenefit, ha, h0, h, eq_comm]

/-- Raw independent labeled draws with common law P give all scalar sampling
hypotheses needed by the compound bound. Independence after applying the frozen
pair and equality of the expectation to population risk benefit are conclusions. -/
theorem fresh_paired_scores_properties
    {μ : Measure Ω} (T : ι → Ω → X × Y) (hT : ∀ i, Measurable (T i))
    (hindep : iIndepFun T μ) (P : Measure (X × Y)) (hlaw : ∀ i, μ.map (T i) = P)
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ) :
    iIndepFun (fun i ω => zeroOneBenefit f₀ fₐ (T i ω)) μ ∧
    (∀ i, Measurable (fun ω => zeroOneBenefit f₀ fₐ (T i ω))) ∧
    (∀ i ω, zeroOneBenefit f₀ fₐ (T i ω) ∈ Icc (-1 : ℝ) 1) ∧
    (∀ i, (∫ ω, zeroOneBenefit f₀ fₐ (T i ω) ∂μ) = populationBenefit f₀ fₐ P) := by
  have hs := measurable_zeroOneBenefit f₀ fₐ h₀ hₐ
  refine ⟨hindep.comp (fun _ => zeroOneBenefit f₀ fₐ) (fun _ => hs),
    (fun i => hs.comp (hT i)), ?_, ?_⟩
  · intro i ω
    exact abs_le.mp (abs_zeroOneBenefit_le_one f₀ fₐ (T i ω))
  · intro i
    rw [← integral_map (hT i).aemeasurable hs.aestronglyMeasurable, hlaw i]
    rfl

end KBound
