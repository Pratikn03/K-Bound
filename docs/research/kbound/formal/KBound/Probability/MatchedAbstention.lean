import KBound.Impossibility
import KBound.Probability.MeasureFrontier
import Mathlib.Probability.Kernel.Composition.MeasureComp
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Measure.Prod

/-!
# Randomized abstention under identical evidence laws

The decision rule is an arbitrary measurable Markov kernel on the three-action
space. Thus its randomization is common across worlds and depends only on the
observed evidence; equality of its action laws is derived, not assumed. Opposite
strict benefits and the zero-benefit boundary are treated separately. The laws
of the original observations may live on different measurable spaces.
-/

namespace KBound

open Decision MeasureTheory ProbabilityTheory Set
open scoped ENNReal

instance decisionMeasurableSpace : MeasurableSpace Decision := ⊤

instance decisionMeasurableSingletonClass : MeasurableSingletonClass Decision :=
  ⟨fun _ => trivial⟩

/-- A genuine probability-law trichotomy bound, rather than arithmetic on
uninterpreted action weights. -/
theorem decision_abstention_ge_of_directional_bounds
    (ρ : Measure Decision) [IsProbabilityMeasure ρ] {alpha : ℝ≥0∞}
    (ha : ρ {adapt} ≤ alpha) (hf : ρ {freeze} ≤ alpha) :
    1 - 2 * alpha ≤ ρ {abstain} := by
  have hcomp : ({adapt} ∪ {freeze} : Set Decision)ᶜ = {abstain} := by
    ext a
    cases a <;> simp
  have hu : ρ ({adapt} ∪ {freeze}) ≤ 2 * alpha := by
    calc
      ρ ({adapt} ∪ {freeze}) ≤ ρ {adapt} + ρ {freeze} := measure_union_le _ _
      _ ≤ alpha + alpha := add_le_add ha hf
      _ = 2 * alpha := (two_mul alpha).symm
  calc
    1 - 2 * alpha ≤ 1 - ρ ({adapt} ∪ {freeze}) := tsub_le_tsub_left hu 1
    _ = ρ {abstain} := by
      rw [← prob_compl_eq_one_sub
        ((measurableSet_singleton adapt).union (measurableSet_singleton freeze)), hcomp]

/-- Applying the same randomized measurable decision kernel to identical
observable laws gives identical action laws. -/
theorem matched_evidence_randomized_action_law
    {Ω₁ Ω₂ E : Type*} [MeasurableSpace Ω₁] [MeasurableSpace Ω₂] [MeasurableSpace E]
    {P : Measure Ω₁} {Q : Measure Ω₂} {W₁ : Ω₁ → E} {W₂ : Ω₂ → E}
    (κ : Kernel E Decision)
    (hevidence : P.map W₁ = Q.map W₂) :
    κ ∘ₘ P.map W₁ = κ ∘ₘ Q.map W₂ := by
  rw [hevidence]

/-- Full probability form of the paper's matched-evidence corollary. Directional
error controls are supplied in their respective negative/positive worlds; their
conversion into common-law action bounds is part of the proof. -/
theorem randomized_matched_evidence_abstention
    {Ω₁ Ω₂ E : Type*} [MeasurableSpace Ω₁] [MeasurableSpace Ω₂] [MeasurableSpace E]
    {P : Measure Ω₁} {Q : Measure Ω₂} [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {W₁ : Ω₁ → E} {W₂ : Ω₂ → E}
    (hW₁ : Measurable W₁) (_hW₂ : Measurable W₂)
    (κ : Kernel E Decision) [IsMarkovKernel κ]
    (hevidence : P.map W₁ = Q.map W₂)
    {delta₁ delta₂ : ℝ} (hneg : delta₁ < 0) (hpos : 0 < delta₂)
    {alpha : ℝ≥0∞}
    (hfa : (κ ∘ₘ P.map W₁) {a | a = adapt ∧ delta₁ ≤ 0} ≤ alpha)
    (hff : (κ ∘ₘ Q.map W₂) {a | a = freeze ∧ 0 ≤ delta₂} ≤ alpha) :
    1 - 2 * alpha ≤ (κ ∘ₘ P.map W₁) {abstain} ∧
    1 - 2 * alpha ≤ (κ ∘ₘ Q.map W₂) {abstain} := by
  letI : IsProbabilityMeasure (P.map W₁) := P.isProbabilityMeasure_map hW₁.aemeasurable
  have heq := matched_evidence_randomized_action_law κ hevidence
  have ha : (κ ∘ₘ P.map W₁) {adapt} ≤ alpha := by simpa [hneg.le] using hfa
  have hf : (κ ∘ₘ P.map W₁) {freeze} ≤ alpha := by
    rw [heq]
    simpa [hpos.le] using hff
  have hab := decision_abstention_ge_of_directional_bounds (κ ∘ₘ P.map W₁) ha hf
  exact ⟨hab, by rwa [← heq]⟩

/-- At the zero-benefit boundary both commitments count as directional errors,
so no second opposite-sign world is needed. -/
theorem randomized_zero_benefit_abstention
    {E : Type*} [MeasurableSpace E] (μ : Measure E) [IsProbabilityMeasure μ]
    (κ : Kernel E Decision) [IsMarkovKernel κ] {alpha : ℝ≥0∞}
    (hfa : (κ ∘ₘ μ) {a | a = adapt ∧ (0 : ℝ) ≤ 0} ≤ alpha)
    (hff : (κ ∘ₘ μ) {a | a = freeze ∧ (0 : ℝ) ≤ 0} ≤ alpha) :
    1 - 2 * alpha ≤ (κ ∘ₘ μ) {abstain} := by
  apply decision_abstention_ge_of_directional_bounds (κ ∘ₘ μ)
  · simpa using hfa
  · simpa using hff

/-- The zero-world bound transfers to every world in its evidence fibre. This
is the probability clause used on the full closed band, including M=β=0. -/
theorem randomized_matched_zero_boundary_abstention
    {E : Type*} [MeasurableSpace E] {μ ν : Measure E} [IsProbabilityMeasure μ]
    (κ : Kernel E Decision) [IsMarkovKernel κ] (hevidence : μ = ν)
    {alpha : ℝ≥0∞}
    (hfa : (κ ∘ₘ μ) {a | a = adapt ∧ (0 : ℝ) ≤ 0} ≤ alpha)
    (hff : (κ ∘ₘ μ) {a | a = freeze ∧ (0 : ℝ) ≤ 0} ≤ alpha) :
    1 - 2 * alpha ≤ (κ ∘ₘ ν) {abstain} := by
  rw [← hevidence]
  exact randomized_zero_benefit_abstention μ κ hfa hff

/-- Strict pointwise soundness at benefit zero permits only abstention. -/
theorem zero_benefit_sound_iff_abstain (a : Decision) :
    SoundForBenefit a 0 ↔ a = abstain := by
  cases a <;> simp [SoundForBenefit]

/-- The full measurable correctness-field class supplies the zero-benefit
world needed by the randomized closed-band bound; its existence is derived
from the label-kernel construction, not supplied as an abstract richness axiom.
The rule may use any measurable label-free function of the input. -/
theorem measurable_closed_band_randomized_abstention
    {X Y E : Type*} [MeasurableSpace X] [MeasurableSpace Y] [MeasurableSpace E]
    [MeasurableSingletonClass Y] [MeasurableEq Y]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) {M beta : ℝ} (hband : |M| ≤ beta)
    (g : X → E) (hg : Measurable g)
    (κ : Kernel E Decision) [IsMarkovKernel κ] {alpha : ℝ≥0∞}
    (hfa : ∀ η : CorrectnessField X,
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ beta →
      (κ ∘ₘ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η).map (fun xy => g xy.1))
        {a | a = adapt ∧ populationBenefit f₀ fₐ
          (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η) ≤ 0} ≤ alpha)
    (hff : ∀ η : CorrectnessField X,
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ beta →
      (κ ∘ₘ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η).map (fun xy => g xy.1))
        {a | a = freeze ∧ 0 ≤ populationBenefit f₀ fₐ
          (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η)} ≤ alpha) :
    1 - 2 * alpha ≤ (κ ∘ₘ μ.map g) {abstain} := by
  letI : IsProbabilityMeasure (μ.map g) := μ.isProbabilityMeasure_map hg.aemeasurable
  obtain ⟨η, _, hbudget, hzero, _, _, hevidence⟩ :=
    measurable_closed_band_zero_target (E := E) μ f₀ fₐ h₀ hₐ κ₀ hD hband
  have ha := hfa η hbudget
  have hf := hff η hbudget
  rw [hevidence g hg, hzero] at ha hf
  exact randomized_zero_benefit_abstention (μ.map g) κ ha hf

/-- Pointwise strict soundness over the actual full target class forces the
unique remaining action throughout the closed band, including its endpoints. -/
theorem measurable_closed_band_pointwise_abstention
    {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    [MeasurableSingletonClass Y] [MeasurableEq Y]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) {M beta : ℝ} (hband : |M| ≤ beta)
    (a : Decision)
    (hsound : ∀ η : CorrectnessField X,
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ beta →
      SoundForBenefit a (populationBenefit f₀ fₐ
        (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η))) :
    a = abstain := by
  obtain ⟨η, _, hbudget, hzero, _⟩ :=
    measurable_closed_band_zero_target (E := Unit) μ f₀ fₐ h₀ hₐ κ₀ hD hband
  have hs := hsound η hbudget
  rw [hzero] at hs
  exact (zero_benefit_sound_iff_abstain a).mp hs

end KBound
