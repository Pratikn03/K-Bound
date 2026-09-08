import KBound.Impossibility
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Measure.Map

/-! Full action-law interface for randomized forced abstention. Randomness and
observations may be combined in the world sample spaces; equality of the induced
action laws is an explicit premise, not inferred from equal seed marginals. -/

namespace KBound
namespace RandomizedActionLaw

open MeasureTheory Decision

variable [MeasurableSpace Decision] [MeasurableSingletonClass Decision]

/-- A probability law on the three actions has abstention mass at least one
minus the two directional masses. ENNReal subtraction is truncated at zero. -/
theorem abstention_lower_bound {ρ : Measure Decision} [IsProbabilityMeasure ρ]
    {alpha : ENNReal} (ha : ρ {adapt} ≤ alpha) (hf : ρ {freeze} ≤ alpha) :
    1 - 2 * alpha ≤ ρ {abstain} := by
  have hcompl : ({abstain} : Set Decision)ᶜ = {adapt} ∪ {freeze} := by
    ext a
    cases a <;> simp
  have hmass : ρ ({abstain} : Set Decision)ᶜ ≤ 2 * alpha := by
    rw [hcompl]
    calc ρ ({adapt} ∪ {freeze}) ≤ ρ {adapt} + ρ {freeze} := measure_union_le _ _
      _ ≤ alpha + alpha := add_le_add ha hf
      _ = 2 * alpha := (two_mul alpha).symm
  calc 1 - 2 * alpha ≤ 1 - ρ ({abstain} : Set Decision)ᶜ :=
      tsub_le_tsub_left hmass 1
    _ = ρ {abstain} := by
      rw [← prob_compl_eq_one_sub (measurableSet_singleton abstain).compl]
      simp

/-- Opposite-sign worlds with a common action distribution and worldwise
false-direction bounds force abstention in both worlds. -/
theorem common_law_forces_abstention
    {ρneg ρpos : Measure Decision} [IsProbabilityMeasure ρneg]
    {deltaNeg deltaPos : ℝ} {alpha : ENNReal}
    (hneg : deltaNeg < 0) (hpos : 0 < deltaPos)
    (hcommon : ρneg = ρpos)
    (hfa : ρneg {a | a = adapt ∧ deltaNeg ≤ 0} ≤ alpha)
    (hff : ρpos {a | a = freeze ∧ 0 ≤ deltaPos} ≤ alpha) :
    1 - 2 * alpha ≤ ρneg {abstain} ∧
      1 - 2 * alpha ≤ ρpos {abstain} := by
  have ha : ρneg {adapt} ≤ alpha := by simpa [hneg.le] using hfa
  have hf : ρneg {freeze} ≤ alpha := by
    rw [hcommon]
    simpa [hpos.le] using hff
  have hb := abstention_lower_bound ha hf
  exact ⟨hb, hcommon ▸ hb⟩

/-- Sample-space interface: the measurable randomized rules can be different
functions on different spaces. Their pushforward laws must agree. The two error
bounds are probabilities in the original worlds, not free arithmetic variables. -/
theorem randomized_rules_force_abstention
    {Ωneg Ωpos : Type*} [MeasurableSpace Ωneg] [MeasurableSpace Ωpos]
    {μneg : Measure Ωneg} {μpos : Measure Ωpos}
    [IsProbabilityMeasure μneg] [IsProbabilityMeasure μpos]
    {ruleNeg : Ωneg → Decision} {rulePos : Ωpos → Decision}
    (hmneg : Measurable ruleNeg) (hmpos : Measurable rulePos)
    {deltaNeg deltaPos : ℝ} {alpha : ENNReal}
    (hneg : deltaNeg < 0) (hpos : 0 < deltaPos)
    (hcommon : μneg.map ruleNeg = μpos.map rulePos)
    (hfa : μneg {ω | ruleNeg ω = adapt ∧ deltaNeg ≤ 0} ≤ alpha)
    (hff : μpos {ω | rulePos ω = freeze ∧ 0 ≤ deltaPos} ≤ alpha) :
    1 - 2 * alpha ≤ μneg {ω | ruleNeg ω = abstain} ∧
      1 - 2 * alpha ≤ μpos {ω | rulePos ω = abstain} := by
  letI : IsProbabilityMeasure (μneg.map ruleNeg) := Measure.isProbabilityMeasure_map hmneg.aemeasurable
  have ha : (μneg.map ruleNeg) {a | a = adapt ∧ deltaNeg ≤ 0} ≤ alpha := by
    simp only [hneg.le, and_true]
    change (μneg.map ruleNeg) {adapt} ≤ alpha
    rw [Measure.map_apply hmneg (measurableSet_singleton adapt)]
    simpa only [hneg.le, and_true] using hfa
  have hf : (μpos.map rulePos) {a | a = freeze ∧ 0 ≤ deltaPos} ≤ alpha := by
    simp only [hpos.le, and_true]
    change (μpos.map rulePos) {freeze} ≤ alpha
    rw [Measure.map_apply hmpos (measurableSet_singleton freeze)]
    simpa only [hpos.le, and_true] using hff
  have hb := common_law_forces_abstention hneg hpos hcommon ha hf
  simpa only [Measure.map_apply hmneg (measurableSet_singleton abstain),
    Measure.map_apply hmpos (measurableSet_singleton abstain)] using hb

/-- Endpoint regression: zero directional errors force full abstention mass. -/
theorem zero_errors_full_abstention {ρ : Measure Decision} [IsProbabilityMeasure ρ]
    (ha : ρ {adapt} = 0) (hf : ρ {freeze} = 0) : ρ {abstain} = 1 := by
  have hb := abstention_lower_bound (alpha := 0) ha.le hf.le
  apply le_antisymm
  · calc ρ {abstain} ≤ ρ Set.univ := measure_mono (Set.subset_univ _)
      _ = 1 := measure_univ
  · simpa using hb

end RandomizedActionLaw
end KBound

#print axioms KBound.RandomizedActionLaw.abstention_lower_bound
#print axioms KBound.RandomizedActionLaw.common_law_forces_abstention
#print axioms KBound.RandomizedActionLaw.randomized_rules_force_abstention
#print axioms KBound.RandomizedActionLaw.zero_errors_full_abstention
