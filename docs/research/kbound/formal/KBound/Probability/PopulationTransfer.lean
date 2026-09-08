import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.Tactic.Linarith

/-!
# Conditional cell-to-population transfer

The coverage hypotheses below are mathematical premises, not empirical findings.
The measure takes ENNReal values: `1 - (alpha + delta)` is truncated subtraction,
i.e. max(0, 1 - alpha - delta) for finite nonnegative real budgets. No independence,
IID, exchangeability, conformal validity, or target coverage premise is established.
-/

open MeasureTheory Set
open scoped ENNReal

namespace KBound.PopulationTransfer

theorem pointwise {dhat cell pop eps b : ℝ}
    (hSample : |dhat - cell| ≤ eps) (hCell : |cell - pop| ≤ b) :
    |dhat - pop| ≤ eps + b := by
  calc
    |dhat - pop| = |(dhat - cell) + (cell - pop)| := by congr 1; ring
    _ ≤ |dhat - cell| + |cell - pop| := abs_add_le _ _
    _ ≤ eps + b := add_le_add hSample hCell

theorem positive_of_coverage {dhat pop radius : ℝ}
    (hCoverage : |dhat - pop| ≤ radius) (hDecision : 0 < dhat - radius) :
    0 < pop := by
  have h := (abs_le.mp hCoverage).2
  linarith

theorem negative_of_coverage {dhat pop radius : ℝ}
    (hCoverage : |dhat - pop| ≤ radius) (hDecision : dhat + radius < 0) :
    pop < 0 := by
  have h := (abs_le.mp hCoverage).1
  linarith

theorem positive {dhat cell pop eps b : ℝ}
    (hSample : |dhat - cell| ≤ eps) (hCell : |cell - pop| ≤ b)
    (hDecision : 0 < dhat - (eps + b)) : 0 < pop :=
  positive_of_coverage (pointwise hSample hCell) hDecision

theorem negative {dhat cell pop eps b : ℝ}
    (hSample : |dhat - cell| ≤ eps) (hCell : |cell - pop| ≤ b)
    (hDecision : dhat + (eps + b) < 0) : pop < 0 :=
  negative_of_coverage (pointwise hSample hCell) hDecision

def coverage {Ω : Type*} (x y r : Ω → ℝ) : Set Ω :=
  {w | |x w - y w| ≤ r w}

theorem coverage_inter_subset {Ω : Type*} (dhat cell pop eps b : Ω → ℝ) :
    coverage dhat cell eps ∩ coverage cell pop b ⊆
      coverage dhat pop (fun w => eps w + b w) := by
  intro w hw
  exact pointwise hw.1 hw.2

/-- Pointwise containment allows arbitrary functions, including random radii. -/
theorem failure_subset_union {Ω : Type*} (dhat cell pop eps b : Ω → ℝ) :
    (coverage dhat pop (fun w => eps w + b w))ᶜ ⊆
      (coverage dhat cell eps)ᶜ ∪ (coverage cell pop b)ᶜ := by
  intro w hw
  by_cases h : w ∈ coverage dhat cell eps
  · exact Or.inr (fun hc => hw (pointwise h hc))
  · exact Or.inl h

/-- Union bound from marginal coverage only; no independence hypothesis. -/
theorem complement_union_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {A B : Set Ω}
    (hA : MeasurableSet A) (hB : MeasurableSet B) {alpha delta : ℝ≥0∞}
    (hPA : 1 - alpha ≤ μ A) (hPB : 1 - delta ≤ μ B) :
    μ (Aᶜ ∪ Bᶜ) ≤ alpha + delta := by
  have ha : μ Aᶜ ≤ alpha := by
    rw [prob_compl_eq_one_sub hA]
    exact tsub_le_iff_tsub_le.mp hPA
  have hb : μ Bᶜ ≤ delta := by
    rw [prob_compl_eq_one_sub hB]
    exact tsub_le_iff_tsub_le.mp hPB
  exact (measure_union_le _ _).trans (add_le_add ha hb)

/-- A general bridge used by the function-level probability theorem. The target
event need not be measurable for the outer-measure inequality; if measurable,
this is an ordinary probability lower bound. -/
theorem event_transfer {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {A B G : Set Ω}
    (hA : MeasurableSet A) (hB : MeasurableSet B) {alpha delta : ℝ≥0∞}
    (hPA : 1 - alpha ≤ μ A) (hPB : 1 - delta ≤ μ B)
    (hTransfer : A ∩ B ⊆ G) : 1 - (alpha + delta) ≤ μ G := by
  have hFail : μ (A ∩ B)ᶜ ≤ alpha + delta := by
    simpa only [compl_inter] using complement_union_bound μ hA hB hPA hPB
  rw [prob_compl_eq_one_sub (hA.inter hB)] at hFail
  exact (tsub_le_iff_tsub_le.mp hFail).trans (measure_mono hTransfer)

/-- Conditional population coverage from the two measurable premise events.
No independence and no constant-radius requirement. -/
theorem population_coverage {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (dhat cell pop eps b : Ω → ℝ)
    (hSample : MeasurableSet (coverage dhat cell eps))
    (hCell : MeasurableSet (coverage cell pop b)) {alpha delta : ℝ≥0∞}
    (hPSample : 1 - alpha ≤ μ (coverage dhat cell eps))
    (hPCell : 1 - delta ≤ μ (coverage cell pop b)) :
    1 - (alpha + delta) ≤ μ (coverage dhat pop (fun w => eps w + b w)) :=
  event_transfer μ hSample hCell hPSample hPCell (coverage_inter_subset ..)

/-- Includes both incorrect signs, counting a zero population effect as false
when a strict positive or negative certificate was issued. -/
def falseDirection {Ω : Type*} (dhat pop radius : Ω → ℝ) : Set Ω :=
  {w | (0 < dhat w - radius w ∧ pop w ≤ 0) ∨
       (dhat w + radius w < 0 ∧ 0 ≤ pop w)}

theorem false_direction_subset_failure {Ω : Type*} (dhat pop radius : Ω → ℝ) :
    falseDirection dhat pop radius ⊆ (coverage dhat pop radius)ᶜ := by
  intro w hw hc
  rcases hw with hp | hn
  · exact (not_lt_of_ge hp.2) (positive_of_coverage hc hp.1)
  · exact (not_lt_of_ge hn.2) (negative_of_coverage hc hn.1)

/-- Outer-measure bound; hence also an ordinary probability bound whenever the
false-direction event is measurable. -/
theorem false_direction_probability {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (dhat cell pop eps b : Ω → ℝ)
    (hSample : MeasurableSet (coverage dhat cell eps))
    (hCell : MeasurableSet (coverage cell pop b)) {alpha delta : ℝ≥0∞}
    (hPSample : 1 - alpha ≤ μ (coverage dhat cell eps))
    (hPCell : 1 - delta ≤ μ (coverage cell pop b)) :
    μ (falseDirection dhat pop (fun w => eps w + b w)) ≤ alpha + delta := by
  apply le_trans (measure_mono ((false_direction_subset_failure ..).trans
    (failure_subset_union dhat cell pop eps b)))
  exact complement_union_bound μ hSample hCell hPSample hPCell

end KBound.PopulationTransfer

#print axioms KBound.PopulationTransfer.pointwise
#print axioms KBound.PopulationTransfer.population_coverage
#print axioms KBound.PopulationTransfer.false_direction_probability
