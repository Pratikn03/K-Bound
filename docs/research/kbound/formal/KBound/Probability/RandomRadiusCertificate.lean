import KBound.Probability.MeasureCertificate
import Mathlib.MeasureTheory.Constructions.BorelSpace.Real

/-!
# Coverage-to-action for a random extended-nonnegative radius

Paper: Proposition `thm:certificate` / `thm:cert`.

The estimate and benefit are finite real-valued maps on one probability space;
the radius is an arbitrary map into `ℝ≥0∞`, and need not be constant, independent
of the estimate, or independent of the benefit. Infinite radius means abstention.
For finite radius the decision is exactly the real interval rule in `Certificate`.
Strict decisions exclude intervals whose relevant endpoint equals zero.

The theorem assumes marginal coverage. It does not establish exchangeability,
label-blind construction, validity of a fitted radius, conditional error given
selection, or simultaneous/repeated-use protection. Only measurability of the
coverage event is needed for the outer-measure bounds; the convenience theorem
derives that premise from measurable maps, as in the manuscript.
-/

namespace KBound

open Decision MeasureTheory
open scoped ENNReal

/-- The manuscript interval rule for finite estimates and extended nonnegative
radius. The `∞` branch must precede `toReal`, because `∞.toReal = 0`. -/
noncomputable def extendedCertificate (dhat : ℝ) (eps : ℝ≥0∞) : Decision :=
  if eps = ∞ then abstain else certificate dhat eps.toReal

@[simp] theorem extendedCertificate_infinite (dhat : ℝ) :
    extendedCertificate dhat ∞ = abstain := by
  simp [extendedCertificate]

theorem extendedCertificate_finite {dhat : ℝ} {eps : ℝ≥0∞}
    (heps : eps ≠ ∞) :
    extendedCertificate dhat eps = certificate dhat eps.toReal := by
  simp [extendedCertificate, heps]

/-- A lower endpoint at exactly zero cannot produce either strict decision. -/
theorem extendedCertificate_lower_zero {dhat : ℝ} {eps : ℝ≥0∞}
    (hzero : dhat - eps.toReal = 0) :
    extendedCertificate dhat eps = abstain := by
  by_cases heps : eps = ∞
  · simp [heps]
  · rw [extendedCertificate_finite heps]
    apply certificate_abstains_when_interval_straddles_zero
    · rw [hzero]
      exact lt_irrefl 0
    · have heps_nonneg := ENNReal.toReal_nonneg (a := eps)
      linarith

/-- An upper endpoint at exactly zero cannot produce either strict decision. -/
theorem extendedCertificate_upper_zero {dhat : ℝ} {eps : ℝ≥0∞}
    (hzero : dhat + eps.toReal = 0) :
    extendedCertificate dhat eps = abstain := by
  by_cases heps : eps = ∞
  · simp [heps]
  · rw [extendedCertificate_finite heps]
    apply certificate_abstains_when_interval_straddles_zero
    · have heps_nonneg := ENNReal.toReal_nonneg (a := eps)
      linarith
    · rw [hzero]
      exact lt_irrefl 0

/-- The degenerate zero-radius, zero-estimate interval abstains. -/
@[simp] theorem extendedCertificate_zero_zero :
    extendedCertificate 0 0 = abstain := by
  simp [extendedCertificate, certificate]

theorem extended_adapt_sound_on_coverage {dhat delta : ℝ} {eps : ℝ≥0∞}
    (hcov : ENNReal.ofReal |dhat - delta| ≤ eps)
    (hcert : extendedCertificate dhat eps = adapt) :
    0 < delta := by
  by_cases heps : eps = ∞
  · simp [heps] at hcert
  · exact adapt_sound_on_coverage
      ((ENNReal.ofReal_le_iff_le_toReal heps).mp hcov)
      (by simpa [extendedCertificate, heps] using hcert)

theorem extended_freeze_sound_on_coverage {dhat delta : ℝ} {eps : ℝ≥0∞}
    (hcov : ENNReal.ofReal |dhat - delta| ≤ eps)
    (hcert : extendedCertificate dhat eps = freeze) :
    delta < 0 := by
  by_cases heps : eps = ∞
  · simp [heps] at hcert
  · exact freeze_sound_on_coverage
      ((ENNReal.ofReal_le_iff_le_toReal heps).mp hcov)
      (by simpa [extendedCertificate, heps] using hcert)

variable {Ω : Type*}

/-- Coverage with a random extended-nonnegative radius. -/
def randomRadiusCoverageEvent (dhat delta : Ω → ℝ) (eps : Ω → ℝ≥0∞) : Set Ω :=
  {ω | ENNReal.ofReal |dhat ω - delta ω| ≤ eps ω}

def randomRadiusFalseAdaptEvent (dhat delta : Ω → ℝ) (eps : Ω → ℝ≥0∞) : Set Ω :=
  {ω | extendedCertificate (dhat ω) (eps ω) = adapt ∧ delta ω ≤ 0}

def randomRadiusFalseFreezeEvent (dhat delta : Ω → ℝ) (eps : Ω → ℝ≥0∞) : Set Ω :=
  {ω | extendedCertificate (dhat ω) (eps ω) = freeze ∧ 0 ≤ delta ω}

theorem measurableSet_randomRadiusCoverageEvent [MeasurableSpace Ω]
    {dhat delta : Ω → ℝ} {eps : Ω → ℝ≥0∞}
    (hdhat : Measurable dhat) (hdelta : Measurable delta) (heps : Measurable eps) :
    MeasurableSet (randomRadiusCoverageEvent dhat delta eps) := by
  have habs : Measurable (fun ω => |dhat ω - delta ω|) :=
    continuous_abs.measurable.comp (hdhat.sub hdelta)
  exact measurableSet_le habs.ennreal_ofReal heps

theorem randomRadiusFalseAdaptEvent_subset_compl (dhat delta : Ω → ℝ)
    (eps : Ω → ℝ≥0∞) :
    randomRadiusFalseAdaptEvent dhat delta eps ⊆
      (randomRadiusCoverageEvent dhat delta eps)ᶜ := by
  rintro ω ⟨hcert, hnonpos⟩ hcov
  exact (not_lt.mpr hnonpos) (extended_adapt_sound_on_coverage hcov hcert)

theorem randomRadiusFalseFreezeEvent_subset_compl (dhat delta : Ω → ℝ)
    (eps : Ω → ℝ≥0∞) :
    randomRadiusFalseFreezeEvent dhat delta eps ⊆
      (randomRadiusCoverageEvent dhat delta eps)ᶜ := by
  rintro ω ⟨hcert, hnonneg⟩ hcov
  exact (not_lt.mpr hnonneg) (extended_freeze_sound_on_coverage hcov hcert)

/-- Both directional errors share one coverage-failure event, so their union
costs `α`, not `2α`. No independence is used. -/
theorem randomRadiusEitherError_subset_compl (dhat delta : Ω → ℝ)
    (eps : Ω → ℝ≥0∞) :
    randomRadiusFalseAdaptEvent dhat delta eps ∪
      randomRadiusFalseFreezeEvent dhat delta eps ⊆
      (randomRadiusCoverageEvent dhat delta eps)ᶜ := by
  exact Set.union_subset (randomRadiusFalseAdaptEvent_subset_compl dhat delta eps)
    (randomRadiusFalseFreezeEvent_subset_compl dhat delta eps)

/-- Manuscript coverage-to-action proposition, false-adapt direction, with a
random radius in `[0,∞]` and arbitrary dependence on the finite real inputs. -/
theorem measure_randomRadius_false_adapt_le_alpha [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {dhat delta : Ω → ℝ} {eps : Ω → ℝ≥0∞} {alpha : ℝ≥0∞}
    (hs : MeasurableSet (randomRadiusCoverageEvent dhat delta eps))
    (hcov : 1 - alpha ≤ μ (randomRadiusCoverageEvent dhat delta eps)) :
    μ (randomRadiusFalseAdaptEvent dhat delta eps) ≤ alpha :=
  measure_le_alpha_of_subset_compl hs
    (randomRadiusFalseAdaptEvent_subset_compl dhat delta eps) hcov

/-- Manuscript coverage-to-action proposition, false-freeze direction. -/
theorem measure_randomRadius_false_freeze_le_alpha [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {dhat delta : Ω → ℝ} {eps : Ω → ℝ≥0∞} {alpha : ℝ≥0∞}
    (hs : MeasurableSet (randomRadiusCoverageEvent dhat delta eps))
    (hcov : 1 - alpha ≤ μ (randomRadiusCoverageEvent dhat delta eps)) :
    μ (randomRadiusFalseFreezeEvent dhat delta eps) ≤ alpha :=
  measure_le_alpha_of_subset_compl hs
    (randomRadiusFalseFreezeEvent_subset_compl dhat delta eps) hcov

/-- The union of directional errors is bounded by the same single `α`. -/
theorem measure_randomRadius_either_error_le_alpha [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {dhat delta : Ω → ℝ} {eps : Ω → ℝ≥0∞} {alpha : ℝ≥0∞}
    (hs : MeasurableSet (randomRadiusCoverageEvent dhat delta eps))
    (hcov : 1 - alpha ≤ μ (randomRadiusCoverageEvent dhat delta eps)) :
    μ (randomRadiusFalseAdaptEvent dhat delta eps ∪
      randomRadiusFalseFreezeEvent dhat delta eps) ≤ alpha :=
  measure_le_alpha_of_subset_compl hs
    (randomRadiusEitherError_subset_compl dhat delta eps) hcov

/-- The manuscript's measurable-map assumptions imply the required event
measurability; the supplied coverage premise remains an assumption. -/
theorem measure_randomRadius_either_error_le_alpha_of_measurable [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {dhat delta : Ω → ℝ} {eps : Ω → ℝ≥0∞} {alpha : ℝ≥0∞}
    (hdhat : Measurable dhat) (hdelta : Measurable delta) (heps : Measurable eps)
    (hcov : 1 - alpha ≤ μ (randomRadiusCoverageEvent dhat delta eps)) :
    μ (randomRadiusFalseAdaptEvent dhat delta eps ∪
      randomRadiusFalseFreezeEvent dhat delta eps) ≤ alpha :=
  measure_randomRadius_either_error_le_alpha
    (measurableSet_randomRadiusCoverageEvent hdhat hdelta heps) hcov

end KBound
