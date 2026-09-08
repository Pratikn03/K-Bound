import KBound.Probability.MeasureCertificate
import Mathlib.MeasureTheory.Constructions.BorelSpace.Real

/-!
# Marginal coverage with a measurable finite-or-infinite radius

The radius is ENNReal, so it is nonnegative and may equal infinity on any
measurable subset. Infinity always abstains. Coverage is marginal on the same
probability space as the estimate, truth, and radius; no independence or
conditional coverage is assumed. A zero true effect counts as directional error.
-/

namespace KBound.ExtendedRadiusCertificate

open MeasureTheory KBound.Decision
open scoped ENNReal

noncomputable def action (estimate : ℝ) (radius : ℝ≥0∞) : Decision :=
  if radius = ⊤ then abstain else certificate estimate radius.toReal

@[simp] theorem action_top (estimate : ℝ) : action estimate ⊤ = abstain := by
  simp [action]

theorem action_finite (estimate : ℝ) {radius : ℝ≥0∞} (hr : radius ≠ ⊤) :
    action estimate radius = certificate estimate radius.toReal := by
  simp [action, hr]

/-- The extended rule agrees exactly with the original rule at real nonnegative radii. -/
theorem action_ofReal (estimate radius : ℝ) (hr : 0 ≤ radius) :
    action estimate (ENNReal.ofReal radius) = certificate estimate radius := by
  rw [action_finite _ ENNReal.ofReal_ne_top, ENNReal.toReal_ofReal hr]

def coverage {Ω : Type*} (estimate truth : Ω → ℝ) (radius : Ω → ℝ≥0∞) : Set Ω :=
  {ω | ENNReal.ofReal |estimate ω - truth ω| ≤ radius ω}

def falseDirection {Ω : Type*} (estimate truth : Ω → ℝ)
    (radius : Ω → ℝ≥0∞) : Set Ω :=
  {ω | (action (estimate ω) (radius ω) = adapt ∧ truth ω ≤ 0) ∨
       (action (estimate ω) (radius ω) = freeze ∧ 0 ≤ truth ω)}

theorem measurableSet_coverage {Ω : Type*} [MeasurableSpace Ω]
    {estimate truth : Ω → ℝ} {radius : Ω → ℝ≥0∞}
    (he : Measurable estimate) (ht : Measurable truth) (hr : Measurable radius) :
    MeasurableSet (coverage estimate truth radius) := by
  exact measurableSet_le (ENNReal.measurable_ofReal.comp
    (continuous_abs.measurable.comp (he.sub ht))) hr

theorem false_direction_subset_failure {Ω : Type*}
    (estimate truth : Ω → ℝ) (radius : Ω → ℝ≥0∞) :
    falseDirection estimate truth radius ⊆ (coverage estimate truth radius)ᶜ := by
  intro ω hw hc
  by_cases hr : radius ω = ⊤
  · simp [falseDirection, action, hr] at hw
  · have hcov : |estimate ω - truth ω| ≤ (radius ω).toReal :=
      (ENNReal.ofReal_le_iff_le_toReal hr).mp hc
    rcases hw with ⟨ha, ht⟩ | ⟨ha, ht⟩
    · rw [action_finite _ hr] at ha
      exact (not_lt_of_ge ht) (adapt_sound_on_coverage hcov ha)
    · rw [action_finite _ hr] at ha
      exact (not_lt_of_ge ht) (freeze_sound_on_coverage hcov ha)

/-- Both directional errors together have probability at most alpha. The
measurability theorem below also makes this an ordinary measurable-event bound. -/
theorem directional_error_probability {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {estimate truth : Ω → ℝ} {radius : Ω → ℝ≥0∞} {alpha : ℝ≥0∞}
    (he : Measurable estimate) (ht : Measurable truth) (hr : Measurable radius)
    (hcov : 1 - alpha ≤ μ (coverage estimate truth radius)) :
    μ (falseDirection estimate truth radius) ≤ alpha :=
  measure_le_alpha_of_subset_compl (measurableSet_coverage he ht hr)
    (false_direction_subset_failure estimate truth radius) hcov

theorem measurableSet_falseDirection {Ω : Type*} [MeasurableSpace Ω]
    {estimate truth : Ω → ℝ} {radius : Ω → ℝ≥0∞}
    (he : Measurable estimate) (ht : Measurable truth) (hr : Measurable radius) :
    MeasurableSet (falseDirection estimate truth radius) := by
  have hfinite : MeasurableSet {ω | radius ω ≠ ⊤} :=
    (measurableSet_eq_fun hr measurable_const).compl
  have hp : MeasurableSet {ω | 0 < estimate ω - (radius ω).toReal} :=
    measurableSet_lt measurable_const (he.sub hr.ennreal_toReal)
  have hn : MeasurableSet {ω | estimate ω + (radius ω).toReal < 0} :=
    measurableSet_lt (he.add hr.ennreal_toReal) measurable_const
  have hset : falseDirection estimate truth radius =
      {ω | radius ω ≠ ⊤} ∩
        (({ω | 0 < estimate ω - (radius ω).toReal} ∩ {ω | truth ω ≤ 0}) ∪
         (({ω | 0 < estimate ω - (radius ω).toReal})ᶜ ∩
           {ω | estimate ω + (radius ω).toReal < 0} ∩ {ω | 0 ≤ truth ω})) := by
    ext ω
    by_cases hr : radius ω = ⊤
    · simp [falseDirection, action, hr]
    · by_cases hp : (radius ω).toReal < estimate ω <;>
        by_cases hn : estimate ω + (radius ω).toReal < 0 <;>
        simp [falseDirection, action, certificate, hr, sub_pos, hp, hn]
  rw [hset]
  exact hfinite.inter ((hp.inter (measurableSet_le ht measurable_const)).union
    ((hp.compl.inter hn).inter (measurableSet_le measurable_const ht)))

end KBound.ExtendedRadiusCertificate

#print axioms KBound.ExtendedRadiusCertificate.directional_error_probability
#print axioms KBound.ExtendedRadiusCertificate.measurableSet_falseDirection
#print axioms KBound.ExtendedRadiusCertificate.action_top
