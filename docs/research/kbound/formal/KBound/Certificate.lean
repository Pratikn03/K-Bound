import KBound.Basics
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# K-Bound certificate core (`thm:cert`)

Paper: `manuscript/chapters/ch05_theory_certificates.tex`, Theorem `thm:cert`.
-/

namespace KBound

open Decision

/-- The K-Bound trichotomy: commit only when the certified interval clears zero. -/
noncomputable def certificate (dhat eps : ℝ) : Decision :=
  if 0 < dhat - eps then
    adapt
  else if dhat + eps < 0 then
    freeze
  else
    abstain

/-- On the coverage event, an `adapt` decision has genuinely positive benefit. -/
theorem adapt_sound_on_coverage {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hcert : certificate dhat eps = adapt) :
    0 < delta := by
  unfold certificate at hcert
  by_cases hpos : 0 < dhat - eps
  · have hupper : dhat - delta ≤ eps := (abs_le.mp hcov).2
    linarith
  · by_cases hneg : dhat + eps < 0
    · simp [hpos, hneg] at hcert
    · simp [hpos, hneg] at hcert

/-- On the coverage event, a `freeze` decision has genuinely negative benefit. -/
theorem freeze_sound_on_coverage {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hcert : certificate dhat eps = freeze) :
    delta < 0 := by
  unfold certificate at hcert
  by_cases hpos : 0 < dhat - eps
  · simp [hpos] at hcert
  · by_cases hneg : dhat + eps < 0
    · have hlower : -eps ≤ dhat - delta := (abs_le.mp hcov).1
      linarith
    · simp [hpos, hneg] at hcert

/-- False-adapt can only occur outside the coverage event. -/
theorem cert_false_adapt_implies_coverage_failure {dhat delta eps : ℝ}
    (hcert : certificate dhat eps = adapt)
    (hfalse : ¬ 0 < delta) :
    ¬ |dhat - delta| ≤ eps := by
  intro hcov
  exact hfalse (adapt_sound_on_coverage hcov hcert)

/-- False-freeze can only occur outside the coverage event. -/
theorem cert_false_freeze_implies_coverage_failure {dhat delta eps : ℝ}
    (hcert : certificate dhat eps = freeze)
    (hfalse : ¬ delta < 0) :
    ¬ |dhat - delta| ≤ eps := by
  intro hcov
  exact hfalse (freeze_sound_on_coverage hcov hcert)

/-- If the interval straddles zero, the rule abstains. -/
theorem certificate_abstains_when_interval_straddles_zero {dhat eps : ℝ}
    (hleft : ¬ 0 < dhat - eps)
    (hright : ¬ dhat + eps < 0) :
    certificate dhat eps = abstain := by
  simp [certificate, hleft, hright]

/-- Paper `thm:cert` (false-adapt containment). -/
theorem cert_false_adapt_sound {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hcert : certificate dhat eps = adapt) :
    0 < delta :=
  adapt_sound_on_coverage hcov hcert

/-- Paper `thm:cert` (false-freeze containment). -/
theorem cert_false_freeze_sound {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hcert : certificate dhat eps = freeze) :
    delta < 0 :=
  freeze_sound_on_coverage hcov hcert

/-- Paper remark: both error events are covered by one bad coverage event. -/
theorem cert_both_errors_subset_coverage_failure {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps) :
    ¬ (certificate dhat eps = adapt ∧ delta ≤ 0) ∧
    ¬ (certificate dhat eps = freeze ∧ 0 ≤ delta) := by
  constructor
  · rintro ⟨hadapt, h_nonpos⟩
    have hpos : 0 < delta := adapt_sound_on_coverage hcov hadapt
    linarith
  · rintro ⟨hfreeze, h_nonneg⟩
    have hneg : delta < 0 := freeze_sound_on_coverage hcov hfreeze
    linarith

end KBound
