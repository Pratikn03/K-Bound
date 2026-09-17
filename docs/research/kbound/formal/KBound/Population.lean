import KBound.Basics
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# Finite-sample population-transfer decision layer

This module formalizes the deterministic part of the population procedure used
in the supplement.  `epsilon` is the cell-level prediction radius and
`samplingRadius` is the separately justified population-sampling radius.  The
probabilistic statements that justify either radius remain explicit hypotheses;
this file does not silently turn a benchmark replay into a population claim.
-/

namespace KBound

open Decision

/-- The compound half-width used by the population interval. -/
noncomputable def populationRadius (epsilon samplingRadius : ℝ) : ℝ :=
  epsilon + samplingRadius

/-- The interval used for the population-benefit decision. -/
noncomputable def populationInterval
    (estimate epsilon samplingRadius : ℝ) : Set ℝ :=
  Set.Icc
    (estimate - populationRadius epsilon samplingRadius)
    (estimate + populationRadius epsilon samplingRadius)

/-- Population action from the compound interval, with zero left abstained. -/
noncomputable def populationDecision
    (estimate epsilon samplingRadius : ℝ) : Decision :=
  if 0 < estimate - populationRadius epsilon samplingRadius then
    adapt
  else if estimate + populationRadius epsilon samplingRadius < 0 then
    freeze
  else
    abstain

/-- Nonnegative component radii give a nonnegative compound radius. -/
theorem populationRadius_nonneg {epsilon samplingRadius : ℝ}
    (hepsilon : 0 ≤ epsilon) (hsampling : 0 ≤ samplingRadius) :
    0 ≤ populationRadius epsilon samplingRadius := by
  simp [populationRadius]
  linarith

/-- Absolute-error containment is exactly interval membership. -/
theorem target_mem_populationInterval
    {estimate epsilon samplingRadius target : ℝ}
    (hcontained : |target - estimate| ≤ populationRadius epsilon samplingRadius) :
    target ∈ populationInterval estimate epsilon samplingRadius := by
  constructor
  · have hlow := (abs_le.mp hcontained).1
    linarith
  · have hupp := (abs_le.mp hcontained).2
    linarith

/-- A strict population ADAPT action is sound whenever the compound interval
contains the target benefit. -/
theorem populationDecision_adapt_sound
    {estimate epsilon samplingRadius target : ℝ}
    (hcontained : |target - estimate| ≤ populationRadius epsilon samplingRadius)
    (hdecision : populationDecision estimate epsilon samplingRadius = adapt) :
    0 < target := by
  unfold populationDecision at hdecision
  by_cases hpositive : 0 < estimate - populationRadius epsilon samplingRadius
  · have hlow := (abs_le.mp hcontained).1
    linarith
  · by_cases hnegative : estimate + populationRadius epsilon samplingRadius < 0
    · simp [hpositive, hnegative] at hdecision
    · simp [hpositive, hnegative] at hdecision

/-- A strict population FREEZE action is sound whenever the compound interval
contains the target benefit. -/
theorem populationDecision_freeze_sound
    {estimate epsilon samplingRadius target : ℝ}
    (hcontained : |target - estimate| ≤ populationRadius epsilon samplingRadius)
    (hdecision : populationDecision estimate epsilon samplingRadius = freeze) :
    target < 0 := by
  unfold populationDecision at hdecision
  by_cases hpositive : 0 < estimate - populationRadius epsilon samplingRadius
  · simp [hpositive] at hdecision
  · by_cases hnegative : estimate + populationRadius epsilon samplingRadius < 0
    · have hupp := (abs_le.mp hcontained).2
      linarith
    · simp [hpositive, hnegative] at hdecision

/-- A population ABSTAIN action is the only non-committal outcome when the
compound interval meets zero. -/
theorem populationDecision_abstain_of_zero_mem
    {estimate epsilon samplingRadius : ℝ}
    (hzero : (0 : ℝ) ∈ populationInterval estimate epsilon samplingRadius) :
    populationDecision estimate epsilon samplingRadius = abstain := by
  have hleft := hzero.1
  have hright := hzero.2
  unfold populationDecision
  simp [not_lt.mpr hleft, not_lt.mpr hright]

end KBound
