import KBound.Probability.LeCam
import Mathlib.Data.Real.Basic

/-!
# Measure-packaged two-point Le Cam / TV layer

Paper: `prop:lecam-finite`, `thm:minimax-opt` (two-point testing).

Records the finite TV identity under an explicit two-point probability packaging
(`TwoPointLaw`) so the Le Cam affinity bound is stated at the measure-model
layer used by the paper, not only as bare reals.
-/

namespace KBound

/-- Two-point testing law: success probability of world `0` (Bernoulli parameter). -/
structure TwoPointLaw where
  p : ℝ
  hp0 : 0 ≤ p
  hp1 : p ≤ 1

/-- Total variation between two two-point laws. -/
noncomputable def twoPointTV (μ ν : TwoPointLaw) : ℝ :=
  tvDist2 μ.p ν.p

/-- **Measure-packaged TV identity** (equals the finite TV model). -/
theorem lecam_tv_two_point_measure (μ ν : TwoPointLaw) :
    twoPointTV μ ν = tvDist2 μ.p ν.p := rfl

/-- Testing error lower-bounds the affinity in the two-point measure model. -/
theorem lecam_testing_error_ge_one_sub_tv_measure (μ ν : TwoPointLaw) (b : Bool) :
    1 - twoPointTV μ ν ≤ mixedError2 b μ.p ν.p := by
  simpa [twoPointTV] using lecam_single_error_ge_one_sub_tv μ.p ν.p b

/-- Convenience: affinity equals the Le Cam min-error identity. -/
theorem lecam_affinity_eq_min_error_measure (μ ν : TwoPointLaw) :
    min (mixedError2 true μ.p ν.p) (mixedError2 false μ.p ν.p)
      = 1 - twoPointTV μ ν := by
  simpa [twoPointTV] using lecam_tv_identity μ.p ν.p

end KBound
