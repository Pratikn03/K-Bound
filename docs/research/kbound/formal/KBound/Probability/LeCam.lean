import KBound.FiniteTesting
import Mathlib.Data.Real.Basic

/-!
# Finite Le Cam TV algebra (`prop:lecam-finite`, `thm:minimax-opt` finite case)

This module records the finite two-point TV identity used by the paper proof. It does not
mechanize KL, absolute continuity, or product-experiment probability measures.
-/

namespace KBound

/-- Le Cam two-point identity (finite TV model). -/
theorem lecam_tv_identity (p0 q0 : ℝ) :
    min (mixedError2 true p0 q0) (mixedError2 false p0 q0) = 1 - tvDist2 p0 q0 :=
  lecam_testing_two_point p0 q0

/-- Any single error lower-bounds the affinity. -/
theorem lecam_single_error_ge_one_sub_tv (p0 q0 : ℝ) (b : Bool) :
    1 - tvDist2 p0 q0 ≤ mixedError2 b p0 q0 := by
  have h := lecam_testing_two_point p0 q0
  rcases b with rfl | rfl
  · rw [← h]
    exact min_le_right (mixedError2 true p0 q0) (mixedError2 false p0 q0)
  · rw [← h]
    exact min_le_left (mixedError2 true p0 q0) (mixedError2 false p0 q0)

end KBound
