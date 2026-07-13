import KBound.Gate
import KBound.FiniteTesting

/-!
# Finite-sample Le Cam regret floor (`prop:lecam-finite`)

Combines `thm:gate` with the two-point Le Cam identity.
-/

namespace KBound

/-- Worst-case regret of a committal gate at constant benefit magnitude `Λ`. -/
def worstCaseRegret (wrongProb Lambda : ℝ) : ℝ :=
  Lambda * wrongProb

/-- Paper `prop:lecam-finite` specialization on two-point evidence. -/
theorem lecam_regret_floor_two_point {p0 q0 Lambda : ℝ} :
    worstCaseRegret (min (mixedError2 true p0 q0) (mixedError2 false p0 q0)) Lambda
      = Lambda * (1 - tvDist2 p0 q0) := by
  simp [worstCaseRegret, lecam_testing_two_point, mul_sub]

end KBound
