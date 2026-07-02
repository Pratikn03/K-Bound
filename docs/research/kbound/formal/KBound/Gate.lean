import KBound.Basics

/-!
# Plug-in regret identity (`thm:gate`)

Paper: `paper/sections/main_theory_5.tex`, Lemma `thm:gate`.

For a committal gate that commits to adapt on `{Δ̂ > 0}` and freeze otherwise,
wrong-sign regret equals `|Δ|` on the error event. This is the algebraic spine
behind the finite-sample Le Cam regret floor.
-/

namespace KBound

/-- Committal plug-in gate: adapt iff the benefit estimate is positive. -/
noncomputable def plugInGate (dhat : ℝ) : Decision :=
  if 0 < dhat then Decision.adapt else Decision.freeze

/-- Wrong-sign regret on a single outcome with benefit `delta` and estimate `dhat`. -/
noncomputable def wrongSignRegret (delta dhat : ℝ) : ℝ :=
  if signReal delta ≠ signReal dhat then |delta| else 0

/-- Paper `thm:gate` pointwise identity: regret equals `|Δ|` on wrong-sign commits. -/
theorem gate_regret_identity (delta dhat : ℝ) :
    wrongSignRegret delta dhat =
      if signReal delta ≠ signReal dhat then |delta| else 0 := rfl

/-- Constant-magnitude specialization used in the Le Cam floor. -/
theorem gate_regret_constant_magnitude {delta dhat Lambda : ℝ}
    (hLambda : |delta| = Lambda)
    (hwrong : signReal delta ≠ signReal dhat) :
    wrongSignRegret delta dhat = Lambda := by
  simp [wrongSignRegret, hwrong, hLambda]

/-- Low-margin band: if `|Δ| ≤ ε`, committal regret is at most `ε`. -/
theorem gate_regret_low_margin_band {delta dhat eps : ℝ}
    (hmargin : |delta| ≤ eps)
    (hwrong : signReal delta ≠ signReal dhat) :
    wrongSignRegret delta dhat ≤ eps := by
  simpa [gate_regret_identity, hwrong] using hmargin

end KBound
