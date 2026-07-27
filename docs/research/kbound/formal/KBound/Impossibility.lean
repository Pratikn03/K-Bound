import KBound.Certificate

/-!
# Forced abstention under matched evidence (`cor:forced-abstain`)

Paper: Theorem `thm:imp` part (iii) and Corollary `cor:forced-abstain`.
-/

namespace KBound

open Decision

/-- Pointwise sign-soundness for one benefit value. -/
def SoundForBenefit (a : Decision) (delta : ℝ) : Prop :=
  (a = adapt → 0 < delta) ∧ (a = freeze → delta < 0)

theorem matched_opposite_worlds_force_abstain {a : Decision} {delta₁ delta₂ : ℝ}
    (hneg : delta₁ < 0)
    (hpos : 0 < delta₂)
    (hsound₁ : SoundForBenefit a delta₁)
    (hsound₂ : SoundForBenefit a delta₂) :
    a = abstain := by
  cases a with
  | adapt =>
      have hbad : 0 < delta₁ := hsound₁.1 rfl
      linarith
  | freeze =>
      have hbad : delta₂ < 0 := hsound₂.2 rfl
      linarith
  | abstain =>
      rfl

theorem certificate_forced_abstention_under_matched_coverage
    {dhat eps delta₁ delta₂ : ℝ}
    (hcov₁ : |dhat - delta₁| ≤ eps)
    (hcov₂ : |dhat - delta₂| ≤ eps)
    (hneg : delta₁ < 0)
    (hpos : 0 < delta₂) :
    certificate dhat eps = abstain := by
  apply matched_opposite_worlds_force_abstain hneg hpos
  · constructor
    · intro hadapt; exact adapt_sound_on_coverage hcov₁ hadapt
    · intro hfreeze; exact freeze_sound_on_coverage hcov₁ hfreeze
  · constructor
    · intro hadapt; exact adapt_sound_on_coverage hcov₂ hadapt
    · intro hfreeze; exact freeze_sound_on_coverage hcov₂ hfreeze

/-- **Arithmetic core of paper `thm:imp` (iii)** (matched-evidence abstention rate).
Scope, stated honestly: this is a statement about three *real numbers*, not about a
measure or about the certificate.  Given `qa ≤ α` and `qf ≤ α` it concludes
`1 - qa - qf ≥ 1 - 2α` by `linarith`.  Nothing here establishes that `qa` and `qf`
*are* the false-adapt and false-freeze probabilities of the certificate, nor that
abstention is forced; those are the paper's content and are NOT formalised.
Renamed 2026-07-26: the previous name `forced_abstention_probability` claimed
probabilistic content this statement does not have. -/
theorem abstention_mass_ge_one_sub_two_alpha_arith {qa qf alpha : ℝ}
    (hfa : qa ≤ alpha)
    (hff : qf ≤ alpha)
    (_hprob : qa + qf ≤ 1) :
    1 - qa - qf ≥ 1 - 2 * alpha := by
  linarith

end KBound
