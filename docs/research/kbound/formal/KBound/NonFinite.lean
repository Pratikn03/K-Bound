import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# Impossibility without finiteness

`TargetLaw.lean` builds matched opposite worlds from a *finite* discrete law, and
`TheoremMap` records the non-finite realization as pen-and-paper. This file closes the
part of that gap which is not measure theory.

The observation is that the impossibility argument never uses finiteness. It needs only
an evidence type `E` -- arbitrary, possibly a continuum -- a rule `E → Decision`, and two
worlds that the rule cannot tell apart with benefits of opposite sign. Everything after
that is case analysis. So the finite construction in `TargetLaw.lean` is a convenience,
not a load-bearing assumption.

`continuum_matched_witness` then exhibits the configuration on `ℝ` with the evidence map
`x ↦ |x|`, which collapses `c` and `-c`. No measure theory is required because the
worlds are points rather than laws; what remains pen-and-paper is the construction of
matched *probability measures* on a non-finite space, which is a strictly stronger
statement and is not claimed here.
-/

namespace KBoundNF

inductive Decision | adapt | freeze | abstain
deriving DecidableEq

/-- A decision is sound for a benefit when committing is justified by its sign. -/
def Sound (a : Decision) (b : ℝ) : Prop :=
  match a with
  | Decision.adapt   => 0 < b
  | Decision.freeze  => b < 0
  | Decision.abstain => True

/-- A world over an arbitrary evidence type. `E` is unconstrained: finite, countable or
a continuum. -/
structure World (E : Type*) where
  evidence : E
  benefit  : ℝ

/-- **Matched opposite worlds force abstention, over any evidence type.**
No finiteness, no measurability, no cardinality assumption on `E`. -/
theorem matched_opposite_forces_abstain {E : Type*}
    (rule : E → Decision) (wp wn : World E)
    (hmatch : wp.evidence = wn.evidence)
    (hp : 0 < wp.benefit) (hn : wn.benefit < 0)
    (hsp : Sound (rule wp.evidence) wp.benefit)
    (hsn : Sound (rule wn.evidence) wn.benefit) :
    rule wp.evidence = Decision.abstain := by
  have hsn' : Sound (rule wp.evidence) wn.benefit := by rw [hmatch]; exact hsn
  cases h : rule wp.evidence with
  | adapt   => rw [h] at hsn'; simp [Sound] at hsn'; exfalso; linarith
  | freeze  => rw [h] at hsp;  simp [Sound] at hsp;  exfalso; linarith
  | abstain => rfl

/-- Contrapositive: a rule that commits on matched opposite worlds is unsound on one. -/
theorem no_uniformly_sound_committal_rule {E : Type*}
    (rule : E → Decision) (wp wn : World E)
    (hmatch : wp.evidence = wn.evidence)
    (hp : 0 < wp.benefit) (hn : wn.benefit < 0)
    (hcommit : rule wp.evidence ≠ Decision.abstain) :
    ¬ (Sound (rule wp.evidence) wp.benefit ∧ Sound (rule wn.evidence) wn.benefit) := by
  rintro ⟨h1, h2⟩
  exact hcommit (matched_opposite_forces_abstain rule wp wn hmatch hp hn h1 h2)

/-- A non-injective evidence map on a continuum. -/
def collapse : ℝ → ℝ := fun x => |x|

theorem collapse_matches (c : ℝ) : collapse c = collapse (-c) := by
  simp [collapse]

/-- **The configuration occurs on a continuum.** Evidence type `ℝ`, benefits `±c`. -/
theorem continuum_matched_witness (c : ℝ) (hc : 0 < c) :
    ∃ wp wn : World ℝ,
      wp.evidence = wn.evidence ∧ 0 < wp.benefit ∧ wn.benefit < 0 :=
  ⟨⟨collapse c, c⟩, ⟨collapse (-c), -c⟩, collapse_matches c, hc, by linarith⟩

/-- Any rule sound on both arms of the continuum witness must abstain there. -/
theorem continuum_impossibility (rule : ℝ → Decision) (c : ℝ) (hc : 0 < c)
    (h1 : Sound (rule (collapse c)) c)
    (h2 : Sound (rule (collapse (-c))) (-c)) :
    rule (collapse c) = Decision.abstain :=
  matched_opposite_forces_abstain rule ⟨collapse c, c⟩ ⟨collapse (-c), -c⟩
    (collapse_matches c) hc (by linarith) h1 h2

end KBoundNF

-- Axiom audit, Lean 4.29.1 + Mathlib v4.29.1: all three depend only on
-- [propext, Classical.choice, Quot.sound]; never sorryAx.
#print axioms KBoundNF.matched_opposite_forces_abstain
#print axioms KBoundNF.continuum_matched_witness
#print axioms KBoundNF.continuum_impossibility
