import Mathlib.Data.Set.Basic

/-!
# A swap-orbit cross-section does not identify a sign on an evidence fibre

The four-world example below has two swap orbits and one evidence fibre. The
selected class contains exactly one world from each orbit, but its two members
have opposite benefits and identical evidence. Thus orbit-wise selection alone
cannot establish the sufficiency direction of an identification theorem.

The final theorem gives the exact set-theoretic replacement: a Boolean label
factors through the evidence if and only if it is constant on each evidence
fibre of the declared class. This is not a measurable-selection theorem; a
measurable decision rule needs the corresponding measurable structure as well.
-/

namespace KBound

namespace OrbitFibreCounterexample

/-- The first bit chooses an orbit; the second chooses the benefit sign. -/
abbrev World := Bool × Bool

def swap (w : World) : World := (w.1, !w.2)

def evidence (_ : World) : Unit := ()

def benefit (w : World) : Int := if w.2 then 1 else -1

/-- Different orbits are oriented differently. -/
def selected (w : World) : Prop := w.1 = w.2

theorem swap_involutive : Function.Involutive swap := by
  intro w
  rcases w with ⟨a, b⟩
  cases a <;> cases b <;> rfl

theorem swap_preserves_evidence (w : World) : evidence (swap w) = evidence w := rfl

theorem swap_negates_benefit (w : World) : benefit (swap w) = -benefit w := by
  rcases w with ⟨a, b⟩
  cases a <;> cases b <;> decide

/-- Exactly one representative is selected in every two-element swap orbit. -/
theorem selected_exactly_one (w : World) : selected w ↔ ¬selected (swap w) := by
  rcases w with ⟨a, b⟩
  cases a <;> cases b <;> simp [selected, swap]

/-- Despite exact orbit-wise selection, the evidence fibre has opposite signs. -/
theorem orbit_selection_not_fibre_orientation :
    (∀ w : World, selected w → ¬selected (swap w)) ∧
    ∃ p q : World, selected p ∧ selected q ∧ evidence p = evidence q ∧
      0 < benefit p ∧ benefit q < 0 := by
  refine ⟨fun w hw => (selected_exactly_one w).mp hw, ?_⟩
  exact ⟨(true, true), (false, false), rfl, rfl, rfl, by decide, by decide⟩

/-- No rule using only the observed evidence can recover both selected signs. -/
theorem no_evidence_decoder :
    ¬∃ decision : Unit → Bool, ∀ w : World, selected w → decision (evidence w) = w.2 := by
  rintro ⟨decision, hdecision⟩
  have hneg := hdecision (false, false) rfl
  have hpos := hdecision (true, true) rfl
  change decision () = false at hneg
  change decision () = true at hpos
  rw [hneg] at hpos
  contradiction

end OrbitFibreCounterexample

/-- The precise additional condition for set-theoretic sign identification is
fibre-wise consistency, not merely one representative per swap orbit. No
measurability of the decoder is asserted by this elementary factorization. -/
theorem bool_decoder_iff_constant_on_fibres {W E : Type*}
    (C : Set W) (observable : W → E) (label : W → Bool) :
    (∃ decision : E → Bool, ∀ w, w ∈ C → decision (observable w) = label w) ↔
    (∀ u, u ∈ C → ∀ v, v ∈ C → observable u = observable v → label u = label v) := by
  classical
  constructor
  · rintro ⟨decision, hdecision⟩ u hu v hv huv
    rw [← hdecision u hu, ← hdecision v hv, huv]
  · intro hconstant
    refine ⟨fun e => decide (∃ v, v ∈ C ∧ observable v = e ∧ label v = true), ?_⟩
    intro w hw
    cases hlabel : label w with
    | false =>
      have hnone : ¬∃ v, v ∈ C ∧ observable v = observable w ∧ label v = true := by
        rintro ⟨v, hv, hvw, hvlabel⟩
        have hsame := hconstant v hv w hw hvw
        rw [hvlabel, hlabel] at hsame
        contradiction
      simp [hnone]
    | true =>
      have hexists : ∃ v, v ∈ C ∧ observable v = observable w ∧ label v = true :=
        ⟨w, hw, rfl, hlabel⟩
      simp [hexists]

end KBound
