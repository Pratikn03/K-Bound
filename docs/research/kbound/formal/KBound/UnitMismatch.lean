import Mathlib.Order.Basic

/-!
# Unit mismatch forces a coverage miss (`A5`)

Companion to the numerical witness in `scripts/loo_unit_mismatch_witness.py` and the
locked artifact `research_lock/LOO_UNIT_MISMATCH_WITNESS_v1.json`.

The witness shows that a radius calibrated leave-one-CELL-out attains its nominal level
for a new cell of a *seen* condition (0.9005 at nominal 0.90) and collapses on a *new*
condition (0.1378), because within-condition residuals live at scale `sigma` while
across-condition errors live at scale `tau > sigma`.

What is deterministic in that story -- and therefore what belongs in Lean -- is this: a
radius calibrated at the smaller scale *cannot* cover an error at the larger one. No
probability is needed; it is transitivity. The probabilistic content (that `sigma` and
`tau` really are the two scales, and that the gap is `tau/sigma`) is what the simulation
supplies.

Stated over an arbitrary `LinearOrder`, so it applies to `ℝ` and to any ordered value
type with no further hypotheses. All three results are proved in term mode and depend on
NO axioms -- not even `propext` / `Classical.choice` / `Quot.sound`; see the
`#print axioms` block at the end of this file.
-/

namespace KBound

/-- **Unit mismatch forces a miss.**
If the radius was calibrated at the within-unit scale `sigma`, the across-unit scale
`tau` is strictly larger, and the realised deployment error is at least `tau`, then the
interval misses. The certificate's coverage event fails by construction, not by bad luck.
-/
theorem unit_mismatch_forces_miss {α : Type*} [LinearOrder α]
    {sigma tau epsCell err : α}
    (h_cal : epsCell ≤ sigma)
    (h_gap : sigma < tau)
    (h_err : tau ≤ err) :
    ¬ (err ≤ epsCell) := fun h =>
  absurd (le_trans h_err (le_trans h h_cal)) (not_le.mpr h_gap)

/-- Contrapositive: covering an error of size at least `tau` *requires* a radius that
reaches past the within-unit scale. Across-unit coverage cannot be bought with a
within-unit radius at any level `alpha`. -/
theorem covering_requires_across_unit_radius {α : Type*} [LinearOrder α]
    {sigma tau epsCell err : α}
    (h_gap : sigma < tau)
    (h_err : tau ≤ err)
    (h_cov : err ≤ epsCell) :
    sigma < epsCell :=
  lt_of_lt_of_le h_gap (le_trans h_err h_cov)

/-- The miss is monotone in the calibrated radius: shrinking it can only preserve a
miss. This rules out the response "just use a slightly larger cell-out radius" -- the
fix has to change the unit, not the level. -/
theorem miss_mono {α : Type*} [LinearOrder α] {e e' err : α}
    (h_le : e' ≤ e) (h_miss : ¬ (err ≤ e)) : ¬ (err ≤ e') :=
  fun h => h_miss (le_trans h h_le)

end KBound

-- Axiom audit. Verified on Lean 4.29.1 + Mathlib v4.29.1:
--   'KBound.unit_mismatch_forces_miss' does not depend on any axioms
--   'KBound.covering_requires_across_unit_radius' does not depend on any axioms
--   'KBound.miss_mono' does not depend on any axioms
#print axioms KBound.unit_mismatch_forces_miss
#print axioms KBound.covering_requires_across_unit_radius
#print axioms KBound.miss_mono
