import Mathlib.Algebra.Order.Group.Abs

/-!
# (A7) Estimator stability transfers leave-one-out coverage

`UnitMismatch.lean` shows what goes wrong when a radius is calibrated at the wrong
*unit*. This file addresses the other half of the leave-one-out question: what has to
be assumed for a LOO-calibrated radius to say anything about the *full-data* fit that
is actually deployed.

The answer is a stability hypothesis, and it is worth being precise about why. There
is no theorem that leave-one-out empirical calibration is exact conformal -- the plain
jackknife has no distribution-free finite-sample guarantee, and the finite-sample
counterpart (jackknife+) buys `1 - 2*alpha`, not `1 - alpha`. What *is* true, and is
proved here, is a deterministic transfer: if the deployed full-data fit differs from
the leave-one-out fit by at most `beta`, then LOO coverage at radius `eps` gives
full-fit coverage at radius `beta + eps`.

So `beta` is the price of using LOO calibration, and it is an assumption about the
estimator, not a property the data can establish. That is why it joins the contract as

    (A7) Estimator stability: the full-data fit and each leave-one-out fit differ by
         at most beta in prediction.

`unstable_fit_voids_transfer` is the contrapositive and is the operationally useful
direction: an observed miss beyond `beta + eps` is *proof* that stability failed at
level `beta`. It converts a coverage failure into a falsified assumption rather than
bad luck, which is what makes (A7) checkable after the fact.

Stated over an ordered additive group, so it applies to `ℝ` unchanged. Both results
are term-mode and depend only on Lean's standard axioms (`propext`, `Quot.sound`) --
never `sorryAx`; see the `#print axioms` block below.
-/

namespace KBound

/-- **(A7) transfer.** Stability of the fit converts leave-one-out coverage into
full-data coverage at an inflated radius. The inflation `beta` is exactly the
stability modulus; there is no free lunch, and setting `beta = 0` asserts the fit does
not depend on any single calibration point. -/
theorem stability_transfers_loo_coverage
    {α : Type*} [AddCommGroup α] [LinearOrder α] [IsOrderedAddMonoid α]
    {dhatFull dhatLoo delta beta eps : α}
    (h_stab : |dhatFull - dhatLoo| ≤ beta)
    (h_loo : |dhatLoo - delta| ≤ eps) :
    |dhatFull - delta| ≤ beta + eps :=
  le_trans (abs_sub_le dhatFull dhatLoo delta) (add_le_add h_stab h_loo)

/-- **(A7) is falsifiable.** A miss beyond `beta + eps`, given LOO coverage at `eps`,
proves the fit was less stable than `beta`. The failure is attributable rather than
mysterious: it did not "just miss", the stability assumption was false. -/
theorem unstable_fit_voids_transfer
    {α : Type*} [AddCommGroup α] [LinearOrder α] [IsOrderedAddMonoid α]
    {dhatFull dhatLoo delta beta eps : α}
    (h_miss : beta + eps < |dhatFull - delta|)
    (h_loo : |dhatLoo - delta| ≤ eps) :
    beta < |dhatFull - dhatLoo| :=
  not_le.mp fun hle =>
    absurd (stability_transfers_loo_coverage hle h_loo) (not_le.mpr h_miss)

end KBound

-- Axiom audit. Verified on Lean 4.29.1 + Mathlib v4.29.1:
--   both depend on [propext, Quot.sound] only -- Lean's standard axioms, no sorryAx.
#print axioms KBound.stability_transfers_loo_coverage
#print axioms KBound.unstable_fit_voids_transfer
