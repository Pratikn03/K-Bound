import KBound.Probability.ChannelCounterexample
import Mathlib.Tactic

/-!
# Corrected historical one-bit/channel statement and a conditional H-rate bridge

The historical orbit-selection sufficiency statement is false: an evidence
fibre can contain opposite labels even when a class chooses one representative
from every swap orbit.  The exact replacement is the fibre-consistency
criterion below.  The H/ratio-rate result is intentionally a contract theorem:
it propagates a declared source miss budget through a nonnegative domination
factor.  It does not estimate H, establish exchangeability, or turn a
benchmark replay into prospective population evidence.
-/

namespace KBound

/-- The corrected one-bit/channel theorem: a label-free decoder exists exactly
when the Boolean orientation is constant on every observable evidence fibre.
This is the fibre condition missing from the historical orbit-only claim. -/
theorem historical_one_bit_decoder_iff_fibre_consistent {W E : Type*}
    (C : Set W) (observable : W → E) (label : W → Bool) :
    (∃ decision : E → Bool, ∀ w, w ∈ C → decision (observable w) = label w) ↔
    (∀ u, u ∈ C → ∀ v, v ∈ C → observable u = observable v → label u = label v) :=
  bool_decoder_iff_constant_on_fibres C observable label

/-- A declared episode-level H/ratio-rate contract.

`sourceMiss ≤ alpha` is the source calibration/rate premise and
`targetMiss ≤ H * sourceMiss` is the change-of-measure or ratio-rate premise.
The theorem below derives the target budget only when `H ≥ 0`; all statistical
and exchangeability content remains an explicit premise of the contract. -/
structure HRatioRateContract where
  sourceMiss : ℝ
  targetMiss : ℝ
  H : ℝ
  alpha : ℝ
  source_nonneg : 0 ≤ sourceMiss
  target_nonneg : 0 ≤ targetMiss
  H_nonneg : 0 ≤ H
  alpha_nonneg : 0 ≤ alpha
  source_bound : sourceMiss ≤ alpha
  ratio_bound : targetMiss ≤ H * sourceMiss

/-- Conditional H/ratio-rate propagation: `targetMiss ≤ H * alpha`. -/
theorem h_ratio_rate_transfer (contract : HRatioRateContract) :
    contract.targetMiss ≤ contract.H * contract.alpha := by
  nlinarith [contract.ratio_bound, contract.source_bound, contract.H_nonneg]

/-- If the domination factor is at most one, the target inherits the source
calibration budget. -/
theorem h_ratio_rate_budget (contract : HRatioRateContract)
    (hH : contract.H ≤ 1) :
    contract.targetMiss ≤ contract.alpha := by
  have htransfer := h_ratio_rate_transfer contract
  nlinarith [htransfer, contract.alpha_nonneg, contract.H_nonneg]

end KBound
