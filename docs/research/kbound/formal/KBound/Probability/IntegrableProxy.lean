import KBound.Frontier
import Mathlib.MeasureTheory.Integral.Bochner.Basic

namespace KBound.IntegrableProxy

open MeasureTheory

/-- Actual integrals under one measure. Both the advantage and proxy are
integrable; therefore the residual is integrable too. No independence is needed. -/
theorem expectation_decomposition {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (G t : Ω → ℝ) (hG : Integrable G μ) (ht : Integrable t μ) :
    (∫ ω, G ω ∂μ) = (∫ ω, t ω ∂μ) + ∫ ω, G ω - t ω ∂μ := by
  rw [integral_sub hG ht]
  ring

/-- A strict positive proxy mean exceeding an externally supplied bound on the
actual residual expectation certifies positive expected advantage. -/
theorem positive_of_residual_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (G t : Ω → ℝ) (hG : Integrable G μ) (ht : Integrable t μ)
    (beta : ℝ) (hresidual : |∫ ω, G ω - t ω ∂μ| ≤ beta)
    (hmargin : beta < ∫ ω, t ω ∂μ) : 0 < ∫ ω, G ω ∂μ := by
  rw [expectation_decomposition μ G t hG ht]
  exact frontier_positive_raw hmargin hresidual

/-- The symmetric strict negative statement. These are sufficiency results,
not a general-loss or multiclass identified-set/necessity theorem. -/
theorem negative_of_residual_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (G t : Ω → ℝ) (hG : Integrable G μ) (ht : Integrable t μ)
    (beta : ℝ) (hresidual : |∫ ω, G ω - t ω ∂μ| ≤ beta)
    (hmargin : (∫ ω, t ω ∂μ) < -beta) : (∫ ω, G ω ∂μ) < 0 := by
  rw [expectation_decomposition μ G t hG ht]
  exact frontier_negative_raw (by linarith) hresidual

end KBound.IntegrableProxy

#print axioms KBound.IntegrableProxy.expectation_decomposition
#print axioms KBound.IntegrableProxy.positive_of_residual_bound
#print axioms KBound.IntegrableProxy.negative_of_residual_bound
