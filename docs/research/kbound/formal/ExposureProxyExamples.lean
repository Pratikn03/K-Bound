import KBound.Probability.ConditionalExposure
import KBound.Probability.IntegrableProxy

open MeasureTheory Set
open scoped ENNReal ProbabilityTheory

namespace KBound.ExposureProxyExamples

-- The numerator and denominator are actual event masses under one finite law.
example {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsFiniteMeasure μ]
    (A B : Set Ω) (alpha : ℝ) (hexposure : 0 < μ.real A)
    (hmarginal : μ.real (A ∩ B) ≤ alpha) :
    μ.real (A ∩ B) / μ.real A ≤ min 1 (alpha / μ.real A) := by
  exact ConditionalExposure.event_ratio_le_min μ A B alpha hexposure hmarginal

-- Zero and adverse effects both count as errors, and exposure must be positive.
example {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (action : Ω → Decision) (truth : Ω → ℝ) (alpha : ℝ)
    (hexposure : 0 < μ.real {ω | action ω = Decision.adapt})
    (hmarginal : μ.real {ω | action ω = Decision.adapt ∧ truth ω ≤ 0} ≤ alpha) :
    μ.real {ω | action ω = Decision.adapt ∧ truth ω ≤ 0} /
      μ.real {ω | action ω = Decision.adapt} ≤
        min 1 (alpha / μ.real {ω | action ω = Decision.adapt}) := by
  exact ConditionalExposure.false_adapt_ratio_le_min μ action truth alpha hexposure hmarginal

-- The same bound is composed with mathlib's actual conditional measure.
example {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (action : Ω → Decision) (truth : Ω → ℝ) (alpha : ℝ)
    (hA : MeasurableSet {ω | action ω = Decision.adapt})
    (hexposure : 0 < μ.real {ω | action ω = Decision.adapt})
    (hmarginal : μ.real {ω | action ω = Decision.adapt ∧ truth ω ≤ 0} ≤ alpha) :
    (μ[|{ω | action ω = Decision.adapt}]).real {ω | truth ω ≤ 0} ≤
      min 1 (alpha / μ.real {ω | action ω = Decision.adapt}) := by
  exact ConditionalExposure.conditional_false_adapt_le_min μ action truth alpha hA hexposure hmarginal

-- Actual integrals, not independent scalar names supplied for expectations.
example {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (G t : Ω → ℝ) (hG : Integrable G μ) (ht : Integrable t μ) :
    (∫ ω, G ω ∂μ) = (∫ ω, t ω ∂μ) + ∫ ω, G ω - t ω ∂μ := by
  exact IntegrableProxy.expectation_decomposition μ G t hG ht

example {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (G t : Ω → ℝ) (hG : Integrable G μ) (ht : Integrable t μ) (beta : ℝ)
    (hresidual : |∫ ω, G ω - t ω ∂μ| ≤ beta)
    (hmargin : beta < ∫ ω, t ω ∂μ) : 0 < ∫ ω, G ω ∂μ := by
  exact IntegrableProxy.positive_of_residual_bound μ G t hG ht beta hresidual hmargin

example {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (G t : Ω → ℝ) (hG : Integrable G μ) (ht : Integrable t μ) (beta : ℝ)
    (hresidual : |∫ ω, G ω - t ω ∂μ| ≤ beta)
    (hmargin : (∫ ω, t ω ∂μ) < -beta) : (∫ ω, G ω ∂μ) < 0 := by
  exact IntegrableProxy.negative_of_residual_bound μ G t hG ht beta hresidual hmargin

end KBound.ExposureProxyExamples
