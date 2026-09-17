import KBound.Population

namespace KBoundPopulationTest

open KBound KBound.Decision

example {estimate epsilon samplingRadius target : ℝ}
    (hcontained : |target - estimate| ≤ epsilon + samplingRadius)
    (hdecision : populationDecision estimate epsilon samplingRadius = adapt) :
    0 < target := by
  exact populationDecision_adapt_sound hcontained hdecision

example {estimate epsilon samplingRadius target : ℝ}
    (hcontained : |target - estimate| ≤ epsilon + samplingRadius)
    (hdecision : populationDecision estimate epsilon samplingRadius = freeze) :
    target < 0 := by
  exact populationDecision_freeze_sound hcontained hdecision

example {estimate epsilon samplingRadius : ℝ} :
    populationDecision estimate epsilon samplingRadius =
      populationDecision estimate epsilon samplingRadius := by
  rfl

end KBoundPopulationTest
