import KBound.Basics
import KBound.Certificate
import KBound.Disagreement
import KBound.ThreeWorld

/-!
# Multicandidate routing (`thm:multicand`, `thm:multiclass-multicand`)

Algebraic cores for multiclass harm and single-candidate certificate soundness.
The Bonferroni union bound is probabilistic and remains outside this package.
-/

namespace KBound

open Decision

/-- Multiclass routing: harmfulness in accuracy gap matches `Δ ≤ 0`. -/
theorem multiclass_routing_harm_equiv {muD pa p0 : ℝ} (hmu : 0 < muD) :
    (pa ≤ p0) ↔ multiclassBenefit muD pa p0 ≤ 0 :=
  multiclass_harm_iff_nonpos hmu

/-- Single-candidate certificate false-adapt soundness (paper `thm:cert`). -/
theorem single_candidate_false_adapt_sound {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hcert : certificate dhat eps = Decision.adapt) :
    0 < delta :=
  cert_false_adapt_sound hcov hcert

end KBound
