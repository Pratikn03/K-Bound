import Mathlib.Probability.Independence.Basic
import Mathlib.MeasureTheory.Constructions.Pi

/-! # Independent zero-symmetric coordinates imply the full sign-flip null

This proves the sufficient condition in supplement554–563 for a finite vector
of measurable real gaps on any probability space. No normality, identical
distribution, moment or nonzero-coordinate assumption is required.
-/

namespace KBound.IndependentSignFlip

open MeasureTheory ProbabilityTheory

def coordinateFlip {ι : Type*} (s : ι → Bool) (v : ι → ℝ) : ι → ℝ :=
  fun i => if s i then -v i else v i

theorem independent_symmetric_coordinates {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (G : ι → Ω → ℝ) (hG : ∀ i, Measurable (G i))
    (hind : iIndepFun G mu)
    (hsym : ∀ i, (mu.map (G i)).map (fun x : ℝ => -x) = mu.map (G i))
    (s : ι → Bool) :
    (mu.map (fun omega i => G i omega)).map (coordinateFlip s) =
      mu.map (fun omega i => G i omega) := by
  have hjoint := (iIndepFun_iff_map_fun_eq_pi_map (fun i => (hG i).aemeasurable)).mp hind
  let f : ι → ℝ → ℝ := fun i x => if s i then -x else x
  have hf (i : ι) : Measurable (f i) := by
    cases hs : s i
    · simpa only [f, hs, Bool.false_eq_true, if_false] using (measurable_id : Measurable (id : ℝ → ℝ))
    · simpa only [f, hs, if_true] using (measurable_neg : Measurable (fun x : ℝ => -x))
  letI (i : ι) : IsProbabilityMeasure (mu.map (G i)) :=
    Measure.isProbabilityMeasure_map (hG i).aemeasurable
  letI (i : ι) : IsProbabilityMeasure ((mu.map (G i)).map (f i)) :=
    Measure.isProbabilityMeasure_map (hf i).aemeasurable
  rw [hjoint]
  change (Measure.pi (fun i => mu.map (G i))).map (fun v i => f i (v i)) = _
  rw [Measure.pi_map_pi (fun i => (hf i).aemeasurable)]
  congr 1
  funext i
  cases hs : s i
  · simpa only [f, hs, Bool.false_eq_true, if_false] using
      (Measure.map_id : (mu.map (G i)).map id = mu.map (G i))
  · simpa only [f, hs, if_true] using hsym i

end KBound.IndependentSignFlip

#print axioms KBound.IndependentSignFlip.independent_symmetric_coordinates
