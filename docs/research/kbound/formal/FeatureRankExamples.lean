import KBound
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic

open scoped BigOperators

example {ι : Type*} (rows : ι → Fin 11 → ℝ)
    (h6 : ∀ i, rows i 6 = rows i 2 - rows i 5)
    (h7 : ∀ i, rows i 7 = rows i 0 - rows i 3) :
    Module.finrank ℝ (Submodule.span ℝ (Set.range rows)) ≤ 9 :=
  KBound.FeatureRank.rowspan_finrank_le rows h6 h7

example : Module.finrank ℝ (LinearMap.range KBound.FeatureRank.encode) = 9 :=
  KBound.FeatureRank.schema_finrank

example : (KBound.FeatureRank.encode ![4, 2, 8, 1, 3, 5, 7, 9, 6]) 6 = 3 ∧
    (KBound.FeatureRank.encode ![4, 2, 8, 1, 3, 5, 7, 9, 6]) 7 = 3 := by
  change (8 : ℝ) - 5 = 3 ∧ (4 : ℝ) - 1 = 3
  norm_num

example : (∀ _i : Unit, (fun _ : Fin 11 => (0 : ℝ)) 6 = 0 - 0 ∧
      (fun _ : Fin 11 => (0 : ℝ)) 7 = 0 - 0) ∧
    Module.finrank ℝ (Submodule.span ℝ (Set.range (fun _ : Unit => (0 : Fin 11 → ℝ)))) = 0 := by
  simp

example : ![(0 : ℝ), 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] ∉
    LinearMap.range KBound.FeatureRank.encode := by
  rw [KBound.FeatureRank.mem_range_iff]
  change ¬ ((1 : ℝ) = 0 - 0 ∧ (0 : ℝ) = 0 - 0)
  norm_num
