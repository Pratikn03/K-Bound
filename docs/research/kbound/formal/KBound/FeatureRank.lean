import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic

/-! # Exact feature-schema dimension versus observed data rank

The printed eleven-coordinate order has pbal_drop at6 and entropy_drop at7.
They equal coordinate2 minus5 and coordinate0 minus3 respectively. The resulting
linear schema has dimension9, so every family of rows satisfying these exact
relations spans at most9 dimensions. No empirical matrix rank is asserted.
-/

namespace KBound.FeatureRank

def encode : (Fin 9 → ℝ) →ₗ[ℝ] (Fin 11 → ℝ) where
  toFun x := ![x 0, x 1, x 2, x 3, x 4, x 5, x 2 - x 5, x 0 - x 3, x 6, x 7, x 8]
  map_add' x y := by
    ext i
    fin_cases i <;> simp <;> ring
  map_smul' a x := by
    ext i
    fin_cases i <;> simp [mul_sub]

def decode (v : Fin 11 → ℝ) : Fin 9 → ℝ :=
  ![v 0, v 1, v 2, v 3, v 4, v 5, v 8, v 9, v 10]

theorem encode_relations (x : Fin 9 → ℝ) :
    encode x 6 = encode x 2 - encode x 5 ∧
      encode x 7 = encode x 0 - encode x 3 := by
  simp [encode]

theorem encode_injective : Function.Injective encode := by
  have h : Function.LeftInverse decode encode := by
    intro x
    ext i
    fin_cases i <;> rfl
  exact h.injective

theorem mem_range_iff (v : Fin 11 → ℝ) :
    v ∈ LinearMap.range encode ↔ v 6 = v 2 - v 5 ∧ v 7 = v 0 - v 3 := by
  constructor
  · rintro ⟨x, rfl⟩
    exact encode_relations x
  · intro h
    refine ⟨decode v, ?_⟩
    ext i
    fin_cases i <;> simp [encode, decode, h.1, h.2]

theorem schema_finrank : Module.finrank ℝ (LinearMap.range encode) = 9 := by
  rw [LinearMap.finrank_range_of_inj encode_injective]
  simp

theorem rowspan_finrank_le {ι : Type*} (rows : ι → Fin 11 → ℝ)
    (h6 : ∀ i, rows i 6 = rows i 2 - rows i 5)
    (h7 : ∀ i, rows i 7 = rows i 0 - rows i 3) :
    Module.finrank ℝ (Submodule.span ℝ (Set.range rows)) ≤ 9 := by
  rw [← schema_finrank]
  apply Submodule.finrank_mono
  apply Submodule.span_le.mpr
  rintro _ ⟨i, rfl⟩
  exact (mem_range_iff (rows i)).mpr ⟨h6 i, h7 i⟩

end KBound.FeatureRank

#print axioms KBound.FeatureRank.encode_relations
#print axioms KBound.FeatureRank.encode_injective
#print axioms KBound.FeatureRank.mem_range_iff
#print axioms KBound.FeatureRank.schema_finrank
#print axioms KBound.FeatureRank.rowspan_finrank_le
