import Lake
open Lake DSL

package "kbound_formal" where
  -- Pinned Mathlib supplies measure/kernel, filtration, conditional-expectation,
  -- sub-Gaussian and KL foundations as well as ordered-real proof tactics.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.29.1"

@[default_target]
lean_lib KBound where
  roots := #[`KBound]
