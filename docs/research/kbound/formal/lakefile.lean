import Lake
open Lake DSL

package "kbound_formal" where
  -- Mathlib is used for ordered real arithmetic (`linarith`) and absolute-value lemmas.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.29.1"

@[default_target]
lean_lib KBound where
  roots := #[`KBound]
