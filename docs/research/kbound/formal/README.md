# K-Bound Lean 4 Formalization (Mathlib)

Kernel-checked Mathlib proofs for the K-Bound theory spine live here.

## Quick build (recommended)

```bash
cd docs/research/kbound/formal
bash build.sh
```

Or manually:

```bash
cd docs/research/kbound/formal
lake update          # cache failure on T9 is OK — see below
lake build           # KBound is the default target
```

## If you see `failed to fetch cache` or `cannot execute binary file`

This is **normal** on an external T9 drive. The Mathlib prebuilt cache binary is often wrong-architecture or non-executable on exFAT. **Ignore it.** `lake build` compiles Mathlib from source instead.

First full build: **15–40 minutes**. Later builds are incremental.

## If you see `non UTF-8 data` / `._RefreshComponent.tsx`

macOS creates `._*` junk files on external disks. Fix:

```bash
find . -name '._*' -delete
lake build
```

Or run `bash build.sh` (does this automatically).

## Formal audit command

To build the Lean package and print the current mechanized coverage checklist:

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --json-out formal_audit_report.json
```

## Strict-100 gate (Wave 4 closure)

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --strict-100
```

This passes when: Lean build succeeds, no forbidden tokens (`sorry`/`admit`/`axiom`), `NOT_YET_MECHANIZED` and `OPEN_RESEARCH_FRONTIER` are empty, and every name in `VERIFIED_THEOREMS` is `#check`-able in `KBound/TheoremMap.lean`.

## What is mechanized (no `sorry`)

| Paper label | Lean file | Key theorems |
|-------------|-----------|--------------|
| `thm:cert` | `KBound/Certificate.lean` | `cert_false_adapt_sound`, `cert_false_freeze_sound` |
| `thm:gate` | `KBound/Gate.lean` | `gate_regret_identity` |
| `thm:imp`, `cor:forced-abstain` | `KBound/Impossibility.lean` | `forced_abstention_probability`, `matched_opposite_worlds_force_abstain` |
| `prop:lecam-finite` | `KBound/FiniteTesting.lean`, `LeCam.lean` | `lecam_testing_two_point`, `lecam_regret_floor_two_point` |
| `thm:frontier` | `KBound/Frontier.lean` | `frontier_identifiable_positive`, `frontier_decision_abstain` |
| `lem:reduction`, `thm:disagree` | `KBound/Disagreement.lean` | `binary_sign_reduction` |
| `thm:disagree-mc` | `KBound/Disagreement.lean` | `multiclass_sign_reduction` |
| `cor:samplecomp` | `KBound/Corollaries.lean` | `one_sided_commit_when_radius_small` |
| finite conformal rank algebra | `KBound/Conformal.lean` | `finite_uniform_rank_coverage_add_miss`, `finite_uniform_rank_miss_le_alpha` |
| exchangeability bridge | `KBound/Probability/ConformalExchangeability.lean` | `exchangeable_conformal_miss_le_alpha`, `exchangeable_cert_false_adapt_sound` |
| e-process betting core | `KBound/Probability/EProcess.lean` | `bettingFactor_le_one`, `betting_wealth_step_le` |
| one-bit dichotomy | `KBound/Dichotomy.lean` | `binary_sign_flip_on_accuracy_complement`, `multiclass_benefit_swap_pa_p0` |
| Le Cam TV layer | `KBound/Probability/LeCam.lean` | `lecam_tv_identity`, `lecam_single_error_ge_one_sub_tv` |
| rates | `KBound/Probability/Rates.lean` | `rate_implies_commit`, `rate_conformal_miss` |
| multicandidate algebraic core | `KBound/Multicandidate.lean` | `multiclass_routing_harm_equiv`, `single_candidate_false_adapt_sound` |
| three-world multiclass harm core | `KBound/ThreeWorld.lean` | `multiclass_harm_iff_nonpos`, `multiclass_benefit_pos_of_pa_gt` |

Full index: `KBound/TheoremMap.lean`

## Toolchain

- Lean 4.29.1 (`lean-toolchain`)
- Mathlib v4.29.1 (`lakefile.lean`)
