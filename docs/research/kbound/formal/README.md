# K-Bound Lean 4 Formalization (Mathlib)

Kernel-checked Mathlib proofs for the K-Bound theory spine and paper-faithful
foundation closures live here.

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

## Formal audit commands

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --strict-core
python3 formal_audit.py --build --strict-core --json-out formal_audit_report.json
python3 formal_audit.py --full-foundations  # expected FAIL; prints remaining foundations
```

- `--strict-core` / `--strict-100`: Lean build + no `sorry`/`admit`/`axiom` + every
  `VERIFIED_THEOREMS` name is `#check`-able.
- `--full-foundations`: a deliberately stronger audit covering a general
  measure-theoretic exchangeability theorem, filtered e-process/Ville theorem,
  general KL/TV Le Cam layer, concentration theory, and general target-law
  construction. It currently fails and lists those gaps.

Current release status: **65 named checks pass when `lake build` succeeds**,
including the closed/open frontier band and both zero-versus-strict boundary
witnesses. This is a valuable mechanized algebraic and finite-model spine. It is
not full foundational mechanization of the probability results named in the
paper; the stronger audit must remain red until those theorems are genuinely
formalized.

## What is mechanized (no `sorry`)

| Paper label | Lean file | Key theorems |
|-------------|-----------|--------------|
| `thm:cert` | `KBound/Certificate.lean` | `cert_false_adapt_sound`, `cert_false_freeze_sound` |
| `thm:gate` | `KBound/Gate.lean` | `gate_regret_identity` |
| `thm:imp`, `cor:forced-abstain` | `KBound/Impossibility.lean` | `abstention_mass_ge_one_sub_two_alpha_arith`, `matched_opposite_worlds_force_abstain` |
| `prop:lecam-finite` | `KBound/FiniteTesting.lean`, `LeCam.lean`, `LeCamMeasure.lean` | `lecam_testing_two_point`, `lecam_tv_two_point_measure` |
| `thm:frontier` | `KBound/Frontier.lean` | sufficiency, all decision branches, closed/open-band witnesses, and both zero-versus-strict boundary witnesses |
| `thm:frontier` necessity/maximality lift | `KBound/TargetLaw.lean` | finite discrete target laws, matched constant evidence, concrete opposite-benefit worlds, and a lift under explicit assumed `RichAt` |
| `lem:reduction`, `thm:disagree` | `KBound/Disagreement.lean` | `binary_sign_reduction` |
| `cor:samplecomp` | `KBound/Corollaries.lean` | `one_sided_commit_when_radius_small` |
| finite conformal rank algebra | `KBound/Conformal.lean` | `finite_uniform_rank_miss_le_alpha` |
| uniform-index conformal | `KBound/Probability/UniformConformal.lean` | `uniformIndex_false_adapt_le` |
| exchangeable-score reduction | `KBound/Probability/Exchangeable.lean` | `uniformIndexLaw_miss_le_alpha`, `uniformIndexLaw_false_adapt_le` |
| anytime / Ville finite core | `KBound/Probability/EProcess.lean`, `Ville.lean` | deterministic one-step wealth inequality and pointwise Markov indicator bound; no filtered supermartingale/optional-stopping development |
| one-bit swap involution | `KBound/Dichotomy.lean` | `evidence_swap_involution`, `swap_flips_benefit_preserves_evidence` |
| rate / Hoeffding bridge | `KBound/Probability/Rates.lean` | radius nonnegativity and conditional commit implication; the Hoeffding concentration theorem is not mechanized here |
| multicandidate algebraic core | `KBound/Multicandidate.lean` | `multiclass_routing_harm_equiv` |
| three-world multiclass harm core | `KBound/ThreeWorld.lean` | `multiclass_harm_iff_nonpos` |

Full index: `KBound/TheoremMap.lean`

## Explicit external or unmechanized assumptions

- Exchangeability must still be connected to the finite uniform-rank premise.
- Benchmark calibration transfer and risk alignment are deployment assumptions.
- The anytime claim still needs a filtered nonnegative-supermartingale and
  maximal/optional-stopping layer.
- The Le Cam development is a finite two-point model, not a general KL/TV
  product-experiment theorem.
- `TargetLaw.lean` assumes `RichAt`; it does not prove that an arbitrary real
  benchmark target class satisfies that richness condition.

## Toolchain

- Lean 4.29.1 (`lean-toolchain`)
- Mathlib v4.29.1 (`lakefile.lean`)
