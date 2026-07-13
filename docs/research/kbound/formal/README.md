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
lake update
lake build           # KBound is the default target
```

The first full build may compile Mathlib from source and take 15-40 minutes.
Later builds are incremental. `build.sh` also removes macOS `._*` metadata files
before invoking Lake.

## Formal audit command

To build the Lean package and print the current mechanized coverage checklist:

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --json-out formal_audit_report.json
```

## Strict-core gate (Wave 4 closure)

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --strict-core
```

Legacy command:

```bash
python3 formal_audit.py --build --strict-100
```

`--strict-100` is kept as a backward-compatible alias for the strict-core gate. It does **not**
mean that all measure-theoretic probability, optional-stopping, KL/TV, or martingale-rate
arguments have been rebuilt from first principles in Mathlib.

Strict-core passes when: Lean build succeeds, no forbidden tokens (`sorry`/`admit`/`axiom`),
and every name in `VERIFIED_THEOREMS` is `#check`-able in `KBound/TheoremMap.lean`.

To force the deeper foundations question to fail loudly until it is actually mechanized:

```bash
python3 formal_audit.py --build --full-foundations
```

Current full-foundation blockers are documented by the audit output:

- full measure-theoretic conformal exchangeability;
- anytime/e-process optional stopping;
- full KL/TV probabilistic Le Cam layer;
- rate/martingale concentration theory;
- full evidence-preserving one-bit swap involution.

The current Lean status is therefore: **kernel-checked algebraic theorem spine plus finite-sample
bridge lemmas**, not a full foundational Mathlib probability development.

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
| exchangeability finite bridge | `KBound/Probability/ConformalExchangeability.lean` | `exchangeable_conformal_miss_le_alpha`, `exchangeable_cert_false_adapt_sound` |
| one-step e-process algebra | `KBound/Probability/EProcess.lean` | `bettingFactor_le_one`, `betting_wealth_step_le` |
| one-bit sign-flip core | `KBound/Dichotomy.lean` | `binary_sign_flip_on_accuracy_complement`, `multiclass_benefit_swap_pa_p0` |
| finite Le Cam TV algebra | `KBound/Probability/LeCam.lean` | `lecam_tv_identity`, `lecam_single_error_ge_one_sub_tv` |
| deterministic rate corollaries | `KBound/Probability/Rates.lean` | `rate_implies_commit`, `rate_conformal_miss` |
| multicandidate algebraic core | `KBound/Multicandidate.lean` | `multiclass_routing_harm_equiv`, `single_candidate_false_adapt_sound` |
| three-world multiclass harm core | `KBound/ThreeWorld.lean` | `multiclass_harm_iff_nonpos`, `multiclass_benefit_pos_of_pa_gt` |

Full index: `KBound/TheoremMap.lean`

## Toolchain

- Lean 4.29.1 (`lean-toolchain`)
- Mathlib v4.29.1 (`lakefile.lean`)
