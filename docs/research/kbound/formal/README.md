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
python3 formal_audit.py --build --full-foundations --json-out formal_audit_report.json
```

- `--strict-core` / `--strict-100`: Lean build + no `sorry`/`admit`/`axiom` + every
  `VERIFIED_THEOREMS` name is `#check`-able.
- `--full-foundations`: same, and requires the paper-faithful foundation gap list to be
  empty (Wave 6 closed those gaps).

Wave 6 status: **`--full-foundations` PASS** (53 kernel-checked theorem checks).
This is the paper-faithful bar (exchangeable-score reduction, discrete Ville,
two-point Le Cam packaging, Hoeffding-radius commit bridge, evidence swap
involution) — not a from-scratch Mathlib probability textbook.

## What is mechanized (no `sorry`)

| Paper label | Lean file | Key theorems |
|-------------|-----------|--------------|
| `thm:cert` | `KBound/Certificate.lean` | `cert_false_adapt_sound`, `cert_false_freeze_sound` |
| `thm:gate` | `KBound/Gate.lean` | `gate_regret_identity` |
| `thm:imp`, `cor:forced-abstain` | `KBound/Impossibility.lean` | `forced_abstention_probability`, `matched_opposite_worlds_force_abstain` |
| `prop:lecam-finite` | `KBound/FiniteTesting.lean`, `LeCam.lean`, `LeCamMeasure.lean` | `lecam_testing_two_point`, `lecam_tv_two_point_measure` |
| `thm:frontier` | `KBound/Frontier.lean` | `frontier_identifiable_positive`, `frontier_decision_abstain` |
| `lem:reduction`, `thm:disagree` | `KBound/Disagreement.lean` | `binary_sign_reduction` |
| `cor:samplecomp` | `KBound/Corollaries.lean` | `one_sided_commit_when_radius_small` |
| finite conformal rank algebra | `KBound/Conformal.lean` | `finite_uniform_rank_miss_le_alpha` |
| uniform-index conformal | `KBound/Probability/UniformConformal.lean` | `uniformIndex_false_adapt_le` |
| exchangeable-score reduction | `KBound/Probability/Exchangeable.lean` | `exchangeable_scores_miss_le_alpha`, `exchangeable_scores_false_adapt_le` |
| anytime / Ville | `KBound/Probability/EProcess.lean`, `Ville.lean` | `betting_wealth_supermartingale_step`, `ville_bound_false_adapt` |
| one-bit swap involution | `KBound/Dichotomy.lean` | `evidence_swap_involution`, `swap_flips_benefit_preserves_evidence` |
| rate / Hoeffding bridge | `KBound/Probability/Rates.lean` | `hoeffding_radius_le`, `rate_commit_from_concentration` |
| multicandidate algebraic core | `KBound/Multicandidate.lean` | `multiclass_routing_harm_equiv` |
| three-world multiclass harm core | `KBound/ThreeWorld.lean` | `multiclass_harm_iff_nonpos` |

Full index: `KBound/TheoremMap.lean`

## Toolchain

- Lean 4.29.1 (`lean-toolchain`)
- Mathlib v4.29.1 (`lakefile.lean`)
