# K-Bound Lean 4 Formalization (Mathlib)

Lean proofs for the K-Bound theory spine and explicitly scoped probability
foundations live here. A proof establishes its encoded statement under its
assumptions; it does not validate a dataset or guarantee an experimental gain.

## Quick build (recommended)

```bash
cd docs/research/kbound/formal
bash build.sh
```

Or manually:

```bash
cd docs/research/kbound/formal
lake build KBound
python3 formal_audit.py --build --strict-core
```

## Pinned dependencies and cache access

Use the checked-in toolchain and Lake manifest. The build does not run
`lake update`, remove AppleDouble files, or clean dependency caches. A missing,
incompatible or inaccessible cache must be resolved, or the dependencies must
compile successfully from the pinned sources; a cache error is not a proof pass.

First-build time depends on hardware and cache availability. Later builds are
incremental. `KBOUND_PYTHON` (or `PYTHON`) selects the audit interpreter.

## Formal audit commands

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --strict-core
python3 formal_audit.py --build --strict-core --json-out /tmp/kbound-formal-audit.json
python3 formal_audit.py --build --full-foundations  # expected FAIL: historical sixth-layer extension
```

- `--build --strict-core`: build the pinned target, check every registered name,
  scan for proof holes, and inspect each declaration's transitive kernel axioms.
  Only `propext`, `Classical.choice`, and `Quot.sound` are allowed. Missing output,
  extra axioms or failed compilation fails the audit. `--strict-100` is only a
  legacy alias; neither option means that every historical claim is proved.
- Without `--build`, the default is a **static inventory check**, not kernel
  verification. `--strict-core` without `--build` fails.
- `--full-foundations` retains the stronger six-layer requirement. It fails
  because the historical one-bit/H/ratio-rate extension is not proved, and its
  orbit-selection sufficiency claim needs correction.

The registry contains **142 declarations: 65 legacy core results and 77 new
probability capstones and counterexample results**. Supporting lemmas are also
compiled; the registry count is not a measure of novelty or scientific quality.
The release runbook writes its kernel/axiom receipt to
`../audits/formal_foundations_2026_08_31.json` and binds it in the outer checksum
inventory. That receipt distinguishes five mechanized layers from the partial
sixth layer.

## Five probability layers and the partial sixth layer

| Layer | Modules | Encoded scope and important limits |
| --- | --- | --- |
| Exchangeable conformal coverage | `MeasureConformal.lean` | Actual measurable exchangeable score laws, strict ranks with ties, calibration thresholds and one-shot residual coverage/directional error bounds. No assumed uniform-rank conclusion. Not benchmark exchangeability or repeated-use coverage. |
| Filtered e-process/Ville | `FilteredVille.lean` | Nonnegative supermartingales, bounded optional stopping, countable-time maximal bounds, dominated e-processes and a constructed predictable betting product. Conditional nulls and filtration assumptions remain explicit. |
| General KL/TV testing | `GeneralLeCam.lean`, `InformationBound.lean` | Arbitrary probability measures and randomized measurable tests; exact TV testing identity, measurable data processing, KL/Bretagnolle–Huber lower bound and finite iid products, including infinite KL and empty products. |
| Concentration | `Concentration.lean` | Genuine bounded independent Hoeffding and adapted martingale-difference bounds. Paired benefits in `[-1,1]` use twice the unit-interval radius. No nonlinear evidence-ratio or empirical-Bernstein rate theorem is claimed. |
| Measurable target-law frontier | `MeasureTarget.lean`, `MeasureFrontier.lean` | Actual label kernels on arbitrary measurable input spaces, unchanged labels off disagreement, preserved input/evidence laws and population loss integrals. Exact clipped strict frontiers over the full measurable correctness-field class, supported on the two predicted labels on disagreement, without an assumed `RichAt`. |
| One-bit/channel extension: partial | `MeasureSwap.lean`, `ChannelCounterexample.lean` | General label-swap/channel invariance and opposite-risk impossibility; a verified counterexample to orbit-selection sufficiency. Set-theoretic sign factorization requires consistency on the entire evidence fibre. This is not a measurable decoder construction or a proof of the historical H/ratio-rate extension. |

All new modules are under `KBound/Probability/`. The target-law equivalences
require measurable predictors/kernels, positive disagreement probability,
`-1/2 <= M <= 1/2` and nonnegative residual budget. Their clipped interval is
`[max(-1/2, M-beta), min(1/2, M+beta)]`; large budgets never require impossible
correctness probabilities. An arbitrary restricted deployment subclass is not
automatically rich, and unrestricted multiclass label kernels are not identified
with this two-prediction-supported construction.

## Retained finite/algebraic spine

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
| anytime / Ville finite core | `KBound/Probability/EProcess.lean`, `Ville.lean` | historical deterministic one-step wealth and pointwise indicator bounds; complemented by `FilteredVille.lean` |
| one-bit swap involution | `KBound/Dichotomy.lean` | `evidence_swap_involution`, `swap_flips_benefit_preserves_evidence` |
| rate / Hoeffding bridge | `KBound/Probability/Rates.lean` | radius nonnegativity and conditional commit implication; complemented by `Concentration.lean` |
| multicandidate algebraic core | `KBound/Multicandidate.lean` | `multiclass_routing_harm_equiv` |
| three-world multiclass harm core | `KBound/ThreeWorld.lean` | `multiclass_harm_iff_nonpos` |

Full index: `KBound/TheoremMap.lean`

## Explicit external or unmechanized assumptions

- Benchmark exchangeability, calibration transfer, preprocessing, independence,
  conditional nulls and risk alignment are not certified by these proofs.
- Covering an observed batch difference is not automatically population-risk
  coverage. Reusing a marginal conformal interval is not an anytime guarantee.
- `TargetLaw.lean` retains its historical finite `RichAt` lift; the new general
  construction is separate and does not certify arbitrary benchmark subclasses.
- Selecting one world from each label-swap orbit need not orient the entire
  evidence fibre. The counterexample has two orbits, identical evidence, and
  opposite selected signs. The historical H model and evidence-ratio rates need
  separate correction/proofs and are excluded from the compact submission.

## Toolchain

- Lean 4.29.1 (`lean-toolchain`)
- Mathlib v4.29.1 (`lakefile.lean`)
