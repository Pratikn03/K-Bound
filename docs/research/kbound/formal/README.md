# K-Bound Lean 4 Formalization (Mathlib)

Lean proofs for the K-Bound theory spine and explicitly scoped probability
foundations live here. A proof establishes its encoded statement under its
assumptions; it does not validate a dataset or guarantee an experimental gain.

## Proposition 3: separately compiled transfer evidence

The maintained conditional cell-to-population proposition maps to
`KBound.PopulationTransfer.pointwise`, `population_coverage`, and
`false_direction_probability` in [PopulationTransfer.lean](KBound/Probability/PopulationTransfer.lean).
The [source-bound receipt link](population_transfer_receipt_link.json) preserves
their successful pinned direct-compilation/axiom output from the authenticated
2026-09-05 workbench receipt. Current source hashes match that record. These
supporting declarations are not individually named in the central registered
audit list; this link does not change its historical **150** count or claim a new
build. The original complete receipt is private; the linked selected record is
not a substitute for the planned portable standard-Lake release receipt.

Both coverage premises remain assumptions, not dataset findings. The link test
`tests/test_population_transfer_receipt_link.py` checks the named source and
receipt identities; it is not a Lean replay or a population-safety certificate.

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

The current registry contains **303 declarations: 65 legacy core results and 238
additional registered results, including supporting lemmas**. Other supporting lemmas are also
compiled; the registry count is not a measure of novelty or scientific quality.
The older 150-, 161- and 221-declaration receipts retain their original counts. The release
runbook writes its kernel/axiom receipt to
`../audits/formal_foundations_2026_08_31.json` and binds it in the outer checksum
inventory. That receipt distinguishes five mechanized layers from the partial
sixth layer.

### Actual-law compatible-fibre radius (2026-09-07)

`ActualFibreRadius.lean` adds four independently reviewed results: actual residual
attainment, the supremum over the full actual fixed-marginal joint-law class,
exact intersection with the reference augmented-law fibre, and its radius
identity. Positive disagreement and0<=beta<=1/2 remain explicit. The actual-class
supremum uses no sampling premise; augmented-law identification uses the fixed
observation model. No arbitrary subclass or empirical radius estimate is inferred.

The current [303-declaration strict receipt](formal_actual_fibre_radius_audit_2026_09_07.json)
and [source/review/test binding](actual_fibre_radius_verification_20260907.json)
preserve those four exact proofs and six new public-root regressions, including
actual radius1/2 at M=0,beta1. Prior299 independent replay passed separately.
The independent31-row semantic census found no omitted named statement/display;
its A02 assembly gap is now proved. Literal paper sampling/nonempty wording,
A17/A21 author disposition and empirical/release gates remain unresolved.

### Literal feature-schema rank bound (2026-09-07)

`FeatureRank.lean` adds five independently reviewed results for the exact printed
coordinate order: the two difference relations define a nine-dimensional real
linear schema, and any satisfying row family has rank at most9. A zero-data
rank0 regression distinguishes this bound from observed exact rank9. Numerical
rank tolerance, probabilistic independence and physical feature realizability
are not proved.

The dated [299-declaration strict receipt](formal_feature_rank_audit_2026_09_07.json)
and [source/review/test binding](feature_rank_verification_20260907.json) preserve
that scope and five new public-root examples. The preceding294 integration also
passed independent replay. The actual-law radius capstone is addressed in the
separate303 stage; sampling/nonempty prose and author disposition remain separate.

### Weighted helpfulness and actual risk alignment (2026-09-07)

`WeightedHelpful.lean` adds three finite weighted score/regret results. The same
nonnegative weights and fixed per-cell helpfulness premise are explicit; means
require positive total weight. `RiskAlignment.lean` adds five results: strict
identification implies alignment on a fixed fibre, but genuine zero/positive
joint target laws at M=beta>0 show the converse fails. Positive disagreement,
the actual residual-defined full class and fixed observation model are explicit.
Neither result verifies empirical helpfulness or learned risk alignment.

The dated [294-declaration strict receipt](formal_weighted_risk_audit_2026_09_07.json)
and [source/review/test binding](weighted_risk_verification_20260907.json) record
these eight exact independently reviewed additions and eleven new public-root
examples. The286 integration also passed independent replay, preserved separately.
The rank proof is addressed in the separate299 stage above; observed rank/prose
obligations and exhaustive semantic review remain open.

### Construction-closed actual subclasses (2026-09-07)

`ActualWorldSubclass.lean` adds fourteen independently reviewed results using
the paper's legal opposite-tilt pair in the interior and the constant-half zero
construction only on the boundary. Actual residuals, benefits and admissibility
are derived, not assumed. The resulting strict-direction existence iff, maximal
commitment and fixed augmented-law/error-to-abstention composition cover
restricted actual target-law classes satisfying these membership conditions.
No interior zero-world membership or closure under every correctness field is
required. Individual ADAPT/FREEZE iff and uniqueness explicitly need a realized,
nonempty subclass. The fixed observation/policy model remains explicit.

The dated [286-declaration strict receipt](formal_actual_world_subclass_audit_2026_09_07.json)
and [source/review/test binding](actual_world_subclass_verification_20260907.json)
bind the exact reviewed source and ten new public-root examples, including a
two-world interior class excluding zero. The minimal pair/zero membership
conditions follow from paper construction closure; they are not an equivalence
to every possible construction. Sampling/nonempty-uniqueness prose, ancillary
clauses and exhaustive semantic review keep the [current ledger](PAPER_CLAIM_MATRIX.md)
qualified; no whole-paper or empirical closure is inferred.

### Actual-world augmented-evidence assembly (2026-09-07)

`ActualWorldEvidence.lean` adds eight independently reviewed results: actual
interior score-class witnesses and preserved off-disagreement kernels, equality
of the actual augmented `(Z,M)` law, the strict-direction iff on that fixed-law
fibre, and the actual closed-band zero-world/error-to-abstention composition.
One fixed marginal-based observation experiment and one fixed Markov policy are
explicit. The finite iid joint-sample pushforward realization is proved; arbitrary
world-dependent batch or seed couplings are not inferred from separate marginals.

The dated [272-declaration strict receipt](formal_actual_world_evidence_audit_2026_09_07.json)
and [source/review/test binding](actual_world_evidence_verification_20260907.json)
bind that exact reviewed source and nine new public-root examples. The later
restricted-subclass interface is recorded in its separate286 stage above.
Ancillary/prose obligations and exhaustive semantic review remain qualified.

### Full actual-world strict-frontier components (2026-09-07)

`ActualWorldFrontier.lean` adds ten independently reviewed theorems. The universal
class contains all actual binary joint probability laws with the fixed input
marginal and a bound on their actual score residual. Constructed correctness
fields supply necessity/attainment witnesses, not a restriction on that class.
The module proves strict ADAPT/FREEZE iff frontiers, the clipped benefit interval,
nonemptiness, closed-band zero attainment, the zero-margin/budget case and the
printed rule's pointwise action semantics.

The dated [264-declaration strict receipt](formal_actual_world_frontier_audit_2026_09_07.json)
and [source/review/test binding](actual_world_frontier_verification_20260907.json)
bind that reviewed integration and eight public-root examples. Its later augmented-evidence
assembly and restricted-subclass interface are recorded in their separate stages above.

### Arbitrary-joint correctness-kernel/score bridge (2026-09-07)

`JointKernelScore.lean` adds eight independently reviewed theorems. The actual
conditional label kernel of every binary joint probability law supplies a
measurable correctness field, with no Standard Borel assumption on the input
space. Its disagreement integral equals the actual correctness event mass;
bounded scores supply integrability and the exact score/residual/benefit/sign
identities. Conditional-probability interpretation requires positive disagreement
mass. At zero mass only the stated totalized algebra/event identities apply.

The dated [254-declaration strict receipt](formal_joint_kernel_score_audit_2026_09_07.json)
and [source/review/test binding](joint_kernel_score_verification_20260907.json)
bind the reviewed source. Five public-root examples include both correctness
endpoints, arbitrary input spaces, zero disagreement and dependent binary laws.
The full actual-class components were subsequently integrated as described above;
target-witness evidence/probability composition, ancillary clauses, prose disposition
and exhaustive review remain separate.

### Evidence-to-action transport (2026-09-07)

`EvidenceTransport.lean` adds eight independently reviewed theorems. Equal input
marginals lift to equal input-batch evidence laws under explicit finite iid
sampling. A fixed Markov policy, or a fixed measurable rule with an independent
common seed, gives equal actual action laws and the matched-world abstention
bound. Separate seed/evidence marginal equalities do not suffice; the sampling
design and common joint observation law remain explicit.

The dated [246-declaration strict receipt](formal_evidence_transport_audit_2026_09_07.json)
and [source/review/test binding](evidence_transport_verification_20260907.json)
bind the reviewed integration. Three public-root transport examples complement
the six conformal and four earlier paper examples. Assembly with target-law
witnesses, the full actual-world score-defined frontier, ancillary clauses and
exhaustive semantic review remain separate obligations.

### Exact conformal construction (2026-09-06)

`ExactConformal.lean` adds 17 independently reviewed theorems: the literal ceiling
rank, a measurable min/max specification of the k-th calibration residual with
ties, actual infinity when the rank exceeds calibration size, and marginal
coverage under full score-law exchangeability. Its actual absolute-residual
capstone also bounds the union of both strict directional errors. The radius
ignores the held-out score when all other scores remain fixed; this does not
audit pipeline label custody. Generic real inputs are clamped by `ofReal`; exact
paper correspondence is for the nonnegative absolute residuals, not arbitrary
signed-real quantiles. The finite powerset construction is a mathematical
specification, not an efficient production-quantile implementation claim.

The dated [238-declaration strict receipt](formal_exact_conformal_audit_2026_09_06.json)
and [source/review/test binding](exact_conformal_verification_20260906.json) record
the integrated bytes. `ExactConformalExamples.lean` imports only the public root
and checks six general-law and boundary examples. The [current claim ledger](PAPER_CLAIM_MATRIX.md)
retains the remaining score-kernel/world and target-witness compositions,
ancillary clauses and prose qualifications; paper-wide closure remains open.

### Reviewed current-paper proof clauses (2026-09-06)

Five independently reviewed modules add all 60 explicitly authored theorem/helper
declarations: `JointTargetReduction` (9), `PaperCounterexamples` (17),
`DependentSignFlip` (20), `IndependentSignFlip` (1), and `PaperDecisionAlgebra` (13).
They prove arbitrary-joint-law disagreement reductions, actual finite probability
witnesses for calibration and cell/population failures, the nine-coordinate shared-sign
counterexample, invariance under independent symmetric coordinates, and decision/regret
algebra. `PaperProofExamples.lean` imports only the public root and checks the
arbitrary binary joint-law reduction and the actual-law counterexample interfaces.

The [221-declaration strict receipt](formal_paper_clauses_audit_2026_09_06.json)
and [dated source binding](paper_clauses_verification_20260906.json) preserve that
earlier integration stage. The literal conformal construction was subsequently
closed as described above; complete joint correctness-kernel/score and target-witness
composition, ancillary compositions, prose disposition and exhaustive
semantic review remain open. No empirical premises or publication/release gates are closed.

### Randomized action-law and extended-radius interfaces (2026-09-06)

`RandomizedActionLaw.lean` proves the abstention bound for measurable decision
rules on arbitrary world sample spaces. Equal action pushforwards, opposite
fixed benefits, and the two world-specific directional-error bounds imply
abstention probability at least `1 - 2 * alpha` in both worlds. Equality of the
action laws is an explicit premise; equal random-seed marginals alone do not
establish the common joint law needed by a randomized rule.

`ExtendedRadiusCertificate.lean` allows a measurable nonnegative radius to take
finite and infinite values on the same probability space. Infinity abstains;
finite real nonnegative radii agree with the original certificate. Marginal
coverage for the declared finite real target bounds the measurable union of
both directional errors by alpha, and therefore bounds each separate error.
Zero truth counts as an error for either strict direction. No independence,
conditional coverage, or almost-surely finite radius is required.

These two modules add eleven registered theorem declarations. The integrated
`InterfaceExamples.lean` checks the 10-percent-error/80-percent-abstention case,
the zero-benefit boundary, a probability law on mixed finite/infinite branches,
and five pointwise radius cases. The exact earlier 161-declaration build and source binding are
recorded in [the integration verification index](probability_interfaces_verification_20260906.json)
and [the combined strict-audit receipt](formal_interfaces_audit_2026_09_06.json).
These records do not change historical receipts or certify empirical premises.
They do not establish availability before evaluation-label access or
population-risk alignment.

### Confirmatory-v2 audit-floor addition (2026-09-05)

`AuditFloor.lean` proves the randomized common-law floor over any nonempty
bounded residual fibre, including an unattained supremum. It requires a
measurable nonnegative audit and explicit equality of the joint observation/seed
law across worlds. `AuditFloorCorrectness.lean` separately establishes the full
correctness-field radius identity `Gamma = beta`, with a genuine target-law
realization, positive disagreement mass, feasible margin and `0 <= beta <= 1/2`.
An arbitrary restricted deployment class does not inherit this equality.

A fresh **direct pinned Lean compile** passed for all 44 local modules. Actual
transitive axioms were checked for all 150 registered declarations and 15 added
supporting/example declarations, with only the three standard axioms above.
Independent review checked the source/receipt identities and supplied additional
compiled proofs. This is not a standard Lake build, does not refresh the older
release receipt, and does not certify empirical assumptions. The new
[verification index](../../../../protocols/confirmatory_v2/FORMAL_VERIFICATION_20260905.json)
records the precise scope and retained evidence identities. Historical sixth-layer
and release-rebuild gaps remain open.

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
