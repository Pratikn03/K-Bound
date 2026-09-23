# Statistical-assumption review and corrections

Audit date: 2026-09-23. Base: `69548cf44bcbede86053ea3ef6572ef57b2a1615`.
This is a new revision; the frozen release and original result authorities are preserved.

The active manuscript is appropriately conditional about its principal statistical guarantees. The remaining burden is to justify those conditions for a particular sampling design or deployment. Neither passing diagnostics nor adding formal lemmas establishes exchangeability, a nontrivial residual/transport bound, or independent future environments. The strongest concrete defect found in this review was in a legacy diagnostic API and report, not in the active coverage-to-action proposition.

The accompanying `assumption_register.json` records 20 assumptions and 20 study groups with source paths and SHA-256 bindings. Its categories distinguish structural premises, sampling design, external scientific restrictions, numerical scope and descriptive observations. It does not label a validator pass as scientific validation. Main/supplement inputs are the active authority; `paper/sections/assumption_contract.tex` is a historical inactive surface for this revision.

## Concrete defects repaired

1. **Unjustified guarantee and action in the legacy assumption audit.** `docs/research/kbound/kbound_pkg/assumption_audit/__init__.py` originally returned `guarantee_wording="applies"` and `recommended_safe_action="adapt"` whenever its support/drift warning did not trigger. This API receives no benefit interval or justified coverage premise. Identical unlabeled evidence can accompany either sign of benefit. A no-warning result now returns `unresolved` and `abstain`; warnings retain abstention. This is an operational fallback, not a claim that the frozen model is safe or best.
2. **Missing measurements represented as zero.** The legacy drift diagnostic used `0.0` when no residual sample was supplied. It now returns null and says “not evaluated.” Empty/nonfinite evidence, incomplete residual pairs and invalid thresholds fail explicitly rather than appearing not falsified.
3. **Rank and group-input failures in the maintained assumption helper.** `kga/assumptions.py` now rejects alpha outside `(0,1)`, nonfinite or non-vector calibration residuals and group vectors inconsistent with the row count. Empty calibration previously crashed while formatting a missing rank bound; it now leaves an infinite radius and diagnostic-only gate. Counting unique groups is explicitly described as counting declared units, not establishing independent observations.
4. **Coverage-basis record validation scope.** A `CoverageClaimBasis` carries an external argument. Its validation checks fields and inference-unit agreement, not file bytes or the truth of the proposed sampling model. Emitted diagnostics now state `syntax_and_protocol_record_consistency_only`, `artifact_bytes_authenticated=false` and `distributional_assumptions_established=false`. A caller's theoretical-coverage claim remains conditional. The pre-existing `CERTIFY` gate label alone is not proof of coverage; consumers must read the coverage claim/basis and its limitations.

The legacy generator now writes a separate, source-bound posthoc correction at `legacy_assumption_audit_replay_v2/`, refuses to overwrite an existing correction, and never overwrites the old v1 report/results. This is a diagnostic replay, not a new statistical experiment or a run under the original executed-source identity. The original five no-warning results and one drift warning remain; the old recommendation and guarantee language is withdrawn.

Those warning categories match, but the numerical diagnostics are a replay from the currently bound inputs, not a reproduction of every historical measurement. For example, the benign feature-range score is 0.105455 in this replay versus 0.111818 in the preserved v1 record, and the shifted score is 0.133636 versus 0.14. The original measurements are preserved in their original files; the corrected replay does not authenticate their producer history.

The generator has additional historical scope defects that cannot be repaired by changing the old reported number: its `concept_shift_witness` passes the same shifted feature array as `evidence_support_shift` and does not actually construct matched-evidence opposite-benefit worlds. The “helpful” and “low-margin” conditions have no benefit/margin inputs. Its single-feature support shift fails to trigger the configured warning. The correction explicitly preserves and discloses these facts. They are not successful validations of the intended stress questions.

Regression checks use an explicit indistinguishable-evidence binary construction: `f0=0`, `fa=1` and identical score/output evidence are compatible with benefit `+1` if the label is 1 and `-1` if it is 0. The diagnostic must withhold a guarantee in both cases. Additional tests cover absent observations, malformed ranks/groups, and preservation of historical and newly written evidence.

## What the mathematics does and does not require

**Coverage-to-action needs coverage, not a perfect predictor.** If the interval contains the declared benefit, a strictly positive lower endpoint cannot accompany nonpositive benefit. Both wrong-direction commitments are in the same coverage-failure event. Hence their union is bounded by alpha, under that premise. This is a marginal statement on the specified probability space and target. The main source does not claim that its historical empirical intervals have this property everywhere.

**Conditional acceptance is different.** If `p_A=P(ADAPT)>0`, the marginal false-ADAPT bound implies only `P(B<=0 | ADAPT)<=min(1, alpha/p_A)`. All errors may occur among rare accepted updates. When `p_A=0` the conditional quantity is undefined. No ADAPT exposure demonstrates neither successful selection nor an empirical zero conditional risk.

**The finite-rank limitation is about available empirical order statistics.** With `k=ceil((n+1)(1-alpha))`, a finite calibration rank exists only when `k<=n`. At alpha .10, at least nine calibration scores are needed. This statement does not mean nine units are sufficient for a useful or exchangeable design. The infinite-radius fallback still covers every finite target while abstaining; `n/(n+1)` is the largest distribution-free lower bound supplied by a finite empirical rank, not an upper limit on actual coverage. Ties can make coverage larger. The helper retains the legacy field name but now documents this distinction.

**Independence and exchangeability are not interchangeable.** Score exchangeability is the conformal premise; a fixed predictor plus exchangeable calibration/new units is one sufficient construction. Disjoint fitting/scoring records help protect against leakage but do not create exchangeability under geographic or corruption shift. Independent Bernoulli trials with different success probabilities are not the common-probability binomial model. Counting cities, checkpoints or unique group labels establishes a denominator, not the associated law.

**The population bridge needs a separate sampling argument.** The accuracy difference is a mean of fresh conditionally iid variables in `[-1,1]` for a fixed realized predictor pair. Hoeffding then supplies the sampling radius. The union bound with cell coverage does not require independence of the two success events. Adapting on the same evaluation inputs or using evaluation-batch normalization does not supply the required fresh-draw model. Historical `m=2000` replay is a sensitivity parameter, not a newly verified iid sample count. Macro-F1 has a different functional form and does not inherit this accuracy-mean radius.

**LOO and rank clipping remain distinct limitations.** Other LOO models can use a scored outcome even when its own residual is excluded. Ordinary split-conformal validity does not follow. A deterministic stability transfer requires *both* a bound between the relevant fits and a LOO coverage event. Thus a full-fit miss outside `epsilon+beta_stab` implies instability **or** a LOO coverage miss; it does not by itself prove instability. The inactive historical assumption-contract prose omitted that disjunction. The active source does not rely on that assertion. Historical B6 additionally clips required rank nine to eight, so the nominal 90% claim cannot be recovered by restoring provenance alone.

**Repeated decisions need another probability statement.** A fixed family can allocate error budgets and use a union bound only if each per-decision premise holds under the actual selection/sampling process. Marginal single-step coverage does not automatically persist after adaptive candidate selection, calibration reuse, monitoring or optional stopping. No production-wide or anytime guarantee is established here.

**Paired transport has three separate layers.** The exact theorem combines valid simultaneous source-conditional/target-marginal probability boxes with an externally justified aggregate conditional-TV budget and a fixed predictor pair. The true joint table is feasible on that joint event. The numerical solver separately supplies directed weak-duality bounds for the given binary64 LP. Neither solving the LP nor a small feasibility residual proves the statistical assumptions. The SciPy inverse-beta endpoints expanded by one ULP are not certified exact quantile enclosures; exact primal nonemptiness also is not established by floating-point tolerances. The manuscript's qualification should remain unless those numerical layers are proved and checked separately.

## Study-by-study conclusions

| Study | Supported interpretation | Premise or limit that remains |
|---|---|---|
| Current CIFAR Protocol B | Paired, candidate-dependent recorded-cell regret; outcome-disjoint saved-feature computation | Shared checkpoint/pools, six related corruption families, no verified deployment exchangeability; SAR favors always-adapt |
| ImageNet-C and SAR controls | Scoped matched-control and numerical perturbation evidence | Stream seeds are not newly trained models; perturbations test a computation, not every data-generating law |
| CCT-20 | Outcome-blind retention endpoint under the stated protocol | Label metadata used during ranking; cis-to-trans transfer unverified; zero ADAPT and one missed helpful candidate |
| Shared-predictor comparison | Local Tent loss advantage, simpler-rule wins elsewhere and transfer failure | 50.51–60.69% constant CIFAR inclusion vs nominal 90%; no independently excluded calibration environments |
| Conditional group-risk sensitivity | Explicit finite-sample-information limitation and held-out descriptive group rates | Common-probability iid group trials unestablished; 10% arm has zero exposure |
| So2Sat v1 | Negative development feasibility finding | No eligible calibrated/target result |
| So2Sat v2 | Completed 19-city calibration and failed locked screen | No target scoring; exchangeability and validity conditional on passing the screen not established |
| Population simulations | Finite synthetic observations under known generator assumptions | No universal theorem from empirical 100% coverage; no real-data population validation |
| 108-cell population replay | Arithmetic effect of an enlarged radius on existing records | No new iid draws or measured population-benefit authority |
| Paired-transport simulations | Conditional use of pairing, plus failure when rho is violated | External rho and fixed-sample conditions; numerical qualifications; no natural deployment evidence |
| Mixed-sign checkpoint LOO/B6 | Recovered decisions/splits support historical arithmetic | LOO dependence, B6 clipping and incomplete original producer identity; helpful-only authority unresolved |
| Office-Home canonical/stream replay | Conservative retention and descriptive point ordering | Unestablished uniform stability; no independent-checkpoint identity for stream seeds |
| Five-checkpoint Office-Home audit | Posthoc candidate opportunity only | Invalid multiclass routing construction, inadequate ranks and invalid serialization; route stays withdrawn |
| Camelyon17 | Historical candidate behavior and original no-ADAPT retention result | Opened target, related subsets/checkpoint; ground-truth-thresholded reconstruction stays withdrawn |
| RxRx1 | All-freeze retention replicated across model variation | No accepted useful updates or prospective selective routing |
| iWildCam | Audit history | Wrong archived F1 class convention and unsealed sample/runtime/population identities; numerical row withheld |
| PACS / ImageNet-R / CIFAR-10.1 | Transfer and architecture behavior diagnostics | Missing PACS residual replay, LOO/dependence limits, rare-accept conditional error, no prospective transfer |
| Native components / historical ports | Matched comparison only within the explicitly implemented component scope | Not every published learner or development-information budget is reproduced |
| Architecture/partial-adaptation probes | Synthetic configuration results | Named architectures/adapters not executed; null actions provide no routing evidence |
| Local deployment and cost profile | Software fallback behavior and workload-specific measurement | No statistical-premise validation, field reliability, useful production routing or net value |

For the 10% conditional group-risk arm, a zero-error one-sided binomial upper endpoint with family budget `.05/13` is `1-(.05/13)^(1/n)`. Reaching `.10` requires at least 53 exposed calibration groups. The largest available partition is 45. This precomputable limitation does not show every possible policy is unsafe. For So2Sat, the 18th of 19 city maxima targets marginal simultaneous coverage of five checkpoints for one exchangeable new city. It does not give simultaneous 90% coverage of ten cities or coverage conditional on passing the screen.

CCT's two-way product bootstrap and location sign-flip calculation use different uncertainty models. The former resamples checkpoint and location factors; the latter conditions on the evaluated checkpoints and needs joint coordinatewise sign invariance. Enumeration eliminates Monte Carlo approximation, not the premise. Independent zero-symmetric cluster gaps suffice; mere exchangeability or a global sign symmetry does not. Holm and Bonferroni do not repair invalid constituent tests or turn retrospective choices into preregistration.

## Historical evidence remains historical

The old `research_lock/assumption_reports/*.json` files are snapshots emitted from a historical `NUMBERS_PACK.json`, not complete current primary-panel assumption certificates. Their emitter-consistency tests check serialization/arithmetic agreement. In particular, different observed SAR seed harm rates can be evidence against a chosen model but do not logically prove nonexchangeability; the old “quarantined” designation must not be silently applied to the separately defined current Protocol-B panel. Original reports are left intact, with their role recorded here.

No source receipt, checksum, CI result or perturbation test authenticates missing original historical execution or proves a sampling assumption. Exact bytes support traceability; the scientific statement still needs its model and design.

## Remaining work before stronger statistical claims

- For each proposed deployment, name the target distribution, decision unit and required external beta/rho or calibration-transfer argument. Without that argument, retain conditional wording.
- For empirical validity, use a genuinely eligible fresh design with locked roles and sufficiently many independent environments. Existing opened benchmarks and repeated cells cannot be renamed fresh environments. No new target access was performed here.
- For acceptance-conditional or repeated-use protection, supply a separate theorem and matching procedure, rather than relabeling a marginal interval.
- For an end-to-end numerical paired certificate, validate probability-quantile enclosures and exact feasibility as well as the current directed objective bounds.
- Keep refuted/invalid and unsupported historical claims excluded. Their exclusion is a scientific resolution, not a failure to make a check green.

These are conditional requirements for stronger future claims. They are not reasons to hide adverse results or prevent a narrower journal paper from presenting what its evidence actually establishes.
