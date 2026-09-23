# Active theorem review

Review date: 2026-09-23. Frozen baseline: `69548cf44bcbede86053ea3ef6572ef57b2a1615`.
Revision workspace: `kbound-theorem-assumptions-20260923`. The public frozen release is unchanged.

## Result and meaning

The maintained compact and TMLR manuscripts contain **13 named mathematical results**: four theorems, two lemmas, six propositions and one corollary. All 13 statements and their full printed proofs were read, their principal boundary cases were checked, and their alleged Lean counterparts were inspected at the declaration level. The companion register records each claim's assumptions, exact statement and proof hashes, proof authority, formal names and remaining correspondence boundary. Seven further mathematical claims used by these results or their interpretation were also checked.

**No substantive contradiction was found in these 13 proofs under their explicit model and sampling premises.** That is a bounded manual review finding. It is not an unconditional correctness certificate or a claim that every printed sentence has a kernel-checked equivalent. In particular, the frozen baseline's 150 registered checks are not 150 manuscript theorems and do not establish the truth of empirical assumptions.

The audit identifies precise scope gaps rather than treating all mathematical claims as equally formalized. At the frozen baseline, four named results have a scoped measurable mechanization, six have mechanized components or a narrower special case, and three have manually reviewed LaTeX proofs without an exact registered counterpart. The new extended-radius module closes the certificate's previously identified scope gap. The revision therefore has **five scoped mechanizations, five partial mechanizations and three LaTeX-only results**. The formal registry now has 162 declarations; this increase does not change the manuscript's 13-result inventory. The JSON register is authoritative for the final revision mapping.

## Named-result coverage (including the new radius theorem)

| Active label | Mathematical review | Formal scope in this revision |
|---|---|---|
| `lem:fibre` | The patched binary label kernel realizes each measurable correctness field and preserves input evidence. | Actual measurable joint construction; batch and independent auxiliary-evidence lifting remains a printed-to-Lean instantiation step. |
| `lem:reduction` | The contrast vanishes on agreement and equals `2 eta-1` on disagreement. | Exact reduction for the constructed kernel laws; arbitrary binary-law representation is a manual correspondence step. |
| `lem:nonid` | Constant kernels `1/2 +/- delta` give opposite benefits within the full residual class. | Measurable opposite-target construction; no richness claim for a narrower empirical class. |
| `cor:matched-abstain` | Common action probabilities and dual marginal error control yield `1-2 alpha`. | Deterministic forced abstention and real-number arithmetic; no single probability-law capstone for the randomized rule. |
| `prop:closed-band` | The boundary zero-benefit world rules out strict commitment; the zero-budget/zero-margin case is handled. | Zero-target and frontier results; the probability clause has the preceding assembly gap. |
| `thm:headline` / `thm:frontier` | Exact strict frontier follows from the clipped attainable interval. | Measurable correctness-field frontiers, with explicit representation and full-class scope. |
| `thm:compact-beta-minimax` | A common audit distribution and continuity from above prove the supremum floor. | LaTeX-only exact fibre-supremum argument. |
| `thm:certificate` / `thm:cert` | Either erroneous strict direction entails interval noncoverage, also for random infinite radii. | The new `RandomRadiusCertificate.lean` directly handles arbitrary random extended-nonnegative radii, infinity abstention, measurable coverage and both marginal error directions; coverage remains an explicit premise. |
| `thm:population-transfer` | Width-two Hoeffding plus cell coverage, triangle inequality and union bound. | Concentration and deterministic decision components; no single full conditional-episode capstone. |
| `prop:cell-fail` | The exact two-point construction has population benefit `-0.10`, perfect cell coverage and ADAPT probability `0.45`. | LaTeX-only specific probability model; related unit-mismatch lemmas are not its exact formalization. |
| `thm:app-pop-cert` | The P1-P5 episode protocol implies split-conformal cell coverage and fresh-sample population concentration. | Conformal, concentration and action components; actual episode construction, unbounded quantile branch and compound probability assembly remain explicit correspondence work. |
| `prop:sampled-frontier` | Condition on each positive disagreement count, apply width-one Hoeffding, then average; zero count abstains. | Generic Hoeffding and population-frontier components; random-count thinning and the combined procedure are not one registered theorem. |
| `prop:nextphase-transport` | On simultaneous exact binomial containment the true joint table is feasible, so its paired objective lies between extrema. | LaTeX-only statistical/LP proof, separate from numerical endpoints and weak-dual computation. |

## Boundary cases and proof obligations checked

The population reduction needs `d>0` to define its conditional quantities. At `d=0`, the predictors agree almost surely and unconditional benefit is zero. The beta-free identity is binary: arbitrary multiclass predictions can both be wrong on disagreement. The construction allows the two-prediction support restriction, not arbitrary multiclass identification.

The exact feasible normalized-benefit interval is the intersection of `[-1/2,1/2]` and `[M-beta,M+beta]`. It is nonempty for feasible `M` and `beta>=0`. Constant correctness kernels attain all its points. At `beta=0`, nonzero `M` has a determined sign, while `M=0` gives zero benefit. At `|M|=beta`, the zero witness is essential: the claim is not that opposite nonzero benefits always exist there. For `beta>1/2`, no feasible margin lies strictly outside the band. None of these cases requires invalid correctness probabilities or a postulated rich class of actual deployments.

For the audit floor, `|gamma|<=1` bounds the supremum. With a nonempty fibre, an increasing sequence of attainable residual magnitudes converges to its supremum; the common-law threshold events decrease to the required event. Their probability bound survives the limit. The zero-supremum case follows directly from the nonnegative audit. The printed equality `Gamma(C_beta)=beta` is valid for `0<=beta<=1/2`. At fixed `M` the general radius is `min(beta,1/2+|M|)`, so extending equality to every beta would be false. The constant Gamma audit is an oracle bound for the fixed fibre, not a data-driven estimator of that fibre.

Coverage-to-action uses strict signs. An interval endpoint equal to zero causes abstention; an infinite radius also causes abstention for finite prediction/benefit. Both error events lie in the **same** coverage-failure event, so their union is bounded by alpha. This does not supply error control conditional on acceptance or repeated/adaptive decision control.

The population bridge uses accuracy contrasts of range two. Its Hoeffding radius is `sqrt(2 log(2/delta)/m)`, not the smaller unit-range accuracy radius. Capping at two is valid because both the cell average and population benefit are in `[-1,1]`. The union bound does not require independence between conformal coverage and concentration events. It does require positive sample size and genuinely fresh evaluation observations for the frozen pair. Macro-F1 does not inherit this paired-mean argument.

Split-conformal rank coverage requires the calibration and next residual scores to be jointly exchangeable under the actual fitting and candidate-construction protocol. Ties are conservative. If the requested rank exceeds the number of calibration scores, infinity is necessary; finite rank clipping has no general nominal-coverage justification. For the sampled-margin alternative, the correct proof conditions on each `n_D=k>0`, uses the conditional law on disagreement, and averages over counts. Fixed-size marginal inference does not justify optional stopping or choosing the predictor after seeing the evidence sample.

The paired-transport probability argument permits dependent multinomial entries because its joint containment uses a union bound. Source class counts may be conditioned on. An absent class receives `[0,1]`. True zero target class mass causes no division in the coverage proof. Realizing arbitrary LP points as sharp finite models additionally requires a normalized feasible source column for zero-mass classes. The exact set is bounded and closed, so nonempty-set extrema exist; on the joint interval event the true table proves nonemptiness. No inverse or full-rank matrix assumption is needed.

The numerical weak-duality formula was checked algebraically. With `lambda<=0`, `lambda^T Bx>=lambda^T b`; with `x` in the unit cube, each residual product is at least its negative part. These objective bounds are valid for every exactly feasible point of the supplied coefficient problem. They do not prove nonemptiness, certified inverse-beta evaluation, or exact LP feasibility of a floating-point witness. A one-ULP inverse-beta expansion remains a numerical qualification.

## Concrete corrections and priorities

The revision implements the premise-clarity improvements identified by review: explicitly require positive integer `m` in both population-transfer statements, declare nonnegative integer calibration count `N`, and state the score/predictor independence and common-law fresh-input premises next to the sampled-margin formula. The existing proofs already use these conditions; this is a clarification of their domain, not a new result.

The revision adds the general measurable random extended-radius coverage-to-action statement, including the infinite-radius abstention branch. Independent source review checked the infinity-before-`toReal` conversion, strict zero boundaries, pointwise soundness, measurable coverage and shared error event. No defect was found. This directly closes a known printed-to-Lean scope gap without claiming calibration transfer. The next independent formalization targets, if pursued later, are the randomized matched-evidence probability capstone, exact audit-floor supremum theorem, conditional episode compound-coverage theorem and random-count sampling lemma. The paired-transport theorem and certified numerical quantiles/optimization would be a substantially larger project. None should be hidden behind the registry total.

For statistical assumptions, a proof may establish a conditional implication without evidence that its premise holds in a benchmark. A separate assumption register must distinguish structural model restrictions, enforceable design conditions, external transport/residual budgets and empirical diagnostics. This theorem review does not promote outcome-disjoint cross-fitting to exchangeability, poor held-out inclusion to calibrated coverage, or retrospective benchmark cells to fresh population observations.

## Excluded and refuted historical extensions

The historical assertion that choosing one representative from each label-swap orbit suffices for sign identification is false. `ChannelCounterexample.lean` supplies two selected worlds from different orbits with the same evidence and opposite benefits. It remains excluded.

The replacement Boolean fibre-consistency theorem is set-theoretic. It does not assert the existence of a measurable decoder on an arbitrary observation space. The H/ratio-rate result only propagates a supplied nonnegative domination contract. It neither estimates H nor proves the contract for a benchmark. Consequently, a successful strict-core audit must not be renamed full historical-foundations completion.

## Verification limits

This review inspected source and proof contents; it did not rerun experiments or independently rerun the Lean kernel. The new strict formal receipt reports PASS for 162 registered declarations with a clean axiom audit. This reviewer independently matched its receipt hash and all 47 source snapshot entries. The complete source-bound inventory checker belongs to root integration. The register's raw theorem-environment hash is the SHA-256 of the exact UTF-8 `\\begin{...}` through matching `\\end{...}` bytes, including both delimiters. The associated proof hash covers the complete following proof environment. File hashes must be refreshed after any accepted manuscript clarification. These bindings establish which text was reviewed, not that mathematical or empirical premises are true.

The receipt is `formal_random_radius_strict_receipt.json`; its exact source binding is `formal_random_radius_source_snapshot.json`. Those files are also bound in the theorem register. The inventory reviewer read and hash-checked this evidence but did not independently rerun the compiler a second time.
