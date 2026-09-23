# Active-source theory audit

Audit date: 2026-09-22. Scope: active `kbound_submission_body.tex`, its inputs
`paper/sections/theory_core_main.tex` and `theory_certificate.tex`, and
`kbound_submission_supplement.tex`, especially `app:nextphase-paired-proof`.
`theory_setup.tex` is not the authority. Source hashes and the audited Git HEAD
are in `theory_source_snapshot_journal_2026_09_22.json`; concurrent manuscript
integration requires a refreshed binding before release.

**Result:** no substantive correction was required in the inspected active
statements. The conclusions are conditional and narrower than unrestricted
multiclass identification, real-world calibration, or full historical-foundation
closure. The formal README required correction because it described completed
measurable probability layers as still absent.

## Population identity, frontier, and audit floor

| Case / obligation | Finding |
|---|---|
| Measurability and fixed candidate | `ass:deploy` fixes measurable binary predictors and score, assumes a measurable correctness kernel and d=P(D)>0. Candidate fitting randomness must be conditioned on appropriately; adapting on scoring inputs does not supply fresh-sample concentration. |
| d=0 | Explicitly excluded from the conditional margin definition. The unconditional benefit is zero because the predictors agree almost surely; neither strict sign is justified. Do not divide by d or assign an identified conditional M in this case. |
| Binary reduction | On disagreement precisely one prediction is correct, giving Delta=2d(M+gamma). On agreement the loss contrast vanishes. General multiclass predictions can both be wrong, so this identity is not inherited without the stated support restriction. |
| Feasible margin endpoints | M is in [-1/2,1/2]. Constant correctness kernels realize every z in [max(-1/2,M-beta),min(1/2,M+beta)]. Clipping is essential at endpoints and for large beta. |
| beta=0 | gamma=0; nonzero M fixes the sign. M=0 gives Delta=0 and strict abstention. No opposite-world assertion applies to beta=0. |
| Interior | For beta>0 and abs(M)<beta choose 0<delta<min(beta-abs(M),1/2); constant kernels 1/2 +/- delta realize opposite nonzero signs with identical evidence. |
| Boundary | abs(M)=beta>0 admits gamma=-M (zero benefit) and gamma=0 (strict sign M). No opposite nonzero signs are asserted here. Strict soundness fails on the zero world even though both operational choices have equal task risk. |
| Large beta | The frontier remains valid for every beta>=0; beta>1/2 precludes abs(M)>beta. Feasible residuals are [-1/2-M,1/2-M]; no construction uses correctness outside [0,1]. |
| Richness | Necessity and maximality quantify the full declared kernel class, with fixed observables and off-D kernel. For a restricted subclass only sufficiency transfers automatically. The real benchmark class is not proved rich. |
| Randomized rules | Shared evidence plus independent randomization yields identical action probabilities. Dual marginal error bounds imply adapt<=alpha and freeze<=alpha, hence abstain>=1-2alpha, including zero-benefit boundary worlds. This is not an optimal regret theorem. |
| Risk alignment | The definition excludes opposite nonzero signs in a fibre; a fibre containing zero and positive benefit is risk-aligned yet does not certify a strict positive direction. Active text makes this distinction. |
| Supremum audit floor | Nonempty fibre and identical law of the nonnegative audit suffice. Increasing attainable absolute residuals approach Gamma; continuity from above for events beta_hat>=r yields the probability bound even if the supremum is not attained. Here abs(gamma)<=1, so the supremum is finite. |
| Beta range in equality | Gamma(C_beta)=beta is stated only for beta in [0,1/2]. More generally at fixed M the full-class radius is min(beta,1/2+abs(M)); extending the printed equality to arbitrary beta would be false. The constant Gamma audit is an oracle fibre benchmark, not a finite-data estimator. |
| Source calibration | Score calibration or top-label calibration need not survive conditioning on D. The active manuscript explicitly permits gamma!=0 without distribution shift and does not estimate beta from epsilon. |

## Calibration, action, and population bridge

The coverage-to-action result is event containment. It accepts a measurable
random radius in [0,infinity] and finite benefit/prediction; at infinity both
strict tests fail and the rule abstains. When an endpoint equals zero the rule
also abstains. A zero benefit counts as a directional error only upon commitment.
Both wrong-direction events lie in the same coverage-failure event; their union
is therefore also bounded by alpha, although the proposition prints separate
bounds. Marginal control does not imply conditional error given acceptance,
family-wise protection across candidates, time-uniform protection, or protection
under unverified calibration transfer.

Split conformal uses k=ceil((n+1)(1-alpha)); k>n requires infinity, not rank
clipping. Ties preserve conservative rank coverage. Calibration counts the
declared evaluation units, and fitting/candidate construction must leave the
new residual exchangeable with calibration residuals. Dependence within an
accuracy cell differs from exchangeability between cells. Leave-one-out fitted
residuals do not automatically have the same split-conformal guarantee.

For fresh independent accuracy draws W in [-1,1], Hoeffding gives
P(abs(mean(W)-Delta)>sqrt(2 log(2/delta)/m))<=delta. Capping the radius at 2 is
valid because the maximum absolute difference is 2. A union bound with cell
coverage requires no independence between these two events. m must be a
positive sample count; m=0 is outside the formula. The supplement imposes a
frozen candidate and independent evaluation draws, and alpha_cell+delta<1
for a nonvacuous statement. Macro-F1 is not a mean of these W variables and
inherits no such radius. The sampled-margin alternative conditions on n_D;
it abstains for n_D=0 and requires fresh iid inputs and a fixed score/pair.
Historical transductive panels do not establish these population premises.

## Paired-transport proof and numerical boundary

The source-class and target-bin intervals jointly contain the true probabilities
with probability at least 1-alpha by Clopper–Pearson and a union bound over
RK+R entries. Independent multinomial entries are not required. All declared
bins, including zero-count bins, remain. A missing source class receives [0,1]
and does not receive invented calibration information. Class counts can be
conditioned on before averaging the guarantee.

On that event, t=P_T(r,y), s=A_ry*pi_y, and v=abs(t-s) satisfy the supplied
linear constraints under the externally declared aggregate conditional-TV
budget. pi_y=0 causes no division in the coverage proof. Realizing an arbitrary
feasible table as a sharp finite model additionally needs a normalized source
column in its box for a zero-mass class, as the text explicitly states.
The exact nonempty polytope is compact; extrema exist. Empty sets trigger
abstention, not a claimed sign. Singular or rank-deficient A needs no inverse;
an unidentified prior can still give a strictly positive paired objective.

The three-class example gives [.2,.8], and the stated zero-sign threshold 1/6
is consistent with its lower bound and attaining table. The cancellation example
gives paired [.1,.4] versus separate [-.1,.6]; this follows by optimizing a
contrast on one feasible set, not by assuming independent error bars. Shared
agreement uncertainty cancels. The observationally identical negative world
with benefit -.5 violates transport, showing why rho cannot be validated using
only the same unlabeled counts. Multiple candidates/time points require a new
simultaneous construction or budget allocation.

Weak-dual bounds certify objectives for every exactly feasible point of the
supplied binary64 problem; residual/gap checks do not establish exact primal
feasibility or nonemptiness. Inverse-beta endpoints expanded by one float ULP
are not a formally certified Clopper–Pearson implementation. Thus the exact
statistical statement and numerical certificate remain explicitly distinct.
No general feature-space sharpness or theorem-to-solver end-to-end proof is
claimed.

## Correspondence and exclusions

The updated `../formal/README.md` gives theorem-to-module/name correspondence.
In particular, `MeasureTarget` and `MeasureFrontier` establish measurable
construction and strict frontiers without the older `RichAt` premise;
`MeasureConformal`, `FilteredVille`, `GeneralLeCam`, and `Concentration` provide
actual probability layers with explicit assumptions. The old finite-rank module
alone is not an exchangeability proof. The manuscript's arbitrary random,
extended-real radius is not one identical named certificate theorem (fixed-real
corollaries plus a general measure event-containment lemma exist). The exact
fibre-supremum floor and paired-transport LP are audited LaTeX arguments, not
claimed one-to-one additions to the 150-declaration Lean registry.

The unrestricted historical one-bit orbit claim remains refuted. Corrected
fibre consistency and conditional H-budget propagation do not recover it.
Strict-core success must not be relabeled full-foundations completion,
prospective empirical validation, or a priority claim.

## Inherited results and specific extra content

`theory_comparison.json` records eight primary-source comparisons and exact
paper labels. The added-content field is a scoped comparison, not a priority
assertion. The population contribution is the explicit clipped paired-benefit
set and strict-boundary/fibre-audit specialization; generic evidence-law
nonidentifiability already appears in [Jhawar and Wang, Definition 1 and
Theorem 1](https://arxiv.org/html/2609.11235v1). The paired transport construction
uses the matrix-constraint approach of [Li, Han and Ma, Theorem
3.2](https://arxiv.org/html/2609.14802v1), with a paired objective and aggregate
conditional-TV sensitivity budget. [MORPHEUS](https://theyoungkwon.github.io/papers/articles/danilowski_morpheus_iclr-ttu26.pdf)
is direct precedent for label-free performance prediction and selection; KGA's
interval operating point does not establish better ranking. Residual rank
calibration comes from [split conformal](https://www.stat.berkeley.edu/~ryantibs/papers/conformal-jasa.pdf);
its application does not establish [coverage under arbitrary
shift](https://www.stat.cmu.edu/~ryantibs/papers/nexcp.pdf). None of these
comparisons supports an unrestricted novelty or superiority claim.
