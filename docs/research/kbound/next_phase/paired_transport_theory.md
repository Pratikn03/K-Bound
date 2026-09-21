# Conditional paired-benefit transport certificate

Status: implemented research extension, with a synthetic diagnostic locked internally before execution; no public preregistration is claimed. This note makes no deployment, empirical natural-shift, or unrestricted identifiability claim. The construction uses established confidence-region and linear-programming machinery; the application is the paired decision, with an explicit sensitivity assumption.

## Evidence and assumptions

Fix two classifiers before observing the evidence samples: frozen `f0` and candidate `fa`. Alternatively condition on an independent training/adaptation sample that fixes both classifiers. Each source or target evidence example produces an observable bin `r=(f0(X),fa(X))`. Retain every bin in the declared support, including bins with zero observations. Coarsening bins is permitted only if the loss contrast remains constant in each `(r,y)` cell. Classes are `y=0,...,K-1`, and

`c[r,y] = 1{fa(r)=y} - 1{f0(r)=y}`.

Let `A[r,y]=P_source(r | y)` and `q[r]=P_target(r)`. Source counts are independent identically distributed conditional observations within each class, with fixed or conditioned-on class counts; target counts are independent identically distributed draws from the target bin distribution. Fixing the candidate by adapting on these same target evidence examples does **not** meet this assumption. An independent adaptation/evidence split is one possible implementation. The current synthetic experiment fixes all probability tables and classifiers before sampling.

The crucial scientific assumption is an externally justified bound

`sum_y pi[y] TV(P_target(r | y), A[:,y]) <= rho`,

where `pi` is the unknown target prior and `rho` is declared before decisions. A class with zero target mass contributes zero. At `rho=0` this is label shift at the prediction-pair level, weaker than equality of full feature distributions. Positive `rho` is a sensitivity analysis, not a fitted quantity or an assumption test. Unlabeled evidence cannot in general verify it. Full feature-space conditional TV at most `rho` implies this bound by data processing; the reverse implication need not hold. The API requires an assumption-contract identifier to release a directional action; a string does not prove the contract true.

## Simultaneous finite-sample probability region

For `R` bins there are `M=RK+R` binomial probabilities. Construct two-sided Clopper–Pearson intervals for each at error level `alpha/M`. A source cell uses the number of observations in its labeled column as its denominator; a target cell uses the total target count. A class with no source observations gets `[0,1]` in every cell. The multinomial cells need not be independent: the union bound gives simultaneous containment with probability at least `1-alpha`. Conditioning on source class counts and then averaging preserves this bound. Multiple independent candidate claims require another allocation or a simultaneous construction; this module certifies one fixed pair per invocation.

The statistical theorem concerns the mathematically defined binomial intervals. Software uses SciPy inverse-beta quantiles and expands them outward by one binary64 unit. That expansion protects ordinary rounding but is not a formal certification of the inverse-beta algorithm. The numerical LP certificate described below is rigorous for the supplied finite floating-point interval endpoints; it does not turn numerical quantile evaluation or sampling assumptions into a formally verified theorem.

## Probability-table polytope and coverage theorem

Introduce target joint probabilities `t[r,y]`, transported source joint probabilities `s[r,y]`, a prior `pi[y]`, and nonnegative absolute-value slacks `v[r,y]`. Impose

```
sum_y pi[y] = 1
sum_r t[r,y] = sum_r s[r,y] = pi[y]
L_A[r,y] pi[y] <= s[r,y] <= U_A[r,y] pi[y]
L_q[r] <= sum_y t[r,y] <= U_q[r]
v[r,y] >= t[r,y]-s[r,y],  v[r,y] >= s[r,y]-t[r,y]
0.5 sum_(r,y) v[r,y] <= rho
0 <= t,s,pi,v <= 1
```

The upper bound on `v` loses no feasible table because `|t-s|<=1` and minimal slacks suffice. All constraints are linear: confidence endpoints are constants multiplying `pi`.

**Theorem.** Under the assumptions above, let `[ell,u]` be the minimum and maximum of `sum c[r,y]t[r,y]` over this polytope. On a simultaneous probability-interval event of probability at least `1-alpha`, the true target benefit belongs to `[ell,u]`. Consequently releasing ADAPT only if `ell>0` and FREEZE only if `u<0` has unconditional wrong-direction probability at most `alpha`. The ideal mathematical rule abstains on an empty polytope or an interval touching zero. The implementation abstains on detected infeasibility, solver failure, or an interval touching zero; approximate primal checks do not certify exact nonemptiness or detect every empty set. The statement does not bound error conditional on acceptance, cumulative online error, or performance of a candidate selected using these same counts.

**Proof.** On the probability-interval event, take the true target prior and joint table, set `s[r,y]=A[r,y]pi[y]`, and `v=|t-s|`. Marginal, box, and source constraints hold by construction. The declared TV assumption gives the final constraint. Thus the true table is feasible, and extrema enclose its benefit. A wrong strict sign therefore implies failure of the simultaneous interval event. Empty sets cannot occur on that event under the assumptions. No invertibility or full rank is used. ∎

**Sharpness scope.** When nonempty, the finite table polytope is compact, so both exact LP endpoints are attained. Conversely a feasible `(s,t,pi)` defines source conditionals `s[:,y]/pi[y]` for positive `pi[y]`; for zero-mass classes choose any probability column inside the source box, whose intersection with the simplex is required. Together with the target joint table this realizes the stated finite table model. Hence these endpoints are sharp for this input polytope. This is not an optimal confidence procedure, a smallest confidence region among all procedures, attainability by the original feature-space model, or a claim that outward-rounded numerical endpoints themselves are attained.

## Numerical verification and failure behavior

Each minimization has form `min c'x` subject to `B x<=b`, `E x=d`, and `0<=x<=1`. Given any finite dual candidates, replace inequality multipliers by `lambda=min(lambda,0)`. For arbitrary equality multipliers `mu`, put `r=c-B'lambda-E'mu`. Weak duality and the unit box imply

`c'x >= b'lambda+d'mu+sum_j min(0,r[j])`.

This follows directly by substituting the residual identity, using negative inequality multipliers, and minimizing each residual term over `[0,1]`. The implementation does not trust near-zero stationarity: it pays for every residual coordinate. Directed 60-digit Decimal operations start from exact binary64 coefficients, upper-bound each stationarity sum, then lower-bound the residual correction and constant term. Conversion back to binary64 expands down by `nextafter`. This provides a conservative lower bound for the exact LP defined by those binary64 inputs. Maxima are obtained by negating a separately verified lower bound for the negative objective.

Approximate primal residuals and primal/dual gaps are also checked (`1e-7` residual, `1e-6` maximum gap, `1e-8` tolerance for negative approximate gap). These are quality/failure checks, not an additional statistical error term, and do not by themselves certify primal feasibility or exact nonemptiness. Corrupt, nonfinite, failed, or inconsistent solver output causes abstention. Numerical sign decisions additionally use a `1e-10` margin. The returned primal table is an approximate diagnostic witness, not a formal rational feasible point. Directed dual bounds contain the objective of every exact feasible point even when the approximate primal point is not exactly feasible; they do not certify that an exact feasible point exists. Exact decimal constructions below are rational mathematical examples; binary64 point-box checks are not proofs of exact nonemptiness.

The break-even sensitivity program removes the `rho` constraint, adds `c't<=0`, and minimizes `0.5 sum v`. It returns a verified **lower bound** on the smallest allowance admitting nonpositive benefit. The approximate primal objective is retained as diagnostic evidence only; the API deliberately does not label it an upper bound.

## Nonidentification can coexist with a positive decision

For three classes and bins `(2,0),(2,1),(2,2)`, take

```
A = [[.2,.2,0], [.8,.8,0], [0,0,1]], q = [.2,.8,0].
```

At zero sensitivity, feasible priors are `(p,1-p,0)` for every `p in [0,1]`: the rank-two matrix does not identify the prior. The benefit is `.2p+.8(1-p)`, so its exact interval is `[.2,.8]`; ADAPT is identified. Sign identification does not require parameter identification.

The exact break-even TV for this example is `1/6`. To see the lower bound, write `a=pi[0], b=pi[1], g=pi[2]` and let `d` be the moved mass in the first two source columns. Their source gain is `.2a+.8b`; their target positive-gain mass is at least `.2a+.8b-d`. The third column contributes at worst `-g` because the target has zero mass in the third row. Nonpositive benefit requires `d >= .2a+.8b-g`. Moving the entire third source column out of row three costs `g`, so TV is at least `g + max(0,.2a+.8b-g) = max(g,.2a+.8b) >= max(g,.2(1-g)) >= 1/6`. A witness at `pi=(5/6,0,1/6)` puts target column zero masses `(1/6,2/3,0)` and column two masses `(1/30,2/15,0)`, yielding zero gain and TV `1/6`.

The relevant identification criterion is the exact feasible interval excluding zero. A symmetric reach condition is not an equivalence when the interval is asymmetric: `[.2,.8]` has positive sign throughout although at its lower endpoint absolute reach is `.6 > .2`. This guards against reusing a stronger, invalid symmetric formula from legacy theory files.

## Paired objective versus separate accuracy intervals

For any common feasible set, `min(accuracy_a-accuracy_0) >= min accuracy_a-max accuracy_0`, and the corresponding upper inequality is reversed. Direct paired optimization therefore produces a contained interval before numerical rounding. The separate-accuracy arm in this experiment optimizes both accuracies on exactly the same polytope and then subtracts intervals. It is an explicit coupling relaxation, not an implementation or claim of superiority over MaC-LP.

An exact example has bins `(2,0),(2,1),(0,0),(2,2)` and

```
A = [[.1,.1,0], [.4,.4,0], [.5,.5,0], [0,0,1]], q=[.1,.4,.5,0].
```

The first two columns coincide. With zero sensitivity, the paired interval is `[.1,.4]`, while separately subtracting accuracy extrema gives `[-.1,.6]`. Agreement-bin correctness varies with the unidentifiable prior but cancels in the paired objective. This shows a decision benefit from respecting dependence, not new optimization machinery.

Using prior `(.8,.2,0)` gives true benefit `.16`. Preserve the same source table and target bin marginal while assigning every target label to the frozen prediction. The target benefit becomes `-.5`. No procedure receiving only those observed counts can distinguish these two worlds; a positive certificate in the latter world violates its declared transport assumptions. The diagnostic includes this world and a mixture, using identical random count streams to demonstrate the boundary directly.

## Literature, novelty limits, and practical scope

[Li, Han and Ma, arXiv:2609.14802v1](https://arxiv.org/abs/2609.14802) (submitted 13 September 2026) construct finite-sample label-shift confidence regions using probability intervals and matrix constraints, extracting intervals by linear programming. This is direct prior art for the central computational recipe. The present extension changes the estimand to paired benefit, preserves joint prediction outcomes, and exposes an aggregate conditional-TV sensitivity budget. It does not originate interval-constrained LP inference or establish general superiority over their method. The primary article and a targeted title/arXiv code search on 21 September 2026 did not locate an author implementation; the internally locked experiment marks official MaC-LP execution unavailable rather than substituting a proxy.

[Jhawar and Wang, arXiv:2609.11235v1](https://arxiv.org/abs/2609.11235) (submitted 10 September 2026) examine identifiability of adaptation from unlabeled evidence and indistinguishable situations with different optimal decisions. The distinction between identifying an action and identifying a distribution is shared territory. The finite table construction here supplies one assumption-explicit statistical decision procedure; no first claim for the general indistinguishability or action-identification principle is justified.

[BBSE](https://arxiv.org/abs/1802.03916) supplies earlier confusion-matrix label-shift estimation context. [Learn then Test](https://arxiv.org/abs/2110.01052) supplies an alternative way to validate a fixed policy using labeled calibration outcomes. Group/conformal calibration can control specified calibration risks under its own exchangeability assumptions; it does not establish the target transport assumption from unlabeled data. These are complementary assumption regimes, not universally ordered methods.

Useful deployment would require a defensible transport budget, adequate labeled conditional source counts, an independent target evidence sample after adaptation, and resource accounting including candidate adaptation cost. Confidence-region width, decision frequency, false directions, regret, and assumption violations must all be reported. A constant certificate radius within a fold merely changes the threshold on a fixed point predictor and does not change its matched-exposure ranking. The paired-table method can exploit different information, but this synthetic study alone does not establish natural-shift or deployment value. Novelty remains tentative pending broader review.
