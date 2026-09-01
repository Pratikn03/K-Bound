# Multiclass vector-capacity: mathematical adversarial review

Date: 2026-08-31

Status: independent discovery-track review. No claim in this report is promoted to a Lean-verified theorem. Exact finite witnesses were checked with rational arithmetic; that is not a substitute for the program's Lean, parity, novelty, and clean-build gates. This review changes no current K-Bound manuscript, frozen result, target-data access policy, or Git history.

Source specification: `/Users/pratik_n/.codex/attachments/713b862e-738e-4035-bd3a-536839cc7f00/pasted-text.txt`.

## Bottom line

The program has a coherent corrected foundation, but two proposed implications must be narrowed before proof effort:

- T5 is false if a surviving observable-null-space contrast direction is its only premise. Benefit variation need not cross zero, and an affine direction can be infeasible on the realized simplex face.
- T3's unrestricted rank-increment identity does not remain an equality for every scientifically admissible supplement class. Moreover, a literal ban on every invertible answer encoding excludes all algebraically minimal supplements: conditional on existing information, their coordinates encode exactly the missing contrast coordinates.
- T2's row-space criterion is appropriate for uniform identification over the full product-simplex model. It is not a necessary criterion at every realized boundary fiber unless the actual fiber's affine hull is used.
- T6–T9 need a nontrivial commitment/power requirement. Controlling only wrong strict commitments permits an always-ABSTAIN policy with zero labels.
- T6–T8 need a precise feedback contract. Requested labels, returned labels, pending requests, and unlabeled arrivals are different resources. Predictable querying alone does not protect against outcome-dependent delays or invalid stopping rules.

A full-support, three-class, asymmetric-cost rare-stratum family remains a plausible route to strict active label savings. Its proposed rates are unproved in this track; they offer no corresponding order-of-magnitude saving in rare-stratum arrival time.

## 1. Common notation and quantifiers

Write the unknown conditional probabilities as a vector in

\[
P=\prod_{s=1}^{m}\Delta_K\subset\mathbb R^{mK}.
\]

Let \(B\eta=\mathbf 1\) contain every stratum's simplex normalization. Let

\[
M=\begin{bmatrix} B\\ A\end{bmatrix},\qquad
H_{j,(s,y)}=q_s g_{s,j}(y).
\]

Thus \(H\eta\), not the unweighted array of cost contrasts, is the benefit vector. Every rank and null-space calculation below includes \(B\). The matrices, costs, and stratum probabilities are fixed when stating a uniform identification result. If they depend on the observable, the relevant quantifier must be stated separately for each permitted observable.

For a realized right-hand side \(b\), the fiber is

\[
F_b=\{\eta\geq0:B\eta=\mathbf1,\ A\eta=b\}.
\]

All examples below have nonempty fibers, \(K=3\), normalized costs in \([0,1]\), and exact rational parameters.

## 2. T5 counterexample: null-space variation without a sign change

Witness ID: `ADV-T5-NOCROSS-001`.

Use one stratum of mass one, one adapter, \(A=(0,0,0)\), and \(b=0\). Set the frozen and adapter cost columns to

\[
C_0=(1,3/4,1/2),\qquad C_1=(0,0,0).
\]

Then \(g=(1,3/4,1/2)\), and the entire simplex is feasible. The vector

\[
v=(1,-1,0)
\]

satisfies \(Av=Bv=0\), but \(g\cdot v=1/4\ne0\). Nevertheless,

\[
\min_{\eta\in F_b}\Delta(\eta)=1/2,
\qquad
\max_{\eta\in F_b}\Delta(\eta)=1.
\]

Both extrema have exact certificates: \(e_3\) and \(e_1\) attain the bounds, while \(g-(1/2)\mathbf1\geq0\) and \(\mathbf1-g\geq0\) certify them. Every admissible world strictly favors ADAPT. No opposite-decision pair exists in this fiber.

This example simultaneously shows that point identification can fail while sign identification succeeds. A nonzero null-space effect establishes neither ambiguity of the sign nor a label-free decision-error lower bound.

## 3. T2/T5 boundary counterexample: an infeasible affine direction

Witness ID: `ADV-T2-FACE-001`.

Use one stratum of mass one and

\[
A=(1,1,0),\quad b=0,\quad
C_0=(1,0,0),\quad C_1=(0,0,0).
\]

Nonnegativity and \(\eta_1+\eta_2=0\) force

\[
F_b=\{(0,0,1)\}.
\]

The affine null vector \(v=(1,-1,0)\) satisfies \(Mv=0\) and \(g\cdot v=1\). Moreover, \(\operatorname{rank}(M)=2\), whereas \(\operatorname{rank}([M;g])=3\). Yet the realized benefit is point identified at zero. Neither \(e_3+t v\) nor \(e_3-t v\) is feasible for any \(t>0\).

The exact LP certificates use the equality right-hand side \((1,0)^\top\). For minimizing \(g\eta=\eta_1\), dual multiplier \((0,0)\) gives lower bound zero. For maximizing it, multiplier \((0,1)\) gives \(M^\top(0,1)=(1,1,0)\geq g\), also with objective zero. The primal witness is \(e_3\).

This is not a counterexample to a properly quantified uniform-over-all-fibers T2. Other right-hand sides for the same \(A\) have nontrivial variation. It is a counterexample to reusing that uniform criterion as a necessary condition at this one realized fiber.

The issue is not limited to singleton fibers. For two three-class strata with masses \((1/2,1/2)\), impose \(A=(1,1,0,0,0,0)\), \(b=0\). Let the first-stratum cost columns be \((1,0,0)\) and \((0,0,0)\), and the second-stratum columns be \((1/2,1/2,1/2)\) and \((0,0,0)\). Then \(F_b=\{e_3\}\times\Delta_3\) is non-singleton, and \(\Delta\equiv1/4\). Nevertheless, \(v=(1,-1,0,0,0,0)\in\ker M\) has \(Hv=1/2\). Thus a non-singleton boundary model can have a point-identified, strictly positive benefit despite a surviving unrestricted affine contrast direction.

### Corrected T2 statements to formalize

For fixed real matrices \(A,H\) on the full product simplex, the intended uniform statement is

\[
\bigl[\forall\eta,\eta'\in P,\ A\eta=A\eta'\Rightarrow H\eta=H\eta'\bigr]
\iff
\ker M\subseteq\ker H
\iff
\operatorname{row}(H)\subseteq\operatorname{row}(M).
\]

The necessity argument uses the relative interior of the product simplex: a sufficiently small positive and negative perturbation of the uniform conditional distribution is feasible along any vector in \(\ker M\). This argument is unavailable on an arbitrary boundary fiber.

At one nonempty realized fiber, the correct object is instead

\[
L_b=\operatorname{span}\{\eta-\eta':\eta,\eta'\in F_b\}.
\]

Point identification there means \(H L_b=\{0\}\). An equivalent finite-polytope route augments \(M\) by coordinates forced to zero throughout the fiber. Define

\[
Z_b=\{i:\eta_i=0\text{ for every }\eta\in F_b\}.
\]

The proposed exact affine-hull identity is

\[
L_b=\ker\begin{bmatrix}M\\E_{Z_b}\end{bmatrix}.
\]

Here \(E_{Z_b}\) selects the forced-zero coordinates. A proof can average finitely many feasible points to obtain a point positive on every coordinate outside \(Z_b\), then take small two-sided perturbations within that minimal face. These are corrected proof targets, not completed Lean declarations.

## 4. T3: unrestricted dimension versus admissible information

### 4.1 The unrestricted algebraic statement

The rank increment

\[
r=\operatorname{rank}\begin{bmatrix}M\\H\end{bmatrix}
-\operatorname{rank}(M)
\]

is the proposed minimum number of arbitrary additional scalar linear rows needed to make \(\operatorname{row}(H)\subseteq\operatorname{row}([M;L])\). Independence must mean independence modulo \(\operatorname{row}(M)\), not merely ordinary independence among the added rows. State the finite-dimensional real vector space and the unrestricted supplement class explicitly.

This is a dimension calculation. It does not show that the required moments are scientifically available, exactly observable without labels, inexpensive to estimate, or expressible as that many admissible primitive measurements.

### 4.2 Exact primitive-moment counterexample

Witness ID: `ADV-T3-PRIMITIVE-001`.

Use \(A=(0,0,0)\), \(b=0\), \(B=(1,1,1)\), and cost columns

\[
C_0=(1,0,0),\quad C_1=(0,1,0),\quad g=(1,-1,0).
\]

The unrestricted rank increment is one. Predeclare the admissible primitives to be class probabilities \(\eta_1,\eta_2,\eta_3\), rather than the contrast itself. No one primitive identifies the benefit:

| Known primitive | First feasible vector | Second feasible vector | Respective benefits |
| --- | --- | --- | --- |
| \(\eta_1=1/3\) | \((1/3,2/3,0)\) | \((1/3,0,2/3)\) | \(-1/3,+1/3\) |
| \(\eta_2=1/3\) | \((0,1/3,2/3)\) | \((2/3,1/3,0)\) | \(-1/3,+1/3\) |
| \(\eta_3=1/3\) | \((2/3,0,1/3)\) | \((0,2/3,1/3)\) | \(+2/3,-2/3\) |

Two primitive probabilities, for example \(\eta_1\) and \(\eta_2\), suffice. Thus the primitive minimum is two while the rank increment is one. The counterexample also covers arbitrary single class-event probabilities: for three classes, a nontrivial event is a singleton or the complement of one. Equivalently, one event indicator has at most two levels, whereas \(g\) has three distinct levels modulo an additive constant.

For a fixed, predeclared dictionary \(\mathcal D\), the corrected admissible quantity is

\[
k_{\mathcal D}
=\min\{|S|:S\subseteq\mathcal D,
\operatorname{row}(H)\subseteq\operatorname{row}(M)+\operatorname{span}(S)\},
\]

with value infinity when no such subset exists. The rank increment is a lower bound, not an automatic equality. A dictionary consisting only of the first class probability in the example makes uniform identification impossible regardless of repeated measurement of that same population moment.

### 4.3 Literal answer-encoding prohibition

In the quotient of linear functionals by \(\operatorname{row}(M)\), the contrast rows span an \(r\)-dimensional space. Any sufficient supplement with exactly \(r\) independent new rows spans that same quotient space. Its missing-information coordinates and the independent benefit coordinates are therefore related by an invertible linear transformation conditional on the existing observable moments.

Consequently, a literal prohibition on every invertible encoding of the answer rules out every algebraically minimal supplement when \(r>0\). The specification should distinguish:

- an unrestricted algebraic benchmark, which may be answer-equivalent and carries no scientific-availability claim; and
- a predeclared, independently justified primitive measurement class, whose minimum may exceed the benchmark.

Primitive target-label moments can be legitimate measurements without being available label-free. Exact population moment access and finite-sample estimation must remain different statements. This review makes no novelty claim for the unrestricted dimension identity.

## 5. Corrected T5 impossibility and abstention tradeoff

The required sign condition is a genuine feasible crossing:

\[
\min_{\eta\in F_b}\Delta(\eta)<0<
\max_{\eta\in F_b}\Delta(\eta).
\]

A useful sufficient construction is a feasible \(\eta_0\) with \(\Delta(\eta_0)=0\), a vector \(v\in\ker M\) with \(Hv\ne0\), and some \(t>0\) for which both \(\eta_0\pm tv\) are feasible. None of these feasibility or crossing premises can be silently omitted.

Witness ID: `ADV-T5-PAIR-001`.

For \(q=1\), \(A=(0,0,0)\), \(b=0\), and \(g=(1,-1,0)\), take

\[
\eta^+=(1/2,1/4,1/4),\qquad
\eta^-=(1/4,1/2,1/4).
\]

Both distributions have full support and the same unlabeled observable law. Their benefits are respectively \(+1/4\) and \(-1/4\). Use the same normalized cost columns as in Section 4.2.

For any label-free randomized rule, let \(a,f,u\) be its common probabilities of ADAPT, FREEZE, and ABSTAIN under these indistinguishable worlds. Wrong strict commitment has probability \(f\) in the positive world and \(a\) in the negative world. Therefore the proposed exact tradeoff is

\[
\max\{a,f\}\geq\frac{a+f}{2}.
\]

If commitment probability must be at least \(\kappa\), worst-case wrong strict commitment is at least \(\kappa/2\). If commitment is mandatory, the randomized minimax error is at least \(1/2\), attained on this pair by a fair action coin. If abstention carries no error loss and there is no power requirement, always abstaining has zero wrong-commit probability. Equivalently, requiring wrong-commit probability at most \(\delta\) in both worlds forces commitment at most \(2\delta\) on this pair.

With nonidentical observation laws, the standard two-point proof target is an average forced-binary error lower bound \((1-\mathrm{TV}(P^+,P^-))/2\); a minimax lower bound follows from that average. Do not call it an equality for every asymmetric experiment, and do not transfer it unchanged to a loss that makes abstention free. None of these probabilistic statements is promoted before its stated Lean proof and parity check.

## 6. T6–T9 need a commitment task, not safety alone

Specify both safety and usefulness. One suitable one-adapter target is:

- Safety: for every permitted target, the probability of asserting a wrong strict sign is at most \(\delta\). Under a strict-sign convention, ADAPT when \(\Delta\leq0\) and FREEZE when \(\Delta\geq0\) are unsupported strict commitments.
- Power: for every target in a declared separated class \(|\Delta|\geq\gamma>0\), the probability of a correct strict commitment by a declared budget is at least \(1-\beta\).

Alternatively, require a \(\delta\)-correct forced binary decision only on the separated class. State whether the budget is fixed, high-probability, or expected at a stopping time. An always-ABSTAIN policy must fail the usefulness requirement. For multiple adapters, the analogous power requirement needs a declared unique-best or pairwise-gap condition.

For passive labels, one direct observation is \(Z_j=g_{S,j}(Y)\), with mean \(\Delta_j\) and range within \([-1,1]\). A bounded-mean upper-bound proof can therefore estimate the benefit directly; learning every conditional class probability is not automatically necessary. A dependence on the minimum stratum probability cannot be introduced merely by choosing an unnecessarily strong all-strata estimation objective.

Candidate variance-sensitive upper bounds have terms of the form \(V_j/\gamma_j^2\) and a range term divided by \(\gamma_j\), with simultaneous confidence accounting. They are not universal matching lower bounds. Dimension and candidate-count dependence must be derived for a specified contrast family. Boundary cases also need exploration/range terms: zero conditional variance alone does not imply zero observations are sufficient to discover an unknown deterministic label.

## 7. Feedback and resource contract for T6–T8

Report at least these separate quantities at stopping:

| Resource | Meaning |
| --- | --- |
| \(T\) | Number of unlabeled arrival opportunities consumed |
| \(Q\) | Number of labels requested or purchased |
| \(R\) | Number of requested labels that have returned |
| \(Q-R\) | Outstanding label requests |
| Calendar time | Arrival process and delay model applied to the above |

Counting only \(R\) can conceal arbitrarily many outstanding purchased queries. If revealed-label count is the chosen information metric, report \(Q\) alongside it rather than treating the two as interchangeable.

The passive comparison must mean that labeled strata are drawn indiscriminately from \(q\). The active pool comparison may observe the current stratum and choose whether to request its label before observing that label. A so-called passive baseline that can wait for a rare stratum and label only that stratum already has the selection ability responsible for the separation below. Calling it passive does not preserve a label-complexity separation.

For adaptive queries, probabilities must be measurable from the returned history and current permitted unlabeled features. They must not use the current label, future labels, or unreturned feedback. Importance-weighted estimators need correct predictable propensities and positivity on relevant contributions. A fixed stratum-conditional estimator is a legitimate alternative when its sampling assumptions hold; it need not be artificially importance weighted. Optional stopping requires an appropriate sequential confidence construction, not repeated application of a fixed-sample confidence interval.

### Delay assumptions are substantive

Witness ID: `ADV-DELAY-001`.

Use one stratum, \(g=(1,-1/2,0)\), and

\[
\eta=(1/8,5/8,1/4),\qquad\Delta=-3/16.
\]

Suppose all queries are chosen predictably, but class-1 labels return immediately and other labels return only after a long delay. Before those other returns, a returned-only mean sees exclusively contrast value \(+1\), despite a negative population benefit. Query predictability alone does not prevent this bias. If delays never return or their timing reveals outcomes, finite waiting guarantees or information lower bounds can also fail.

A clean initial model uses deterministic or otherwise noninformative delays, together with an explicit finite bound or tail assumption for wall-clock claims. If delays depend on stratum, observations available by a calendar cutoff need not have stratum law \(q\), even when delays are independent of the label conditional on stratum. Complete arrival-indexed blocks, stratum-conditional estimators, or valid censoring adjustments must address that fact. Information from return timing must either be excluded by assumption or included in the observation model.

Bounded delays need not multiply total query count when concurrent requests are allowed. A sequential one-outstanding-query restriction is a different model and can incur a different delay cost. The concurrency model must be fixed before comparing rates.

## 8. Exact asymmetric rare-stratum family

Witness ID: `ADV-T8-RARE-001`.

There are two strata, common \(c\) and rare \(r\), with

\[
q_c=1-\rho,\quad q_r=\rho,\quad 0<\rho\leq1/2.
\]

Use \(A=0_{1\times6}\), \(b=0\), and product-simplex normalization. The common conditional label distribution is \((1/3,1/3,1/3)\) in every world. Costs are fixed before any labels:

| Stratum | Frozen cost column | Adapter cost column | Contrast |
| --- | --- | --- | --- |
| Common | \((1/4,1/8,1/16)\) | \((1/4,1/8,1/16)\) | \((0,0,0)\) |
| Rare | \((1,0,0)\) | \((0,1/2,0)\) | \((1,-1/2,0)\) |

Thus the policies have equal nonzero safety costs in the common stratum and an asymmetric safety tradeoff in the rare stratum. For rational \(0<\varepsilon\leq1/8\), define

\[
\eta_r^+=(1/4+\varepsilon,1/2-\varepsilon,1/4),\qquad
\eta_r^-=(1/4-\varepsilon,1/2+\varepsilon,1/4).
\]

All three class probabilities are positive. Both worlds have exactly the same unlabeled stratum law, and

\[
\Delta(\eta^\pm)=\pm\frac32\rho\varepsilon.
\]

For the concrete rational instance \(\rho=\varepsilon=1/16\):

| World | Rare label vector | Frozen risk | Adapter risk | Benefit |
| --- | --- | --- | --- | --- |
| Positive | \((5/16,7/16,1/4)\) | \(5/32\) | \(77/512\) | \(+3/512\) |
| Negative | \((3/16,9/16,1/4)\) | \(19/128\) | \(79/512\) | \(-3/512\) |

The costs, stratum masses, and worlds are exact finite constructions, not data-selected favorable cost matrices. This is a synthetic theory family, not a natural-shift result.

### Candidate rates: CONJECTURE, unproved, no Lean promotion

Let \(L=\log(1/\delta)\), with power failure probability controlled at the same order. For the separated rare-stratum family, the intended matching orders are:

| Procedure/resource | Candidate order |
| --- | --- |
| Passive iid labels | \(L/(\rho\varepsilon^2)\) |
| Active labels, querying only the rare stratum | \(L/\varepsilon^2\) |
| Active unlabeled arrival opportunities | \(L/(\rho\varepsilon^2)\) |

The proposed lower-bound route is two-point information: conditional rare-label KL divergence is of order \(\varepsilon^2\); a passive observation has this information only with probability \(\rho\), whereas each rare-only queried label is informative. The proposed upper-bound route is a rare-stratum conditional contrast mean, with a bound on the number of rare observations in a passive budget. Naive global Hoeffding can introduce an extra factor \(1/\rho\); a conditional-count or variance-sensitive analysis is needed for matching.

This is a route to proofs, not a completed passive/adaptive lower-bound theorem. A sequential information argument, finite-confidence constants, stopping conventions, delays, and Lean formalization remain open.

With the same global margin convention in both experiments,

\[
\gamma=\frac32\rho\varepsilon,
\]

the candidate label rates are respectively \(\rho L/\gamma^2\) and \(\rho^2 L/\gamma^2\), in the stated regime \(\gamma\leq3\rho/16\). The prospective factor-\(1/\rho\) label saving is not an arrival-time saving: the active method must wait for rare strata.

It is invalid to keep a fixed positive global margin while sending \(\rho\) to zero in this normalized rare-only family. In general \(|\Delta|\leq\rho\) when the contrast is supported only on the rare stratum and its magnitude is at most one.

Under a declared bounded delay of \(D\) arrival steps and sufficient request concurrency, an additive waiting overhead is a plausible initial model. Unbounded or informative delays do not support that conclusion. A quota-stratified sampler with direct access to rare labels has a different arrival resource and must not be compared as if it were the same passive pool model.

### General allocation direction: also unproved

For one contrast, fixed independent labels \(n_s\) per stratum give the candidate conditional-mean variance expression

\[
\sum_s\frac{q_s^2\sigma_s^2}{n_s},
\qquad\sigma_s^2=\operatorname{Var}(g_s(Y)\mid S=s).
\]

The usual variance-allocation candidate is \(n_s\propto q_s\sigma_s\), leading to a variance term proportional to \((\sum_s q_s\sigma_s)^2/N\), compared with \(\sum_s q_s\sigma_s^2/N\) for passive proportions. This alone is not a finite-sample confidence theorem or a universal optimality statement. Unknown variances, boundary/exploration terms, admissible pool allocations, multiple contrasts, and sequential validity need separate treatment. A multi-adapter design would optimize the relevant worst pairwise contrast rather than assume one allocation is optimal for every candidate.

## 9. Zero-mass strata

If \(q_s=0\), the entire stratum block in \(H\) is zero. Its conditional label vector is an arbitrary nuisance representation of a null event, contributes no benefit, and cannot be sampled by waiting for arrivals from \(q\).

- Do not count its conditional degrees of freedom as required benefit-identification information.
- Do not divide by its mass in label or waiting bounds.
- Do not require positive query probability there.
- If structural equations involve its arbitrary conditional vector, justify that representation carefully or eliminate it existentially rather than treating it as target-label evidence.

Setting \(\rho=0\) in the rare-stratum construction makes both benefits zero; it is a tie, not a strict active-separation example. All positive-mass requirements in that family are explicit.

## 10. Validation receipt and recommended disposition

A fresh in-memory Python `fractions.Fraction` check completed with exit code zero and 17 exact checks. A further exact check verified the non-singleton two-stratum boundary variant. The checks covered the noncrossing null direction and extrema, boundary primal/dual witnesses, one-versus-two primitive moments, full-support opposite-sign worlds, normalized asymmetric costs, exact rare-family risks, the zero-mass benefit, and the informative-delay population contrast. No code, data, paper, or Git file was modified by those arithmetic checks.

These checks validate the listed finite arithmetic witnesses only. They do not establish the probabilistic rates, complete any Lean proof, establish novelty, or authorize promotion into K-Bound.

Recommended claim status:

- Retain T2 as a precisely quantified uniform criterion; add a separate minimal-face realized-fiber statement.
- Split T3 into an unrestricted algebraic benchmark and a primitive-dictionary problem. Withdraw unrestricted equality under an unspecified answer-encoding ban.
- Replace T5's null-survival premise by a feasible strict sign crossing and declare the commitment/abstention loss.
- Keep T6–T8 as conjectures under a fixed commitment, feedback, and resource model. The rare-stratum family is a nonvacuity candidate, not a verified rate result.
- Require the always-ABSTAIN, delayed-selection, zero-mass, and resource-accounting mutations before controller claims are promoted.
- Complete a separate literature collision audit before treating the corrected linear algebra or selective-label savings as a new scientific contribution.
