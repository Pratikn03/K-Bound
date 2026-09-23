# Design-based bounds with additional labels

## Population and information access

Fix a finite frame of N records, their frozen and adapted predictions, their correctness rule, and all scores and bins before drawing audit indices. Conditional on this frame and its fixed true labels, let I_1,...,I_m be independent uniform indices sampled with replacement. For any fixed indicator h, h(I_j) are iid Bernoulli with success probability N^-1 sum_i h(i): each marginal follows by counting the indices and the joint law factors by independence of the index draws. Correlation among the original images does not change this finite-frame randomization statement. It does prevent interpreting N images as N independent future environments.

This is an audit-assisted procedure with additional label access. It is not KGA residual calibration and does not validate the historical label-free method. Correct labels, faithful record identities, a fixed frame and valid randomization remain premises. Software records make the implementation reviewable; they do not prove historical nonaccess, external label truth or physical random-source integrity.

## Accuracy benefit, including set-valued labels

For each record let W_i be adapted correctness minus frozen correctness. W_i belongs to {-1,0,1}. For a fixed disagreement stratum of fraction d, all records outside the stratum have W_i=0. Conditional stratum sampling therefore targets Delta=d(p_plus-p_minus). Two binomial intervals, each with failure probability at most alpha/2, give

    Delta in d [L_plus - U_minus, U_plus - L_minus]

with probability at least 1-alpha. Independence between the two intervals is unnecessary. Set-membership correctness is permitted: both predictions can be correct or both incorrect on disagreement. In that setting p_minus is not generally 1-p_plus. No F1 inference follows because F1 is not this paired mean.

For a single-label binary task on disagreement, exactly one prediction is correct. A single interval [L,U] for adapted correctness gives Delta in 2d[L-1/2,U-1/2]. If the fixed full-stratum score mean is s_bar, the residual gamma=p-s_bar is in [L-s_bar,U-s_bar], and

    beta_upper=max(|L-s_bar|,|U-s_bar|)

is an upper confidence bound for |gamma|. This is a random labeled-audit bound, not an unlabeled identification of beta or a guarantee over an arbitrary residual class. If d=0, Delta=0 and the conditional residual is undefined. The implementation reports zero benefit and an absent residual bound, without sampling an empty stratum.

## Transport with sampled source and target labels

Fix a common single-label class universe and bin definition. Sample with replacement from each of two finite frames. Conditional on a class count, the corresponding bin counts have binomial marginals for that frame's class-conditional bin probabilities. Allocate alpha/(2RK+K) to each source/target class-bin interval and target-prior interval. Conditional coverage averaged over class counts, followed by a union bound, gives simultaneous coverage at least 1-alpha. Dependence among multinomial entries or between frames is harmless for this union bound; each frame's own sampling law still matters.

For source and target intervals [a_lo,a_hi] and [b_lo,b_hi], define

    v_y=min(1, (1/2) sum_r max(|b_lo-a_hi|,|b_hi-a_lo|)).

On simultaneous containment, TV(A_y,B_y)<=v_y. Maximize sum_y pi_y v_y over the simplex intersected with target-prior boxes. Exact rational greedy allocation starts at all lower bounds and assigns remaining mass in descending v_y order. An exchange argument proves optimality: shifting any available mass from a smaller to a larger coefficient cannot lower the objective. Its value is an upper confidence bound for the aggregate conditional-TV budget. An empty prior polytope returns the universally valid bound one rather than a success claim.

If a source class has zero finite-frame probability, its conditional law is undefined. The procedure covers every explicitly declared normalized extension of that column, with TV bounded by one; it does not infer a unique column. Multilabel memberships do not form this class partition and are rejected for transport inference.

The failure budget of this random transport bound must be added to that of a subsequent paired-transport interval. Independent samples between the stages are sufficient but unnecessary for the union bound if each asserted event has its stated marginal coverage. Adaptive reuse or multiple decisions needs an explicit further simultaneous allocation.

## Numerical and validation scope

Binomial endpoints are supplied by the separately reviewed outward-tail certificate. SciPy proposes a point; directed positive-polynomial enclosures verify that the proposed endpoint is outward. Resource/precision limits give a wider bound. Exact rational arithmetic is used for residual, scaling, total-variation and simplex calculations. Statistical coverage still follows from binomial sampling and exact test inversion, not from numerical tests alone.

Exhaustive small-sample tests enumerate draw sequences and compare inclusion probabilities; adversarial tests reject corrupted schemas and retain both-correct memberships. These are implementation checks, not a formal proof of every line of Python or a demonstration of field utility. The new Lean modules concern the separately identified manuscript claims. This finite-frame derivation is a written mathematical argument, not represented as a complete Lean mechanization.

The authorized real-record demonstration uses all previously opened CCT-20 evaluation cells. It simulates a new label budget over their frozen outputs, reconstructs the published paired accuracy metric, reports every outcome, and retains its retrospective status. It neither accesses the stopped So2Sat target nor supplies a new independent deployment.
