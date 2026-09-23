# Numerical certification of the paired-transport construction

The maintained implementation now replaces the inverse-beta one-ULP heuristic with outward confidence endpoints justified by binomial-tail inequalities, and requires an exactly feasible rational primal point before emitting a verified LP interval. This closes the specified numerical gaps for newly executed calls under this implementation. It does not authenticate or retroactively change the original 4,860-trial outputs, their executed source, or their reported numerical qualification.

## Probability endpoints

For `x` successes among `n` binomial trials and two-sided error budget `a`, the lower CP endpoint is the root of `P_p(X >= x)=a/2` for `x>0`. Any `p_lower` with `P_{p_lower}(X >= x)<=a/2` is no larger than that root and is a valid outward lower endpoint. The upper endpoint is obtained by the reflection `1-p_lower(n-x,n,a)`. Counts zero and n receive the corresponding exact 0/1 boundary. No observations return `[0,1]`.

`kga/transport_numerics.py` uses an untrusted SciPy quantile only to propose a point on a fixed 44-bit dyadic grid. Correctness of that quantile is never assumed. A proposed endpoint is retained only after a directed-arithmetic proof of the required tail inequality. A failed, nonfinite or unverified proposal can only widen the interval, using zero for an unverified lower endpoint and the corresponding reflected upper endpoint. The final dyadic endpoints are exactly representable in binary64, including the subtraction from one.

The tail is a sum of positive binomial terms. The first term is enclosed using the exact integer binomial coefficient and integer powers computed by repeated directed multiplications. Each subsequent term uses the exact positive recurrence `t_(j+1)/t_j=((n-j)/(j+1))*(p/(1-p))`, enclosed using directed multiplication and division. After term j, these ratios decrease. If a certified upper ratio r is below one, `t_j*r/(1-r)` bounds the remaining tail. Directed summation therefore gives a valid enclosure of the complete probability. The lower/upper tail bound is compared with the exact rational error budget. A precision-indeterminate comparison is never treated as success. A small-n exact integer-polynomial fallback resolves exceptional equality cases.

The implementation uses 80-digit Decimal contexts for directed addition, subtraction, multiplication and division, and integer-power repeated multiplication rather than transcendental Decimal functions. Counts above 100,000 trigger a declared full `[0,1]` interval, rather than an approximate unverified tail. This resource fallback preserves coverage and can make a downstream decision uninformative. Metadata records the method, exact alpha, precision, grid, caps and tail comparisons.

Bonferroni error allocation uses exact rational division of the declared family alpha. It does not rely on a possibly upward-rounded or underflowed floating `alpha/interval_count`. The legacy displayed float allocation is rounded down and may be zero for subnormal budgets; the actual probability calculations and metadata retain the positive exact rational budget. Python integer count totals prevent int64 overflow before the resource guard.

These are conditional binomial confidence bounds. They do not establish that an application supplies iid/common-probability binomial trials, correctly labeled source classes or fixed predictors.

## Exact nonemptiness and objective enclosure

SciPy/HiGHS proposes primal vectors and dual multipliers. Its numerical feasibility report and residual tolerance are no longer sufficient for a verified result. The code first attempts rational reconstruction, then exact rational elimination of an active basis selected from the numerical proposal. It checks **every** supplied inequality, equality and unit-box bound in exact Fraction arithmetic. Basis selection is heuristic; final feasibility checking is exact. If reconstruction fails or exceeds the 180-variable bound, the caller abstains with `exact_primal_not_certified`. This is not proof that the polytope is empty. A solver infeasibility status is explicitly named `solver_reported_infeasible`, with `exact_infeasibility_proved=false`.

The existing directed weak-duality bound remains the objective authority. It projects inequality multipliers to the proper sign and accounts for every stationarity residual over the known unit box. Consequently it encloses the optimum of the supplied binary64 LP independently of the quality of the solver's stationarity claim. An exactly feasible witness establishes nonemptiness; the weak-duality calculation encloses all feasible objectives. Numerical residual/gap checks remain additional sanity checks, not substitutes for these arguments. The full rational feasible vector and numerical coefficient semantics are serialized.

The break-even API still publishes a certified lower bound only. It preserves the historical API distinction rather than relabeling its approximate numerical objective as a rigorous upper endpoint.

## Input semantics

Ordinary binary64 probability inputs remain unchanged. Their exact dyadic values are the LP coefficients. A zero-width float column written as `[.2,.8]` does not sum to exactly one as rational binary64 numbers; it is rejected as an inconsistent boxed simplex. The code does not normalize it or silently widen that declared float point box.

An explicit `Fraction` or `Decimal` input denotes the intended exact rational value. Its probability lower bounds are rounded outward downward and upper bounds outward upward. The transport budget is similarly rounded upward. Metadata records every changed bound and the exact original rational value. This yields a documented outer relaxation of the intended rational probability model. A bound valid over this relaxation remains conservative for the intended model. Accuracy contrasts `-1,0,1` are exact; a non-binary-representable rational contrast is rejected because an objective-rounding enclosure is not supplied by this API.

## New verification and replay

`tests/test_transport_numerics.py` checks directed tail enclosures against exact rational polynomials, exact coverage on small finite binomial grids, corrupted quantile proposals, subnormal error budgets, count overflow, an LP contradiction smaller than numerical feasibility tolerance, exact rational reconstruction and explicit rational-input enclosures. Existing analytic transport, fault-injection and outcome-exclusion tests remain. Numerical and finite-frame targeted tests total **68 passes**, with Ruff lint/format checks passing for the eight reviewed files.

`transport_numerical_replay_01/transport_numerical_verification.json` is a new bounded numerical replay: one deterministic count draw for each of the nine preserved synthetic scenarios and three original sample sizes, evaluated at three rho values. All **81 cases** obtained exact feasible witnesses; outcomes were 16 ADAPT, 2 FREEZE and 63 ABSTAIN. On this host, maximum confidence-box time was 0.08621 seconds and maximum paired-LP solve time 0.02032 seconds. These are arithmetic workload measurements, not neural adaptation or production costs. Original study results are unchanged.

An independent scalar Fraction review regenerated all 27 confidence-box sets, checked all constraints for **162 saved primal witnesses** and recomputed their directed dual bounds. All six recorded numerical-source bindings matched. `transport_independent_review.json` binds this review to the exact numerical replay bytes. These checks support the declared numerical implementation; they do not establish novel theory, real-world transport validity or future effectiveness.

## Manuscript wording supported by this work

“The current implementation encloses binomial confidence endpoints by certified tail inequalities and verifies exact rational primal feasibility before issuing a paired-transport interval. Directed weak-duality bounds enclose the objective of the supplied binary64 LP; explicit rational inputs receive documented outward probability boxes. Uncertified cases abstain. These numerical checks do not validate the sampling or transport assumptions. Historical diagnostic results retain the scope of their executed implementation.”
