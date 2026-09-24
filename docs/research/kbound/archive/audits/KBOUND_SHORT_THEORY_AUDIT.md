# K-Bound Mathematical Audit

Date: 2026-08-24

## Verdict

The maintained papers now present one coherent logical chain without equating population
identifiability with empirical calibration. The core algebra and finite-event implications are
internally consistent. The paper remains conditional on declared deployment classes and coverage;
it is not an assumption-free safety theorem.

## Checked Statements

- Benefit: `Delta = R_T(f_0) - R_T(f_a)`; positive helps, negative harms, zero is risk-equivalent.
- Pointwise correctness: `eta_a(x) = P_T(f_a(X)=Y | X=x)`.
- Reduction:
  `M + gamma = (E[s|D]-1/2) + E[eta_a-s|D] = E[eta_a|D]-1/2` and
  `Delta = 2 mu_T(D)(M+gamma)` under the binary disagreement setup.
- Interior: for `beta > 0` and `|M| < beta`, admissible evidence-identical worlds have opposite
  nonzero benefit signs.
- Boundary: at `|M| = beta > 0`, one admissible world has zero benefit and another has a strict
  benefit with the sign of `M`; opposite nonzero signs are not claimed there.
- Closed band: under strict directional semantics, abstention is the maximal sound three-way action
  on `|M| <= beta`, including the degenerate zero-drift tie.
- Frontier: a strict direction is uniformly supportable over the rich declared class if and only if
  `|M| > beta`.
- Certificate: assumed interval coverage implies marginal false-adapt and false-freeze control.
  `FA_c` is not theorem-controlled.
- Multiclass: `Delta = P_T(D)(p_a-p_0)`; no binary complement identity is reused.

## Population And Empirical Layers

| Quantity | Layer | Meaning |
|---|---|---|
| `M` | population | observable disagreement-region margin |
| `gamma` | population | latent calibration drift |
| `beta` | population | externally declared bound on `|gamma|` |
| `Delta` | shared target | frozen-minus-candidate risk |
| `Delta_hat` | empirical KGA | learned estimate from label-free evidence `Z` |
| `epsilon` | empirical KGA | residual-coverage radius |

`epsilon` does not estimate `beta`; real-data KGA does not compute `M`, `gamma`, or `beta`.
Empirical abstention may result from structural ambiguity, finite data, a weak benefit estimator,
calibration-transfer failure, or a conservative radius. It is not automatically a theorem-specific
non-identifiability diagnosis.

## Lean/Mathlib Verification

`formal/build.sh` completed 2,554 jobs from the pinned Lake manifest. The audit reports:

- formal audit: PASS;
- forbidden proof-hole scan: PASS;
- 65 theorem-map checks;
- measure-level false-adapt and false-freeze containment;
- finite uniform-rank coverage and exchangeable-index implications;
- interior, boundary, closed-band, and distributional frontier declarations;
- finite matched-world construction, unit-mismatch, stability, two-point Le Cam, rate, swap, and
  non-finite witness declarations.

The public build is free of Lean linter warnings. The audit reports ordinary Mathlib axioms such as
classical choice, propositional extensionality, and quotient soundness where expected. A successful kernel build does
not discharge paper-level empirical obligations: class membership, evidence-map validity,
calibration transfer to a new environment, or the claim that a benchmark protocol represents a
future deployment.

## Remaining Mathematical Risk

The risk is scope, not a known contradiction. Necessity needs a class rich enough to contain the
matched label kernels and boundary law. The finite-sample theorem is only as useful as its coverage
premise at the declared scoring unit. The paper now states both qualifications adjacent to the
results they govern.
