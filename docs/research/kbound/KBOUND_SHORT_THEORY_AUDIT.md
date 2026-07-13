# K-Bound Short Paper Mathematical Audit

## Core convention

The manuscript now uses `Delta = R_T(f_0)-R_T(f_a)` consistently: positive is helpful, negative is harmful, and zero makes the fixed actions risk-equivalent.

## Core results

| Result | Final statement | Status |
|---|---|---|
| Disagreement reduction | `Delta=2 mu_T(D)(a_bar-1/2)` and `sign Delta=sign(M+gamma)` | proved in main text |
| Interior impossibility | for `beta>0` and `|M|<beta`, matched augmented evidence can support opposite nonzero benefits | proved in main text |
| Boundary result | at `|M|=beta>0`, zero-versus-strict ambiguity blocks a uniformly sound strict action | proved separately |
| Closed-band action | abstention is maximal under strict three-way soundness on `|M|<=beta` | proved in main text |
| Strict-commitment frontier | a strict adapt/freeze action is uniformly supportable iff `|M|>beta` | proved in main text |
| Finite-sample certificate | marginal interval coverage implies marginal false-adapt and false-freeze at most `alpha` | proved in main text |
| Multiclass bridge | `Delta=P_T(D)(p_a-p_0)` | proved; converse needs target-class richness |
| Regression bridge | squared-loss benefit decomposes as `M_reg+gamma_reg` | derived; converse needs richness |

## Assumption separation

- Risk alignment is structural and supports population identifiability/transfer interpretation.
- Coverage alone is the premise of the elementary finite-sample error implication.
- Exchangeability or an explicit shift correction is one route to coverage.
- Fitting a benefit regressor does not certify risk alignment.
- `beta=0` is the strongest zero-drift assumption, not a conservative unknown-drift setting.

## Population versus empirical layer

- Population: `M`, `gamma`, `beta`.
- Empirical KGA: `Delta_hat`, `epsilon`.
- Real-data KGA does not numerically estimate or apply `M`, `gamma`, or `beta`.
- `epsilon` does not estimate `beta`.
- Empirical abstention does not prove structural non-identifiability.

## Lean scope

The repository's Lean inventory checks indexed algebraic, finite-decision, measure-containment, and conditional uniform-rank results. It includes frontier sufficiency and marginal error containment conditional on coverage. It does not establish the full target-law witness construction, frontier necessity/maximality, risk alignment, multiclass identifiability, calibration transfer, or the general exchangeable-process lift. The final paper states this narrower scope.

## Residual mathematical risk

- A specialist should independently review the target-law richness used by the necessity construction.
- The full measure-theoretic exchangeability lift remains outside the current Lean development.
- No theorem controls conditional false-adapt `FA_c`; it is descriptive only.
