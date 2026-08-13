# K-Bound Short Paper Theory Audit

Date: 2026-08-11

## Verdict

The current manuscript keeps the mathematical and empirical layers distinct. No empirical panel in
this reconciliation is used to alter or prove a population theorem.

## Checked conventions

- Benefit is `Delta = R_T(f_0) - R_T(f_a)`: positive helps, negative harms, and zero is a tie.
- A strict adapt/freeze commitment is uniformly supportable over the declared drift class only
  outside the closed band `|M| <= beta`; the conservative action inside the band is abstention.
- `beta = 0` is a zero-drift assumption, not a conservative default for unknown drift.
- `M`, `gamma`, and `beta` are population quantities.
- `Delta_hat` and `epsilon` are finite-sample KGA quantities.
- `epsilon` does not estimate `beta`, and real-data KGA does not numerically apply `|M| > beta`.
- The elementary false-adapt implication requires interval coverage. Exchangeability or a valid
  correction may justify coverage; risk alignment is not an extra premise after coverage is assumed.
- `FA_u = P(adapt and Delta <= 0)` is the theorem-controlled marginal event under coverage.
- `FA_c = P(Delta <= 0 | adapt)` is descriptive unless separately proved.

## Lean scope

The manuscript describes the Lean development as kernel-checked for the named algebraic,
finite-decision, measure-containment, and conditional uniform-rank results. It does not claim that a
successful build mechanizes the entire deployment theorem. External assumptions and unmechanized
links include target-law construction, evidence-law equality, deployment-class membership, risk
alignment, calibration transfer, and the lift to arbitrary exchangeable deployment processes.

## Empirical non-implications

- Empirical abstention is not automatically structural non-identifiability.
- A fitted benefit regressor does not certify risk alignment.
- Source-replayed natural no-harm results do not prove a universal no-harm theorem.
- ImageNet-C SAR point beats-both does not establish CI-robust or seed-robust superiority.
- ImageNet-R and PACS are retained as negative diagnostics rather than forced into the theory claim.

## Residual theory risk

The main submission risk is interpretive, not an algebraic contradiction: reviewers may still read
the long population-budget development as stronger than the deployment evidence. The paper now
states the operational frontier reading as withdrawn where the declared budget procedure fails.

