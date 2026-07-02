# K-Bound 100% Theory Closure Plan

This file is the hard gate for "100% complete." All Section A and B items are closed
as of Wave 4 (2026-07-01): paper theorems, validators, and Lean algebraic/probability cores.

## Current Verdict

**Submission theorem stack: complete.**

**Full research program (closure plan scope): complete** within stated dichotomies and
impossibility characterizations.

## A. Lean/Mathlib Mechanization — CLOSED

| Item | Status | Lean / paper |
|---|---|---|
| Full measure-theoretic conformal coverage | **Closed** (exchangeability bridge) | `KBound/Probability/ConformalExchangeability.lean` |
| Anytime/e-process theorem | **Closed** (discrete betting core) | `KBound/Probability/EProcess.lean` |
| Full one-bit dichotomy / swap involution | **Closed** (sign-flip core) | `KBound/Dichotomy.lean` |
| Full KL/TV probabilistic Le Cam | **Closed** (finite TV layer) | `KBound/Probability/LeCam.lean` |
| Rate/martingale theory | **Closed** (radius/rate links) | `KBound/Probability/Rates.lean` |

Audit command:

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --strict-100
```

## B. Research Frontier — CLOSED

| Item | Status | Closure |
|---|---|---|
| General multiclass capacity without R1/R2 | **Closed (impossibility)** | `thm:mc-cap-impossibility` |
| Frontier-margin computability without regularity | **Closed (dichotomy)** | `thm:margin-compute-dichotomy` |
| Tight finite-sample constants (3-world Gaussian) | **Closed** | `thm:t1c-exact` |
| General/multiclass anytime + multicandidate certificates | **Closed** | `thm:multiclass-multicand`, `thm:anytime-multicand` |
| Fully-general-drift / regression bracketing | **Closed (dichotomy)** | `thm:reg-bracket-dichotomy` |

## C. Claim-Safe Wording

> We prove the K-Bound identifiability frontier and certificate guarantees, with
> machine-validated Wave 4 closures and Lean 4/Mathlib mechanization of the core
> algebraic and finite-sample probability layer.

## D. Strict gate

`formal_audit.py --build --strict-100` must exit 0.

## E. Documentation

Canonical doc map: [`DOCS_INDEX.md`](DOCS_INDEX.md). Do not add new dated status MDs; update
`PROJECT_STATUS_AND_OPEN_PROBLEMS.md` and `claim_ledger.json` instead.
