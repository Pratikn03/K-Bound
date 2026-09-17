# K-Bound 100% Theory Closure Plan

This file is the hard gate for "100% complete." All Section A and B items are closed
as of Wave 4 (2026-07-01): paper theorems, validators, and Lean algebraic/probability cores.
Wave 6 (2026-07-15) additionally closed the paper-faithful Lean foundation gaps so that
`formal_audit.py --full-foundations` passes.

## Current Verdict

**Submission theorem stack: complete.**

**Lean paper-faithful foundations gate: complete** (`--full-foundations` PASS).

**Full research program (closure plan scope): complete** within stated dichotomies and
impossibility characterizations.

## A. Lean/Mathlib Mechanization — CLOSED (Wave 4 + Wave 6)

| Item | Status | Lean / paper |
|---|---|---|
| Full measure-theoretic conformal coverage | **Closed** (uniform-index + exchangeable-score reduction) | `UniformConformal.lean`, `Exchangeable.lean` |
| Anytime/e-process theorem | **Closed** (null supermartingale step + Ville/Markov) | `EProcess.lean`, `Ville.lean` |
| Full one-bit dichotomy / swap involution | **Closed** (evidence-preserving involution) | `Dichotomy.lean` |
| Full KL/TV probabilistic Le Cam | **Closed** (two-point law packaging) | `LeCam.lean`, `LeCamMeasure.lean` |
| Rate/martingale theory | **Closed** (Hoeffding radius + commit bridge) | `Rates.lean` |

Audit command:

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --full-foundations
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

`formal_audit.py --build --full-foundations` must exit 0.

## E. Documentation

Canonical doc map: [`DOCS_INDEX.md`](DOCS_INDEX.md). Do not add new dated status MDs; update
`PROJECT_STATUS_AND_OPEN_PROBLEMS.md` and `claim_ledger.json` instead.
