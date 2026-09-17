# K-Bound 100% Theory Closure Plan (historical / superseded)

> **Status as of 2026-09-11: superseded historical plan.** This file records an
> earlier proposed closure state and is not evidence that the current
> foundations gate passes. The authoritative current receipt is
> `formal/formal_foundations_report.json`; its Lean build and kernel audit pass,
> while the full-foundations gate remains open for the historical one-bit/H/
> ratio-rate extension.

This file was the hard gate for an earlier “100% complete” proposal. Its Wave 4
and Wave 6 closure statements are historical and must not be used as current
verification. The maintained scoped core is checked by
`formal_audit.py --build --strict-core`; the stronger
`formal_audit.py --build --full-foundations` gate currently fails closed.

## Current Verdict

**Submission theorem stack: scoped core complete; full foundations open.**

**Lean paper-faithful foundations gate: not complete** (`--full-foundations` currently FAILS).

**Full research program: not closed by this historical plan.** The remaining
formal and empirical gates are tracked in the current status ledger.

## A. Lean/Mathlib Mechanization — historical Wave 4/Wave 6 snapshot

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

## B. Research Frontier — historical snapshot (not current closure)

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

The current full-foundations audit is expected to remain non-zero until the
historical one-bit/H/ratio-rate extension is genuinely formalized. Do not waive
that result or relabel it as complete.

## E. Documentation

Canonical doc map: [`DOCS_INDEX.md`](DOCS_INDEX.md). Do not add new dated status MDs; update
`PROJECT_STATUS_AND_OPEN_PROBLEMS.md` and `claim_ledger.json` instead.
