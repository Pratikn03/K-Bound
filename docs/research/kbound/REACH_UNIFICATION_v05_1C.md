# v0.5 Extension — Part 1C: Reach-table unification + sharpened Conjecture 1

**Folds the 1A τ-route and 1B drift-route into the ambiguity reach (`thm:unify` / `def:reach`) as two evaluations of the same ρ(P).**
Fragment: `paper/sections/reach_unification_v05.tex`. Validator: `experiments/kbound/theory_validation/val_reach_unification.py`. Results: `results_reach_unification.json`.

## Status

| Item | Status |
|---|---|
| Prop `prop:reach-table-v05` (two new rows) + proof | **Done; LaTeX compiles** |
| Sharpened Conjecture 1 statement | **Done** |
| Consistency validator, all 5 PASS flags | **Done; run + reproduced from repo (seed 20260609, CPU)** |

Integrative/organizational result (Prop `thm:unify` direction (i) is near-definitional once reaches are computed) — flagged honestly.

## What it says

Both v0.5 results are evaluations of the single reach `ρ(P) = ½(sup−inf I(P))`; the certificate `commit sign(c) iff |c| > ρ` (Prop `thm:unify`) governs both. Two rows added to the reach table:

| family | center c | reach ρ | source |
|---|---|---|---|
| multi-view, **approx.** indep. | recovered Δ̂ | **O(τ)** on normal cone + tangential blind spot | 1A (`thm:multicand-residual`) |
| smooth drift, radius Ld | U − 2T_S | **2 L d W** (observable) | 1B (`thm:smooth-drift`) |

In both, the previously *assumed* structural quantity (exact independence; known drift radius B) becomes a *measured* one (τ; d), at the cost of one residual modulus (tangential-control budget; Lipschitz L).

## Validator results (seed 20260609)

- **Drift row:** brute-forced benefit interval reproduces closed-form reach `2LdW` with **ratio 1.000** (60 random worlds); unified `|c|>ρ` certificate → **zero false certifications**; below the boundary both signs are realized (converse, 100%).
- **Multi-view row:** exact independence → reach ≈ **0.02** (sampling floor, reproduces the Steinhardt–Liang "reach 0" row); generic approximate independence → recovery half-spread tracks observable τ (**Spearman 1.00**); a **realizable** tangential error-correlation gives **τ = 2×10⁻⁹ ≈ 0 yet η = 0.53 and a wrong recovered sign** — confirming τ bounds only the normal reach, the tangent being the irreducible `thm:imp` blind spot.

## Conjecture 1, sharpened

Previously open: unconditional label-free bracketing of sign(Δ) for K≥3 / regression. v0.5 closes **two structured slices** with checkable prices (1A: M≥4 + small τ + controlled tangential; 1B: Lipschitz-coupled drift). What remains open, stated precisely:

> Compute or two-sidedly bound ρ(P) for a **nonparametric** family (unknown conditionals, unbounded drift) using **only** observable side-information and **no** residual structural modulus — equivalently, a label-free statistic upper-bounding ρ(P) **including its tangential component**, with a tight matching converse.

The 1A blind spot shows this is genuinely hard, not a gap in analysis: an agreement-only statistic is provably blind to a tangential family that moves Δ across zero (`thm:imp`), so some extra observable or residual modulus is necessary. v0.5's contribution: convert "unknowable in general" into "knowable up to one named, checkable modulus per family," and name the modulus each family still pays.

## Honest scope

Organizational result; `thm:unify` direction (i) is near-definitional once reaches are computed. The multi-view reach is observable only on the normal cone — we do not claim to bound the tangential component, and the sharpened conjecture says so. All numbers are synthetic consistency checks of the reach identities, not evidence about real workloads.
