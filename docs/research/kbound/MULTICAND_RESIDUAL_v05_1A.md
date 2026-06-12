# v0.5 Extension — Part 1A: Bounded-residual multi-candidate sign identifiability

**Closes a quantitative gap around Definition 5 (`def:cei`) / Proposition `thm:multicand` / Proposition `prop:cei-test`.**
Reviewer-ready theorem + proof: `paper/sections/multicandidate_residual.tex`.
Validator: `experiments/kbound/theory_validation/val_multicandidate_residual.py`.
Raw results: `results_multicandidate_residual.json`. Figures: `fig_multicand_*.png`.

---

## Status note (proven vs. planned)

| Item | Status |
|---|---|
| Theorem `thm:multicand-residual` (a)–(d) stated | **Done** |
| Proofs of (a) observability/`M=3` degeneracy, (b) `O(ε)` recovery, (c) ordering bracket, (d) converse + tangential blind spot | **Done, written, LaTeX compiles** |
| Numerical validator, all 9 PASS flags green | **Done, run this session (seed 20260609, CPU only)** |
| Real-adaptor check (Tent/EATA/SAR actually land in the τ-small regime) | **Planned** — empirical track, not claimed here |
| Part 1B (smooth source→target drift bracket) | **Planned, not started** |
| Part 1C (unify both routes into the reach table, `thm:unify`) | **Planned, not started** |

This is a **solid relaxation of one assumption into a checkable, bracketed one** — not an assumption-free resolution of Conjecture 1 (`conj:gen`), and not "field-shaping." The honest converse (d) is the point: it says exactly what the diagnostic can and cannot certify.

---

## The gap being closed

The paper's multi-candidate result (`thm:multicand`) recovers candidate advantages **exactly**, but only under **exact** conditional error-independence (Definition 5). The diagnostic `prop:cei-test` only uses the residual as a **binary** flag (τ = 0 vs. τ > 0). Co-trained adaptors violate independence *slightly*, so the real question is quantitative:

> If the unlabeled agreements are *near* rank-one (residual τ small but nonzero), how wrong can the recovered accuracy ordering — hence each `sign(Δ)` — be, and can that error be bounded by something **observable**?

## What is proven

Setup: M ≥ 3 candidates, binary Y on the disagreement region D, advantages `b_j = 2a_j − 1` with margin `|b_j| ≥ β`, a designated anchor candidate above chance. Observable centered agreements `c_ij = 2A_ij − 1 = b_i b_j + E_ij`; misspecification `η = ‖off(E)‖_F`, observable rank-one-fit residual `τ = min_β ‖off(C − ββᵀ)‖_F ≤ η`.

- **(a) Observable & sound.** τ is a statistic of unlabeled agreements; τ = 0 iff `off(C)` is rank-one. **For M = 3 the residual is identically zero** (3 entries, 3 unknowns) — no diagnostic power; informative only for **M ≥ 4** (sharpens `prop:cei-test`).
- **(b) Recovery — O(ε·poly M).** Median-of-minors estimator obeys `|b̃_i − b_i| ≤ 5 η_∞/β³`, so `‖b̃ − b*‖₂ ≤ 5√M · η/β³`. (Matrix-perturbation / 2×2-minor argument; explicit constants.)
- **(c) Ordering / sign bracket.** Accuracy half-width `w = (5/2)η_∞/β³`; ordering recovered on all gaps > 2w, `sign(Δ⁽ⁱ⁾) = sign(b̃_i)` on all `|b_i| > 2w`. Certifying all M candidates from n points on D adds a union term `O(β⁻³ √(log M / n))` — this is where the **log M** enters.
- **(d) Converse + abstention.** τ > τ* certifies Definition 5 fails (sound one-sided, exact for M ≥ 4). **Worst-case necessity:** *tangential* error-correlations `E = off(b·uᵀ + u·bᵀ + uuᵀ)` keep τ = 0 while shifting the advantages by u — invisible to **any** agreement-only statistic, re-instantiating the two-world impossibility (`thm:imp`). So small τ is **necessary but not sufficient** in the worst case; τ certifies only the *normal* (detectable) part of the violation.

## What the validator actually shows (real numbers, seed 20260609)

![recovery and residual vs rho](fig_multicand_recovery_vs_rho.png)

- **Exact under independence (ρ = 0):** sign accuracy 1.00, `‖b̃ − b*‖_∞ ≈ 0.03` for M = 3, 4, 5.
- **O(ε) scaling holds:** recovery error vs. true misspecification has **log-log slope 0.86 (R² = 0.96)** — linear in ε, as predicted (slightly sub-linear from median aggregation).
- **τ is a genuine quantitative dial:** strictly **monotone in ρ for M ≥ 4 (Spearman 1.00)**; for M = 3 it stays ≤ 0.007 and is **17× smaller** than M = 4 at ρ = 0.2 — i.e. M = 3 is diagnostically blind, exactly as (a) predicts.
- **Empirical threshold:** committed-sign accuracy stays ≥ 99% up to **τ ≈ 0.069** (M = 4, n = 4000, β = 0.25), then degrades smoothly (0.97 at τ ≈ 0.10, 0.84 at τ ≈ 0.27).
- **Safety:** the τ-gated commit rule makes **zero** false commitments for ρ ≤ 0.2 and ≤ 0.14% across all ρ (the small leakage is precisely the tangential component of (d)); commit rate falls 1.00 → 0.19 as ρ: 0 → 0.5 (abstention engages).

![eps and M scaling](fig_multicand_eps_and_M_scaling.png)
![commit / abstain safety](fig_multicand_commit_abstain.png)

**Honest caveat on `O(ε·log M)`:** the **O(ε)** part is tight (slope 0.86). The **log M** part is a *loose upper bound* — empirically the simultaneous bracket width is **flat in M** (0.069 → 0.049 for M: 4 → 12), because the median over Θ(M²) triples improves per-candidate estimates as M grows, offsetting the union-bound cost. The log M survives only as the formal worst-case sampling term.

## Honest scope

Exact-independence recovery is **classical** (Dawid–Skene; the spectral rank-one construction of Parisi et al. and Jaffe–Nadler; Anandkumar et al. for the multiclass tensor case). The contribution here is entirely in the *approximate* regime: (i) the perturbation bound, (ii) the **observable** residual τ that controls the detectable violation (the verifiability ATC / Agreement-on-the-Line / AETTA rely on but never establish), and (iii) the precise converse pinning down what τ cannot see. Validation is synthetic — it tests the theory under a controlled generic error-correlation model; it is **not** evidence that real adaptors live in the τ-small regime.

## Next (await confirmation before starting)

- **1B — smooth source→target drift bracket.** Replace the discrete two-world drift with a bounded-Lipschitz drift `‖∂_t P_t‖ ≤ L`; bracket `sign(Δ)` by an integral of the observable drift, abstaining when the integrated drift exceeds the margin. Plugs into `sec:regslice`.
- **1C — unify both routes into the reach table (`thm:unify`).** Show the multi-candidate τ-route and the 1B drift-route are two evaluations of the same *ambiguity reach* ρ: the "multi-view" row's reach 0 becomes `ρ = O(τ_normal)` with an irreducible tangential blind spot; the drift row's reach becomes the integrated-drift bound. One certificate `|c| > ρ` covers both.
