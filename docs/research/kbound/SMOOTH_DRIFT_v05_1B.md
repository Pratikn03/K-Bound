# v0.5 Extension — Part 1B: Smooth source→target drift bracket

**Refines Proposition `thm:reg-iff` (bounded-drift boundary) by making the drift radius observable.**
Proof fragment: `paper/sections/smooth_drift.tex`. Validator: `experiments/kbound/theory_validation/val_smooth_drift.py`. Results: `results_smooth_drift.json`. Figures: `fig_smooth_drift_*.png`.

## Status (proven vs. planned)

| Item | Status |
|---|---|
| Prop `thm:smooth-drift` (a) bracket, converse, online form | **Done; LaTeX compiles** |
| Validator, all 7 PASS flags green | **Done; run this session + reproduced from repo (seed 20260609, CPU only)** |
| Real shift-sequence track | **Planned (synthetic only here)** |
| Part 1C (unify 1A τ-route + 1B drift-route into the reach table) | **Planned, not started** |

This is an **incremental refinement**, not a new hard theorem — flagged honestly.

## The gap closed

`thm:reg-iff` gives the exact boundary `|U−2T_S| > 2BW` but leaves the drift radius **B assumed**. 1B replaces "known B" with a **drift-smoothness coupling** (Def `def:drift-smooth`): `‖g_T − g_S‖_{∞,D} ≤ L·d(P_S, P_T)`, where `d` is an **observable** covariate discrepancy (Gaussian W₂). Then `B = L·d` is measured, not assumed, and the whole boundary is computable given one modulus L:

- center `c = U − 2T_S` (observable), reach `ρ = 2LdW` (observable), commit `sign(c)` iff `|c| > ρ + ε_n`.
- **Online form:** along a stream `P_0→…→P_K` the reach accumulates as `2LW·Σ d(P_{k−1},P_k)`; abstain once the spent budget exceeds the margin.

In the reach table (`thm:unify`), the "bounded drift" row's reach `2BW` becomes the observable `2LdW`.

## Validator results (real numbers, seed 20260609)

![boundary / coverage / safety](fig_smooth_drift_boundary.png)

- **Noise-invariance** (Prop `thm:reg-noise`): Δ flat to **3.1×10⁻³** as unknown noise σ: 0→4, while absolute risk grows 0.08→16.
- **Bracket coverage ≥ 99.8%** of within-budget drifts (including the adversarial `b* = −Ld·sign(f₀−f_a)`) at every covariate shift.
- **Zero false commits** within the smoothness budget; commit rate falls 1.0 → 0 as reach `2LdW` overtakes the margin (abstention engages exactly at the computable boundary `d* = |U−2T_S|/(2LW)`).
- **Guard is necessary:** a no-correction rule (commit `sign(U−2T_S)` regardless) hits **100% false-commit** once drift appears; the guarded rule abstains and stays safe (0 false).
- **Converse is real:** over-budget drifts (violating Def `def:drift-smooth`) flip the committed sign — the assumption does the work.

![noise-invariance + baseline](fig_smooth_drift_noise_and_baseline.png)

**Honest caveats.** The Hölder inequality and the `U, T_S, W` objects are `thm:reg-iff`'s; 1B's delta is (i) observability of B via smoothness, (ii) the online accumulation form, (iii) the validated necessity of the guard. The committable region is **narrow** in this synthetic because the benefit margin `|U−2T_S| ≈ 0.16` is small relative to the worst-case reach — with a larger margin the commit region widens. `ε_n` is a data-driven 3σ confidence radius for the importance-weighted source estimate, so it grows (and the rule abstains) when the target leaves source support. Validation is synthetic. Does **not** resolve the general Conjecture 1.

## Next — Part 1C (await confirmation)

Unify the 1A multi-candidate τ-route and the 1B drift-route as two evaluations of the same ambiguity reach ρ in `thm:unify`: the multi-view row's reach 0 → `O(τ_normal)` (1A), the bounded-drift row's `2BW` → observable `2LdW` (1B); one certificate `|c| > ρ` covers both, with the residual/concept-drift terms as the two computable reaches.
