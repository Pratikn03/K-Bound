# Camelyon17 certificate radius: bias-vs-variance diagnostic (Protocol B, Path 1)

## VERDICT: BIAS-LIMITED
- **eps_256_observed = 0.1124** (0.9-quantile of |B_hat(Z)-B|, leave-one-seed cross-fit)
- **eps_meas(256) floor = 0.0441** (binomial measurement noise) -> **obs/floor = 2.55x**
- **eps_meas(1024) variance-limited prediction = 0.0221**; bias-limited prediction = 0.1124 (unchanged)
- **frac_meas = 0.16** (signed-residual var) / 0.39 (abs-residual var) -> residual is dominated by MODEL ERROR, not eval-sample noise.

## What it means
The conformal radius on Camelyon17 is dominated by B_hat(Z) model error
(calibration-drift bias), NOT measurement variance. Bias does not shrink with
eval n, so Protocol B's eps ~ 1/sqrt(n_eval) prediction CANNOT hold: a fair
re-run at n_eval=1024 would only touch the ~16-39% variance slice, leaving the
radius near 0.11. **Path 2 (fair re-run) will not close the certificate** -> skip it.
The open problem reframes to bias/calibration-drift-limited certifiability on
natural shift (ties to the gamma-frontier); reducing estimator bias is the lever.

## Method (reused exactly)
- B_hat = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
  subsample=0.8, random_state=0) — EXACT config from eps_recal_camelyon.py. alpha=0.10 FIXED.
- Residuals r_i=|B_hat(Z_i)-B_i| via **leave-one-seed cross-fit** (held out by seed,
  matching the eps-recal hold-out): genuine out-of-sample model error.
- sigma_meas^2(n)=a0(1-a0)/n+aa(1-aa)/n, n=256; eps_meas=1.6449*sigma (half-normal 0.9q).
  In this pool a0*256 and aa*256 are integers => balanced acc = raw acc => **n_eff=256**.

## n=1024 cross-check (WEAK, n=10 = 5 seeds x 2 methods)
B-std(1024)/B-std(256) = **0.71**, not the 0.50 a 1/sqrt(4) variance-limited halving
needs. Underpowered, but consistent with the bias-limited reading (spread not halving).

Outputs: diag.py, diag_results.json, residual_vs_measurement_floor.png.
