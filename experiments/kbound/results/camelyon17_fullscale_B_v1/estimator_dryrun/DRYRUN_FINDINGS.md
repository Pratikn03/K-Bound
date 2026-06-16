# Camelyon17 estimator dry-run — can a better B_hat(Z) make the certificate fire?

EXPLORATORY (no paper edit). alpha=0.10 FIXED, tau* not tuned. Split BY SEED:
DEV={0,1} (explore, honest within-DEV holdout 0<->1), TEST={2,3} (evaluate ONCE).
Decide logic = exact eps-recal from _locked_B_analysis.py (adapt if Bhat-eps>0,
freeze if Bhat+eps<0, else abstain; eps=(1-alpha)-quantile of |Bhat-B| on CAL).
Variants: GBR baseline, ridge(std), ridge(poly2), RF, MLP, isotonic-GBR,
quantile-reg direct radius (qr_hetero), GBR+debias, Mondrian eps by comp / by |Bhat| bin.

## DEV table (Camelyon, FA = false-adapt; need FA<=0.10 AND commit>=0.30)
| variant | DEV FA | DEV commit | DEV regret_kga |
|---|---|---|---|
| gbr_baseline | 0.067 | 0.620 | 0.0066 |
| ridge_std | 0.000 | 0.296 (<0.3, EXCLUDED) | 0.0254 |
| ridge_poly | 0.078 | 0.472 | 0.0185 |
| rf | 0.055 | 0.463 | 0.0164 |
| mlp | 0.116 | 0.236 | 0.0334 |
| isotonic_gbr | 0.067 | 0.620 | 0.0059 |
| **qr_hetero (LOCKED)** | **0.032** | **0.394** | 0.0144 |
| gbr+debias | 0.095 | 0.681 | 0.0051 |
| gbr+mondrian[comp] | 0.073 | 0.634 | 0.0061 |
| gbr+mondrian[bhatbin] | 0.088 | 0.690 | 0.0040 |
| ridge_poly+mondrian[bhatbin] | 0.077 | 0.454 | 0.020 |

LOCKED CHOICE (best DEV FA among commit>=0.3): **qr_hetero** (heteroskedastic
quantile-regression radius), DEV FA=0.032, commit=0.394.

## TEST seeds {2,3} — the one locked variant vs baseline (evaluated ONCE)
| | FA (need<=0.10) | commit | regret_kga | regret_adapt | regret_freeze |
|---|---|---|---|---|---|
| qr_hetero (LOCKED) | **0.123 FAIL** | 0.565 | 0.0127 | 0.0137 | 0.0586 |
| gbr_baseline | 0.189 FAIL | 0.685 | 0.0100 | 0.0137 | 0.0586 |

Locked variant cuts FA from baseline 0.189 -> 0.123 (and beats both trivials on regret),
but does NOT clear alpha=0.10. **DEV win (0.032) did not transfer (0.123) = OVERFIT.**

## CIFAR-10-C sanity (DEV/TEST seed split, must stay FA<=0.10)
qr_hetero: FA=0.008, commit=0.645 — VALID. baseline: FA=0.000 — VALID. No regression.

## Characterization (NOT used to pick lock): does ANY variant pass TEST?
ridge_std (TEST FA=0.053, commit=0.347) passes; ridge_poly FA=0.089/commit=0.259(<0.3).
A passing estimator exists in hindsight, but it was excluded by the honest DEV gate
(ridge_std DEV commit=0.296, just under 0.30). So the win is real-but-fragile, not
robustly selectable; the honestly-locked choice overfit.

## VERDICT: OVERFIT (leaning NEEDS-NEW-EVIDENCE)
The honestly DEV-selected estimator overfits (TEST FA 0.123 > 0.10). A linear ridge
*would* pass TEST (0.053) but sits exactly on the commit>=0.3 boundary and was not
selected by the pre-committed DEV rule — it is not a robust, reproducible win on logged
Z. The bias is largely in Z, not the estimator.
NEXT STEP: do NOT pre-register an estimator win on this logged grid. Either (a) re-run
with more seeds to test if a margin-of-safety ridge variant survives a real DEV/TEST
gate, or preferably (b) Protocol D GPU re-run with richer evidence features to push
B_hat(Z) bias below the certificate threshold.
