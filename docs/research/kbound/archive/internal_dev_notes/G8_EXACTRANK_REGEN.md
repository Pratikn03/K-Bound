# G8 exact-rank regeneration — RESULT (2026-07-20)
Question: does the EXACT split-conformal rank rule (eps=sorted(rho)[k-1], k=ceil((n+1)(1-alpha)),
kbound_pkg/kbound/certificate.py) keep FA_u<=alpha AND preserve the beats-both, replacing the
interpolated np.quantile that under-covers at finite n?  Script: scripts/g8_exactrank_regen.py
Recomputed from raw per-condition residuals rho_i=|b_hat_i - B_i|; decisions ADAPT iff b_hat-eps>0,
FREEZE iff b_hat+eps<0, else ABSTAIN; FA_u = marginal mean(ADAPT & B<=0). alpha=0.10.

## Result (full 5 seeds; pooled)
Track            | FA_u interp | FA_u EXACT | KGA/adapt/freeze (EXACT) | beats-both EXACT
ImageNet-C SAR   | 0.0074      | 0.0000     | 0.0264/0.0529/0.0319     | TRUE  (survives)
ImageNet-C EATA  | 0.0074      | 0.0000     | 0.0009/0.0001/0.0342     | no-harm (ties adapt)
ImageNet-C TENT  | 0.0000      | 0.0000     | 0.0145/0.0191/0.0145     | no-harm (abstain->freeze)
CIFAR-10-C TENT  | 0.0000      | 0.0000     | 0.0016/0.0080/0.1239     | TRUE  (survives)
CIFAR-10-C EATA  | 0.0000      | 0.0000     | 0.0013/0.0033/0.1313     | TRUE  (survives)

## Verdict — G8 = PASS (upgraded from FIX)
- FA_u <= alpha holds on EVERY track under exact-rank (all 0.0000). The "finite-sample certificate"
  language is HONEST under the exact rule. No relabel/withdrawal needed.
- Every beats-both SURVIVES (SAR, CIFAR Tent, CIFAR EATA). Margins shrink (larger exact eps => more
  abstention => higher KGA regret) but conclusions hold.
- ACTION: update panel/prose numbers to the EXACT-rank values (regret changes: e.g. ImageNet-C SAR
  0.0107->0.0264), state that FA_u/eps use the exact split-conformal rank rule (certificate.py), and
  drop the interpolated-quantile from the headline path. This resolves G1 (manifest sync) direction too.

## CANONICAL pooling verified (2026-07-20) + paper updated
Panel pools per-seed eps over 135 condition-seed pairs (FA_u=1/135=0.0074 confirms). Method-B
regen with INTERPOLATED eps reproduces the panel exactly (SAR .0107, EATA .0003, Tent .0139) =>
pooling matched. EXACT-rank canonical numbers (scripts/g8_canonical_pooling.py, g8_exactrank_ci.py):
  ImageNet-C SAR : KGA 0.0264 / adapt 0.0529 / freeze 0.0319 ; FA_u=0.000 ; BEATS-BOTH by CI
                   (gap-to-adapt [-0.052,-0.003], gap-to-freeze [-0.0088,-0.0027]; both exclude 0)
  ImageNet-C EATA: KGA 0.0009 / 0.0001 / 0.0342 ; FA_u=0.000 ; no-harm (ties adapt)
  ImageNet-C TENT: KGA 0.0145 / 0.0191 / 0.0145 ; FA_u=0.000 ; no-harm (abstain->freeze)
  CIFAR-10-C Tent/EATA: unchanged to 3dp (n=432 => exact≈interp).
PAPER UPDATED: kbound_short.tex SAR row/prose/panel/table 0.0107->0.0264, FA_u 0.007->0.000, CIs
updated, "exact split-conformal radius" stated. Algorithm 1 eps = exact rank rule. Build clean 23pp.
HEADLINE INTACT: ImageNet-C SAR beats-both survives the exact-conformal fix by CI. FA_u=0 everywhere.
