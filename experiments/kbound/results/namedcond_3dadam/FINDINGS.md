# 3D-ADAM named-condition hunt: NEAR-MISS (not a defensible beats-both)

*On-disk hunt for the paper's named condition — a real multimodal deployment where the deployed
fusion flips helpful<->harmful with the harm visible label-free. α=0.10.*

## How 3D-ADAM was found
Screened all 5 on-disk multimodal caches for: (a) a per-category benefit sign mix, and (b) label-free
predictability of the sign (val benefit -> test sign, "harm-AUC"). 3D-ADAM was the clear standout:
**23 categories, 7 helpful / 8 harmful fusion, sign detectable at harm-AUC 0.905.** (The other caches:
realiad_d3 0.69, mvtec3d 0.63, realiad_natdeg 0.40, mulsen 0.38.)

## Held-out test (leave-one-category-out conformal certificate)
| policy | mean test AUROC |
|---|---|
| always-adapt (fused) | 0.7561 |
| always-freeze (best single) | 0.7942 |
| **KGA-routed** | **0.8026** |
| oracle | 0.8306 |
Decisions: FREEZE 5, ABSTAIN 17, ADAPT 1. false-adapt 0/1 = 0.0 (<= α). Point estimate **beats both.**

## Significance (paired bootstrap, 23 categories, 10k resamples) -- the decisive check
- KGA - always-freeze = **+0.0084, 95% CI [0.00, 0.025], P(KGA>freeze)=0.64**  -> **TIE, not significant.**
- KGA - always-adapt  = +0.0465, 95% CI [-0.014, 0.117], P(KGA>adapt)=0.92    -> beats reckless-adapt, not at 0.95.
- **beats_both_significant = False.**

## Honest verdict
Not a defensible beats-both. The point-estimate win over always-freeze is a +0.008 noise margin; under
resampling KGA **ties the safe freeze baseline** and beats only the reckless always-adapt (P=0.92). Same
shape as every other natural/multimodal shift this project has tested: KGA avoids harm, does not gain.

## Caveats compounding the negative
- **Selection from 5 caches** (multiple comparisons): even the point estimate is inflated; a clean claim
  would need replication on independent data.
- KGA wins (in point estimate) by *freezing/abstaining* on 22/23 categories, not by adapting -- it's a
  safety result, not an accuracy-gain result.
- 3D-ADAM is multimodal **anomaly** detection (category-held-out), not a natural covariate shift.

## Implication for closing the 70->~80 gap
The on-disk hunt is exhausted: the most promising cache ties the safe baseline. Closing the gap requires
a genuinely new benchmark with (i) more conditions (so a real margin can reach significance), and (ii) a
regime where fusion *gains*, not merely avoids harm, with the gain label-free detectable. That is an
external data-acquisition effort, not an on-disk re-analysis.
