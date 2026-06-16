# LOCKED Protocol-B — Camelyon17 full-scale (n_eval=1024, 5 seeds)

Pre-registered: `research_lock/CAMELYON17_FULLSCALE_PROTOCOL_B_v1.yaml`. alpha=0.10 FIXED.
eps-rule = (1-alpha)-quantile of CAL residuals |Bhat-B|, CAL/TEST split BY SEED, frozen, decided on TEST.

## Schema (what was actually serialized)
- **Debug n=256**: 432 per-cell records = 72 conditions x 6 TTA candidates x 4 seeds; Z dim 11.
- **New n=1024**: only **5 per-seed aggregate records per method** (tent, eata); Z dim 10; fields {Z,a0,aa,B,seed,n_eval}.
- The pre-registered **72x6 composition grid was NOT written** to the new output dir. The runner
  collapsed to one (Z,B) point per seed. The eps-recal on the new run is therefore at **per-seed
  granularity only** (5 points, C(5,2)=10 CAL splits, 1 test-seed pair each), NOT the per-cell
  granularity of the debug analysis. This is the dominant caveat.

## n=256 vs n=1024 (false-adapt / eps / radius-ratio)
| run | granularity | eps | false-adapt | commit | radius ratio (vs 0.5 pred) |
|-----|-------------|-----|-------------|--------|----------------------------|
| n=256 (archived, paper) | 432-cell grid | 0.030 | **0.185** [0.159,0.211] | 0.65 cov | — |
| n=256 (in-tree reproduction) | 432-cell grid | 0.0296 | 0.139 [0.108,0.170] | 0.58 | — |
| n=1024 tent | 5 per-seed | 0.0292 | **0.333** | 0.60 | 0.99 |
| n=1024 eata | 5 per-seed | 0.0381 | **0.333** | 0.60 | 1.29 |

(eps & coverage of the in-tree reproduction match the paper's 0.030/0.65 exactly; false-adapt 0.139
vs paper's 0.185 differs because the archived run used `theory_v2/realdata/eps_recal/`, absent from
this tree — both are >alpha, same regime.)

## Commit / regret (n=1024, mean +/- 95% CI over 10 splits)
- tent: commit 0.60 [0.28,0.92]; KGA regret 0.020 [0.010,0.030] vs adapt 0.0063, freeze 0.027 — beats freeze only, NOT always-adapt.
- eata: commit 0.60; KGA regret 0.038 vs adapt 0.019, freeze 0.039 — ties freeze, trails adapt.

## VERDICT: FAILS (per YAML success_criteria_stated_in_advance)
False-adapt did **not** drop to <= alpha at n=1024 (0.33 > 0.10), and the conformal radius did
**not** halve (eps essentially unchanged, ratio ~1.0, not 0.5). The 1.9x-alpha gap persists and is
the deeper open problem: detectability-certifiability is not closed by sample size alone.

## What was / wasn't testable (honest scope)
- **Testable**: per-seed eps-recal (eps, commit, regret vs trivials) at n=1024.
- **NOT testable**: the "1/sqrt(n) shrinkage on the 72x6 composition grid" — the new run did not
  serialize the grid, so the radius-vs-n prediction cannot be evaluated at the grid granularity it
  was stated for. The per-seed radius is statistically too coarse (5 points, 1 harmful seed of 5) to
  separate from the debug eps. The prediction is **largely untested**, and on the available
  granularity it is **not confirmed**.
