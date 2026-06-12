# epsilon-recalibration of the Route-1 certificate on Camelyon17 — findings

**Verdict: PRECISE NEGATIVE** (publishable). Harm is detectable but not
certifiable at level alpha=0.10 at this debug-scale sample size.

## Setup
- Data: `experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json`
  (schema `kbound_wilds_camelyon17_v0.5`); 72 conditions x 6 TTA candidates = 432
  cells, seeds {0,1,2,3}. Fields per cell: `Z` (10-dim label-free evidence),
  `a0`,`aa`,`B=aa-a0` (TRUE benefit), `seed`, `candidate`.
- Certificate = canonical `decide_kga`: Delta_hat = GradientBoostingRegressor(Z->B)
  (250 trees, depth 2, lr 0.05), eps = quantile(|Delta_hat-B|, 1-alpha). **alpha=0.10 FIXED.**
- **Sanity check passes**: 5-fold pooled cross-fit reproduces the paper's
  detectability harm-AUC(-Bhat) = **0.9136** (reported 0.9122) and pooled
  certificate_eps = **0.0628** (reported 0.0598). Correct fields confirmed.
- Rationale: the synthetic-calibrated eps violates Thm-2 exchangeability on
  Camelyon17. Re-estimating eps from a Camelyon17 CAL split held out **by seed**
  restores the premise (CAL/TEST residuals share one law). **tau* untouched.**

## Procedure & numbers (mean +/- 95% CI over the C(4,2)=6 seed splits; 2 CAL+2 TEST)
- **eps recalibrated and STABLE**: synthetic pooled eps ~0.060 (per-candidate
  0.085-0.162) collapses to **eps = 0.030** on Camelyon17 CAL (range 0.0067,
  CV 0.11). The repaired eps is small enough that the certificate now **commits**
  (coverage 0.65 [0.56, 0.75]) instead of abstaining.
- **regret_KGA = 0.0139 [0.0071, 0.0206]** — beats always-FREEZE 0.0517
  [0.0369, 0.0665] cleanly, but does **NOT** beat always-ADAPT 0.0130
  [0.0097, 0.0164]: CIs overlap (helpful-dominated panel; mean_B>0).
- **false-adapt = 0.185 [0.159, 0.211]** — once eps shrinks enough to commit,
  false-adapt is ~1.9x alpha. Per-candidate 0.18-0.28 (all >> alpha); per-candidate
  eps also less stable (CV 0.18-0.42).

## Reading
Recalibrating eps DID restore commitment with a small, stable eps — the
exchangeability fix works mechanically. But the committed decisions are not
level-alpha safe (false-adapt ~2x alpha) and KGA cannot beat always-adapt on
this helpful-dominated panel. The honest conclusion: **harm here is label-free
detectable (AUC 0.91) yet not certifiable at alpha=0.10 at n_eval=256/cell** —
a precise negative, not a policy win. The path to a win is more samples per cell
(tighter residuals) and/or a harm-skewed (not helpful-dominated) panel, not a
lower alpha or a touched tau*.
