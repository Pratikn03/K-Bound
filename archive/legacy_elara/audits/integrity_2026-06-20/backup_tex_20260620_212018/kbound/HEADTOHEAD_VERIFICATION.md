# Head-to-head benchmark — adversarial verification (2026-06-20)

*Verdict on whether a "KGA beats POEM/AETTA" result from `experiments/kbound/poem_aetta/`
would be CREDIBLE. Done by direct code read of `baselines.py` + a synthetic harm/help
stress-test. The dedicated verifier agent was cut off by a session limit; this is the
main agent's own check.*

## Verdict: **CREDIBLE-ONLY-WITH-OFFICIAL-BASELINES** (apparatus fair; baselines reduced)

The comparison apparatus is sound and the baselines are *functional* (not strawmen), but
the in-sandbox POEM/AETTA are explicitly **reduced** versions, and one reduction
(batch-summary POEM) materially weakens POEM in helpful regimes. A win over the reduced
baselines is a **promising signal, not yet a publishable "beats SOTA" claim.**

## What holds up

1. **The apparatus is not rigged to win.** The build agent's `verify_headtohead.py`
   classifies synthetic WIN/TIE/LOSE worlds correctly (a POEM-favorable synthetic world
   returns LOSE; a zero-mean world returns TIE). The metric (regret-to-oracle) and
   paired-bootstrap+Holm machinery are byte-identical to the locked stress-grid analysis.
2. **The baselines genuinely detect harm** (synthetic stress-test, 40 conditions each):

   | Regime | POEM adapt / freeze | AETTA adapt / freeze |
   |---|---|---|
   | HELPFUL | 0.55 / 0.45 | 1.00 / 0.00 |
   | HARMFUL | **0.00 / 1.00** | **0.00 / 1.00** |

   Both **freeze 100% of harmful conditions** — real protection, not always-adapt/always-freeze
   degeneracy. AETTA is near-oracle on clean signals.

## What makes a win NOT yet credible (must fix before any paper claim)

1. **POEM is reduced (simplification S1) and it shows.** POEM adapts only **55%** of
   *helpful* conditions because its martingale runs on per-condition batch-summary
   `pre_entropy`, not the raw per-sample entropy stream it was designed for — so it rarely
   "certifies" a shift and under-adapts, leaving benefit on the table. **A KGA win over
   this POEM partly reflects POEM's lost helpful-regime power, not pure KGA superiority.**
   → *Required:* swap in official `yarinbar/poem` with the per-sample stream.
2. **AETTA's dropout-disagreement is proxied** by `1 − conf` / `pbal` (records store no
   dropout passes). It's a faithful-in-spirit, strong detector here, but it is **not** the
   real dropout-PDD estimator. → *Required:* swap in official `taeckyung/AETTA` dropout passes.
3. **Apples-to-apples: KGA has a 3rd action (ABSTAIN); POEM/AETTA do not.** Part of any
   KGA edge could come from *having* abstain, not from *better gating*. → *Required:* give
   POEM/AETTA an equivalent abstain/uncertainty option, **and** report a KGA-without-abstain
   ablation, to isolate that the gating wins.

## Status of the preliminary cached win

The build agent saw KGA regret 0.0016 vs POEM 0.0088 / AETTA 0.0073 on the cached
CIFAR-10-C stress arm (Holm-significant), then deleted it (not committed). That number is
**real and reproducible but rests on the reduced baselines** — so treat it as encouraging
pilot evidence that the mixed-regime story has legs, **not** as a result. The honest
headline only exists after fixes (1)–(3) + the multi-seed GPU run.

## Bottom line

The benchmark is correctly built and pre-registered, and the mixed-regime direction is the
right top-tier bet — the pilot signal is genuinely promising. But to put "beats POEM/AETTA"
in a NeurIPS/ICML/ICLR paper you must run it with the **official baseline repos + an abstain
option for the baselines + multi-seed**. Until then it is an honest pilot, and claiming the
win would be exactly the overclaim the rest of this program avoids.
