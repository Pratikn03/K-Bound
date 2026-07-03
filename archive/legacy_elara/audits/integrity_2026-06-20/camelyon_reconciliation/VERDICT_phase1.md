# PHASE 1 VERDICT — Camelyon17 Protocol G reconciliation (2026-06-20)

## VERDICT: NOT a beats-both win. Domain-pooling artifact → reclassify as safety/abstention (no-harm on OOD).

Interpretation (A) "domain-pooling artifact" is correct. Interpretation (B) "legitimate
held-out OOD win" is **refuted**: the n=54 "held-out test" set is a random-**seed** split that
pools in-distribution `id_val` data with the two OOD domains. On genuinely held-out OOD data
the win disappears. Per the integrity policy the conservative reading is mandated, and here the
evidence is unambiguous (not merely ambiguous-so-conservative).

## Evidence chain (file:line + exact split)

1. **Split is by random seed, never by domain/hospital.**
   `docs/research/kbound/scripts/analyze_F.py` `run_split()`:
   `cal = np.isin(sd, cal_seeds); tst = np.isin(sd, test_seeds)` where `sd` = per-record
   `seed`. Module docstring: "Split BY SEED: DEV = {0,1} ... TEST = {2,3,4}". The decision
   loader `_one_record()` reads `comp` (cell = iid/imbalanced/single_class) and **never reads
   the `domain` field**. Domain (test/val/id_val) is invisible to split and decision.

2. **Records carry a domain label the analysis ignores.**
   `experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json`: 540 records =
   3 domains {test:180, val:180, id_val:180} × 6 adapters. For `eata_online`: 90 records,
   exactly 30/domain, 6 per (domain,seed). So test_seeds {2,3,4} → n=54 = 18 OOD-test +
   18 OOD-val + **18 in-distribution id_val**.

3. **Reported Protocol G is the pooled number (reproduced exactly).**
   `experiments/kbound/results/camelyon17_protocol_G_v1/analyze_F_results.json`:
   n_test=54, regret_kga=3.616898e-05, regret_adapt=1.320168e-03, regret_freeze=7.49240e-02,
   false_adapt=0.025641, verdict_win=true. My re-run of the POOLED slice reproduces all of
   these to the last digit (`recon_results.json` → POOLED_test_val_idval).
   The `camelyon17_diagnostics_resolved_v1/protocol_G_rerun/` file is byte-identical (same
   pooled computation; not a hospital split).

4. **The harm KGA navigates is in-distribution, near-zero magnitude.**
   Harm profile for eata_online (B<0 = adapting reduces true accuracy):
   - OOD **test**:  n=30, mean_B=+0.1568, frac_harm=0.000, min_B=+0.0244  (never harms)
   - OOD **val**:   n=30, mean_B=+0.0829, frac_harm=0.000, min_B=+0.0244  (never harms)
   - **id_val** (in-distribution): n=30, mean_B=−0.0075, frac_harm=0.767, min_B=−0.0596

5. **On held-out OOD alone there is NO beats-both (re-ran the locked gbr+global decision).**
   - OOD **test only** (n=18): regret_kga=0.0, regret_adapt=0.0, regret_freeze=0.1381,
     FA=0.0 → beats_both = **FALSE** (KGA exactly ties always-adapt).
   - OOD **val only** (n=18): regret_kga=0.0 = regret_adapt → beats_both = **FALSE**.
   - OOD **test+val** (n=36, all genuinely OOD): regret_kga=0.0 = regret_adapt →
     beats_both = **FALSE**.
   - **id_val only** (n=18, in-distribution): false_adapt=0.80 (>α), regret_freeze (2.2e-4)
     < regret_kga (7.1e-4) < regret_adapt (4.0e-3) → here always-FREEZE is ~optimal and KGA
     loses to freeze; beats_both = **FALSE**.

## Mechanism (why the pooled number looks like a win)
Always-adapt only incurs regret on the in-distribution id_val domain (the only place eata
sometimes hurts, at sub-percent magnitude). KGA mostly freezes/abstains there, so when id_val
is **pooled** with the two harm-free OOD domains, always-adapt picks up regret 1.32e-3 that KGA
avoids → KGA "beats both." Remove the in-distribution id_val and the OOD-only always-adapt
regret is exactly 0, so the win vanishes. The headline FA of 2.6% is likewise a pooling
dilution of the true in-distribution FA of 80%.

## Consequence for the paper
Camelyon17 is **not** a real-shift beats-both win and must not be headlined as one. Honest
reclassification: **safety/abstention (no-harm)** — on genuine OOD, KGA never false-adapts
(FA 0%) and ties the always-adapt oracle; it does not *beat* both baselines. This is a null
for the win-set and a positive for the safety story.

Artifacts: `camelyon_G_reconciliation.py`, `recon_results.json` (this directory).
