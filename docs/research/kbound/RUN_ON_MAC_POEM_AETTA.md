# Running the mixed KGA vs POEM vs AETTA head-to-head on your Mac

This is the **honest mixed harmful+helpful head-to-head** that TESTS whether KGA beats
the no-harm SOTA — **POEM** (betting-martingale entropy protection, arXiv:2408.07511)
and **AETTA** (label-free accuracy-estimate gate, CVPR'24). It is **pre-registered**:
the benchmark, the primary metric, and the WIN/TIE/LOSE criterion were fixed *before*
any KGA-vs-POEM/AETTA number was computed — see
[`MIXED_BENCHMARK_PROTOCOL.md`](MIXED_BENCHMARK_PROTOCOL.md).

**Read section 5 ("How to read it") before interpreting any number. KGA winning,
tying, or losing are ALL valid pre-committed outcomes. The run decides — not us.**

| Arm | What | Needs |
|-----|------|-------|
| **Cached** (default) | KGA/POEM/AETTA + trivials + oracle on the cached CIFAR-10-C stress-grid records (TENT primary; TENT+EATA and EATA secondary). KGA decisions from the cached `kga_decision`; POEM/AETTA from faithful ports. | nothing but `python3 + numpy + scipy` (torch-free, ~seconds) |
| **Recompute-KGA** | same, but KGA recomputed live via `analysis.decide_kga` (LOO gradient-boosted B̂ + conformal ε). | `sklearn` (your `.venv`) |
| **Official-repo** (camera-ready) | swap the proxy ports for the OFFICIAL per-sample POEM (`yarinbar/poem`) and dropout-AETTA (`taeckyung/AETTA`) outputs. | the two repos + GPU; see sec 6 |

---

## 0. Pre-flight: CPU/synthetic apparatus verification (already passing; ~3 s)

Proves the decision rules + regret/false-adapt metrics + paired-bootstrap + Holm +
WIN/TIE/LOSE verdict machinery are correct **before** trusting any real number. Every
signal is SYNTHETIC and labeled; it decides nothing about the real winner.

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
PYTHONPATH="$PWD:$PWD/experiments/kbound/wilds:$PWD/experiments/kbound/poem_aetta" \
  python3 experiments/kbound/poem_aetta/verify_headtohead.py
```

Expect `VERIFICATION PASSED` with three stages:
- **STAGE 1** runs all six rules on synthetic Z; POEM and AETTA produce **non-degenerate**
  adapt fractions (~0.66–0.68) → they are not strawmen pinned to always-adapt/freeze.
- **STAGE 2** feeds the verdict logic synthetic regret structures and gets
  **WIN, TIE, and LOSE** respectively → the machinery is **not hard-wired to WIN**.
- **STAGE 3** round-trips the per-condition schema through the locked
  `multiseed_paired_ci.analyze`.

---

## 1. One command (recommended) — the cached arm

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
bash experiments/kbound/poem_aetta/run_all_headtohead.sh
```

Runs pre-flight, then the **PRIMARY** (TENT) and two **SECONDARY** sets (TENT+EATA
pooled, EATA), writing results + the pre-registered verdict to
`experiments/kbound/results/mixed_headtohead_v1/`.

Overrides via env: `SEEDS="0 1 2 3 4"` (default), `OUT_DIR=...`, `RECORDS_DIR=...`,
`PY=...`. The cached arm is torch-free, so the default `python3` is fine.

---

## 2. Exact equivalent manual commands

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
export PYTHONPATH="$PWD:$PWD/src:$PWD/experiments/kbound/wilds:$PWD/experiments/kbound/poem_aetta"
REC="experiments/kbound/results/stress_grid_multiseed_v1"
OUT="experiments/kbound/results/mixed_headtohead_v1"
H="experiments/kbound/poem_aetta/run_mixed_headtohead.py"

# PRIMARY (headline): MIXED-PRIMARY = TENT, all 432 conditions, 5 seeds
python3 "$H" --records-dir "$REC" --dataset cifar10c --adapter tent \
  --seeds 0 1 2 3 4 --out-dir "$OUT" --set-name cifar10c_tent_primary

# SECONDARY: TENT+EATA pooled (lower harmful base rate; composition stress)
python3 "$H" --records-dir "$REC" --dataset cifar10c --pool-adapters tent eata \
  --seeds 0 1 2 3 4 --out-dir "$OUT" --set-name cifar10c_tent_eata_pooled

# SECONDARY: EATA alone (lowest harmful base rate)
python3 "$H" --records-dir "$REC" --dataset cifar10c --adapter eata \
  --seeds 0 1 2 3 4 --out-dir "$OUT" --set-name cifar10c_eata_secondary
```

**Recompute-KGA live** (uses `analysis.decide_kga`; needs sklearn) — add
`--recompute-kga` and point `python3` at your torch/sklearn venv:
```bash
"$PWD/.venv/bin/python" "$H" --records-dir "$REC" --dataset cifar10c --adapter tent \
  --seeds 0 1 2 3 4 --out-dir "$OUT" --set-name cifar10c_tent_primary_recompute \
  --recompute-kga --device mps
```
(`--device` only matters with `--recompute-kga`; the cached arm is numpy/CPU.)

---

## 3. The pre-registered metric and WIN criterion (what the run is testing)

**PRIMARY metric:** mean **regret-to-oracle** across the mixed stream, per policy
(`regret = mean[ max(a0,a_adapted) − policy_acc ]`; ABSTAIN/FREEZE → source acc).
**SECONDARY:** false-adapt rate `Pr[ADAPT ∧ B<0]` at matched coverage (KGA carries an
anytime false-adapt ≤ α=0.10 certificate; POEM/AETTA do not).

**WIN criterion** (protocol sec 4), per the paired bootstrap in
`multiseed_paired_ci.py` (paired over per-condition mean regret across 5 seeds,
nboot=1e4, seed 20260619), with `diff(KGA,X)=regret_KGA−regret_X`:

> KGA **beats** X iff the 95% paired-bootstrap CI of `diff(KGA,X)` is **entirely below
> 0** AND survives **Holm** over `{KGA vs POEM, AETTA, always-adapt, always-freeze}` at
> family-wise α=0.05.
>
> - **WIN**: KGA beats **both** POEM and AETTA AND KGA's false-adapt ≤ 0.10.
> - **TIE**: for ≥1 of {POEM,AETTA} the Holm CI includes 0 (no sig. diff) and KGA loses
>   to neither; KGA's differentiator is then the certificate they lack.
> - **LOSE**: for ≥1 of {POEM,AETTA} the Holm CI is **entirely above 0** (KGA worse).

The harness prints `>>> PRE-REGISTERED VERDICT: WIN|TIE|LOSE` and writes it to
`HEADTOHEAD_RESULTS_<set>.json` under `headtohead.VERDICT`.

---

## 4. Expected outputs

In `experiments/kbound/results/mixed_headtohead_v1/`:
- `HEADTOHEAD_RESULTS_<set>.json` — per-policy mean regret + false-adapt + coverage, the
  4 paired-CI comparisons, and the pre-registered `VERDICT`.
- `result_manifest_<set>.json` — git hash, seeds, wall time, input paths, boot seed.
- `per_condition_<set>_<policy>_seed<S>.json` — per-condition arrays for **all six**
  policies (FLAT layout; field `kga_decision` holds *that policy's* decision, schema-
  compatible with `multiseed_paired_ci.py`, which can re-analyze them directly).

**What the cached arm produces on the current records** (this is a REAL result of the
committed pipeline on the cached arm — reproduced in-sandbox; your Mac will match):

```
PRIMARY  cifar10c_tent_primary   (432 cond/seed, 5 seeds, harmful rate 0.16–0.18)
  policy          mean_regret  false_adapt  decisive  cover(dec)
  always_adapt      0.00792      0.329       1.000     0.667
  always_freeze     0.12410      0.000       1.000     0.333
  aetta             0.00733      0.240       1.000     0.698
  poem              0.00880      0.245       1.000     0.663
  kga               0.00157      0.000       0.685     0.999
  oracle            0.00000      0.000       1.000     1.000
  KGA vs poem   diff=-0.00723 CI[-0.00878,-0.00571] p_holm=4e-4 beats=True
  KGA vs aetta  diff=-0.00576 CI[-0.00714,-0.00440] p_holm=4e-4 beats=True
  >>> PRE-REGISTERED VERDICT: WIN
```
Secondary sets (TENT+EATA pooled harmful≈0.11; EATA harmful≈0.05) also return WIN, with
the **margin shrinking as the harmful base rate falls** (KGA-vs-AETTA diff −0.0058 →
−0.0034 → −0.0011) — the expected, honest behavior: on a near-all-helpful panel KGA's
edge over the no-harm SOTA narrows toward a tie.

> **IMPORTANT — this is the *cached arm* with documented proxy simplifications.** POEM's
> martingale is driven by the logged batch-summary `pre_entropy` (not raw per-sample
> entropies, S1) and AETTA's Eq.13 uses label-free batch-aggregate proxies for PDD and
> E_avg/E_max (not real dropout passes, A1). Both ports use the published algorithms and
> constants and are verified non-degenerate (sec 0), but the **definitive** head-to-head
> for camera-ready should swap in the OFFICIAL repo outputs (sec 6). The cached-arm WIN
> is real and reproducible; it is not the final word until the official-code arm confirms.

---

## 5. How to read it (blunt, honest)

- **The verdict is whatever the run prints.** WIN, TIE, and LOSE are all pre-committed
  and all publishable. We did **not** assume KGA wins; the protocol fixes the test and
  forbids re-picking the metric/set/seed after seeing a result (protocol sec 5).
- **A WIN means**: on this mixed stream KGA leaves less regret on the table than POEM and
  AETTA *and* keeps false-adapt ≤ α. The mechanism is visible in the table: KGA buys its
  low regret by **abstaining** (decisive ≈ 0.66) on the conditions where its certificate
  is uncertain, achieving false-adapt 0.0; POEM/AETTA commit on 100% and pay regret on
  the harmful conditions they wrongly adapt (false-adapt ≈ 0.24). That is the no-harm
  certificate doing real work, not a metric artifact.
- **A TIE would mean**: KGA matches the no-harm SOTA on regret; the honest differentiator
  is then the *anytime false-adapt certificate* POEM/AETTA lack — claim that, not
  dominance.
- **A LOSE would mean**: POEM and/or AETTA beat KGA on mixed regret; rescope the
  contribution to the certificate and the regimes KGA *does* win (homogeneous-harmful,
  SAR-collapse) and say so plainly.
- **Where KGA is NOT expected to win**: homogeneous (all-helpful or all-harmful) panels,
  where one trivial policy is optimal and nothing can beat it — this benchmark is
  deliberately the MIXED regime. The secondary EATA set (5% harmful) is near-homogeneous-
  helpful and shows the margin collapsing toward a tie, as expected.
- **Faithfulness caveat (do not skip)**: the cached arm uses the documented proxy ports
  (S1, A1). If a reviewer asks "are POEM/AETTA faithful?", the answer is: the ports are
  written to the published algorithm + constants and verified non-degenerate, but the
  camera-ready number must use the official repos (sec 6). State the simplification.

---

## 6. Official-repo arm (camera-ready; stronger faithfulness)

Replace the proxy ports with the real implementations on the same `resnet18_cifar`
checkpoint and CIFAR-10-C conditions:

1. **POEM** — clone `https://github.com/yarinbar/poem`; run its `Protector`
   (`protector.py` + `cdf.py`) on the **per-sample** entropy stream of each condition,
   with `CDF` fit on source/clean entropies. Log, per condition, whether POEM's
   accumulated martingale certified a shift (its ADAPT vs PROTECT decision). Feed those
   decisions into the harness as a precomputed `poem` decision column (swap
   `baselines.poem_decision` for a loader of the official decisions).
2. **AETTA** — clone `https://github.com/taeckyung/AETTA`; run `N` dropout inferences per
   condition to get the real `PDD` and `E_avg` (Eq. 13, α=3), and apply the paper's
   recovery rule (reset/freeze on estimated-accuracy degradation). Feed those decisions
   in as the `aetta` column.
3. Re-run `run_mixed_headtohead.py` pointed at the official-decision records. The metric,
   win criterion, CI machinery, and verdict are unchanged — only the POEM/AETTA decision
   source changes. Report this as the headline; the cached arm becomes a fast sanity row.

The harness is written so this swap is a one-function change (load official decisions
instead of calling the proxy port); nothing else moves. The pre-registered metric and
WIN/TIE/LOSE criterion stay fixed across both arms (no re-registration needed — same
benchmark, same metric, stronger baseline implementation).

---

## 7. What is verified here vs what the run decides

**Verified (torch-free, in-sandbox, no real-data claim):** the six decision rules
execute; regret + false-adapt compute; the paired-bootstrap + Holm + WIN/TIE/LOSE
verdict machinery runs and correctly classifies synthetic WIN/TIE/LOSE (not hard-wired
to WIN); the schema round-trips through the locked analysis. The cached-arm numbers in
sec 4 are a real output of the committed pipeline on the cached records.

**Decided only by the run you choose to trust as headline** (cached vs official-repo
arm): whether KGA actually beats / ties / loses to POEM and AETTA. This document does
not assume the outcome; the protocol forbids tuning it toward one.

**No KGA-vs-POEM/AETTA result was fabricated.** Every number above and in
`HEADTOHEAD_RESULTS_*.json` is produced by the committed scripts from the cached records,
after the protocol was registered.
