# G5 — Official POEM/AETTA head-to-head: status + finalize (2026-07-23)

## Where G5 actually stands (better than the ledger's one-liner)

The official-arm machinery is **already built, pre-registered, and provisionally scored**:

- **Protocol:** `research_lock/WIN_HUNT_v4_PROTOCOL.yaml` (registered 2026-07-04), arm D.
  Bar frozen: WIN iff KGA beats official POEM AND dropout-AETTA (each 95% paired-bootstrap
  regret-gap CI entirely < 0, Holm at family-wise 0.05); LOSE/TIE reported as-is.
- **Official arms:** `poem_official.py` drives the *official POEM betting martingale*
  (protector.py's betting factor, SF-OGD fraction, wealth recursion; cdf.py's source-CDF
  PIT — reused verbatim from the official-code port) over **raw per-sample entropy
  streams**; `aetta_dropout.py` consumes the **real MC-dropout AETTA estimates** computed
  in the runner (N=10 passes, CIFAR-10 dropout 0.4, the paper's improved estimator with
  α=3, official EMA 0.6/0.4) and applies AETTA's own recovery logic. Every choice the
  papers leave open is pinned and documented in the module headers (P1–P4, D1–D2).
  This closes the "batch-summary port" caveat; the honest paper label is
  **"official-algorithm per-sample POEM and dropout-AETTA (pinned choices documented)"**.
- **Provisional result (seeds 0–1, 432 conditions/seed, VERDICT: WIN):**

  | policy | mean regret | false-adapt |
  |---|---|---|
  | **KGA** | **0.00154** | **0.000** |
  | official POEM (per-sample) | 0.01894 | 0.095 |
  | dropout-AETTA | 0.00233 | 0.144 |

  KGA vs POEM: −0.0174, CI [−0.0239, −0.0115], p_holm 2×10⁻⁴ → beats.
  KGA vs AETTA: −0.0008, CI [−0.00153, **−0.00008**], p_holm 0.031 → beats, **thin**.
  `replacement_eligible = true` (KGA FA 0 ≤ α).

## What this session verified (new)

The **unmodified** `score_official_headtohead.py` was executed in a clean environment on
a byte-copy of `stress_persample_v1` (seeds 0–1): it reproduces the locked provisional
result **to the digit** (all regrets, both CIs, both Holm p-values, verdict WIN).
Verification artifact: `ARM_D_rescore_seeds01.json` alongside this file. The pipeline is
deterministic and environment-portable; after your three runs, the official verdict is
one command.

## What remains (the only real gap): per-sample runs for seeds 2, 3, 4

On the Mac (same env as your seeds 0–1 runs; seed 2's stale partial npz files are
overwritten by a fresh run — the scorer requires the per_condition JSON, which only a
completed run writes):

```bash
cd <repo-root>
for S in 2 3 4; do
  caffeinate -is "$PY_CORE" docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks cifar10c --quick --data-root experiments/kbound/cifar \
    --methods tent --device mps --seed "$S" --log-samples \
    --out-results experiments/kbound/results/stress_persample_v1
done
```

Then score ONCE, seeds 0–4 (the official arm-D verdict; do not score subsets):

```bash
"$PY_CORE" experiments/kbound/poem_aetta/score_official_headtohead.py \
  --run-dir experiments/kbound/results/stress_persample_v1 \
  --dataset cifar10c --adapter tent --seeds 0 1 2 3 4 --nboot 10000
```

Or run `bash docs/research/kbound/g5_finalize/run_g5_finalize.sh`, which wraps both
steps with guards (refuses to score until per_condition JSONs exist for all five seeds).

## Pre-committed outcome handling (no goalposts move after the runs)

- **WIN at 5 seeds:** the paper's head-to-head section upgrades from "protocol-matched
  ports … official reproduction remains a camera-ready check" to the official-algorithm
  claim above; the baseline-faithfulness table gains the P1–P4/D1–D2 pinned-choice rows;
  G5 closes in the ledger. This is the sentence that moves the paper toward the solid 4.
- **TIE vs AETTA (live possibility — the current AETTA margin is ~1 permille with CI edge
  at −0.00008):** report WIN-vs-POEM + TIE-vs-AETTA exactly as scored; the paper claim
  becomes "beats official per-sample POEM and matches dropout-AETTA at 14× lower
  false-adapt (0.000 vs 0.144)" — still a substantial upgrade over ports, and the FA
  contrast is the safety story regardless.
- **LOSE (either):** reported as-is per the frozen bar; ports row stays, official row is
  added with the loss disclosed. (For calibration: POEM's current gap is ~11× in regret
  and 12 of 12 CI/Holm checks against it pass; the AETTA axis is the only close one.)

## Housekeeping

- `tmp/g5_transfer/stress_persample_v1.tgz` (11 MB) was created for this session's
  verification and can be deleted.
- Runtime expectation: your seeds 0–1 were completed in one afternoon session on MPS;
  budget the same order for the three remaining seeds (they can run sequentially
  unattended with `caffeinate`).
- After the runs land, re-tar the run dir (or just sync) and I can independently
  re-verify the 5-seed verdict in-container before you fold it into the manuscript.
