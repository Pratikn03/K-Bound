# Spotlight pilot — does KGA-gating beat the best single TTA method?

This is the **free go/no-go** before committing GPU/cluster compute to the ImageNet-C/ViT
scale-up. It tests, on CIFAR-100-C, whether the K-Bound knowability gate used as a
**meta-TTA router** beats the best single adapter — the result that would make this a
top-tier "beats-SOTA" paper rather than an ~80 safety-certificate paper.

Read `SPOTLIGHT_PILOT_PROTOCOL.md` first — it is pre-registered (sealed before running),
with the WIN criterion and all three possible outcomes fixed in advance.

## Run it (free GPU — Google Colab or Kaggle)

1. New Colab notebook → Runtime → **GPU (T4)**.
2. Upload `kga_meta_tta_pilot.py`, then in a cell:
   ```
   !pip -q install robustbench
   !python kga_meta_tta_pilot.py --seeds 0 1 2 --n_examples 2000 --probe_k 64
   ```
   (~30–90 min. First run downloads CIFAR-100-C ≈ 3 GB to `./data`.)
3. Output → `kga_meta_pilot_results.json` with the verdict.

Quick smoke test (2 min): `--seeds 0 --n_examples 200`.
Pure label-free gate (no probe labels): `--probe_k 0`.

## How to read the verdict (decided in advance, not after)

| verdict | meaning | next step |
|---|---|---|
| **CONVERTS** | KGA-Meta mean error < best single AND 95% CI of the gap excludes 0 | the bet is **alive** → pre-register + run the ImageNet-C / ViT-B scale-up (the spotlight result) |
| **PARTIAL** | beats some adapters, ties the best | principled safety+selection result; strengthens the ~80 paper, **no spotlight claim** |
| **FAILS** | no improvement over best single | honest null; the gate is a guard, not a method; report as-is |

Whatever it prints is the truth — do not retune to flip it (the protocol forbids it).

## Honest scope (so the result is credible)

- **Pool = {frozen, Tent, EATA-lite}** for speed. For the *publishable* comparison, plug in the
  **official** adapters so reviewers trust the baselines:
  - Tent — https://github.com/DequanWang/tent
  - EATA — https://github.com/mr-eggplant/EATA
  - SAR — https://github.com/mr-eggplant/SAR
  - CoTTA — https://github.com/qinenergy/cotta
  - RoTTA — https://github.com/BIT-DA/RoTTA
- **Model/data** via RobustBench (`Hendrycks2020AugMix_WRN`) so numbers are literature-comparable.
- The gate runs at the **target-label-light** operating point (k labels/corruption) that K-Bound
  already legitimizes; `--probe_k 0` tests the stricter pure-label-free gate.
- A CIFAR-100-C CONVERTS is **necessary, not sufficient** — the spotlight claim needs the
  ImageNet-C / ViT-B replication at scale, which needs a real GPU box (grant or host lab).

## The realistic path (be honest with yourself)

1. Run this pilot (free, this week).
2. **If FAILS/PARTIAL:** submit the strong ~80 paper now (TMLR or a top workshop) — it's a genuine,
   honest accept, and exceptional for an undergrad. The spotlight bet is closed via this route.
3. **If CONVERTS:** that's your spotlight bet — but the scale-up needs compute + baseline-repro
   credibility you can't supply solo. **This is why the professor emails matter:** a host lab gives
   you the cluster, the official-baseline code, and a co-author. Lead with "the pilot shows the gate
   beats the best single adapter on CIFAR-100-C; I want to scale it to ImageNet-C/ViT."
