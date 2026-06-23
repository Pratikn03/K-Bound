# Protocol K (CIFAR-10.1 audit): VERIFIED

*Locked analysis: `research_lock/CIFAR101_PROTOCOL_K_v1.yaml` (2026-06-16).*
*Re-scores existing `cifar101_multiseed_v1` per_condition records; no new GPU run.*

## Canonical method (cross-seed G/H template)
- **KGA** = GBR `B_hat(Z)` + global conformal `eps` from **dev stream seeds**
- **Adapter:** `tent` (fixed; low-margin natural-shift probe)
- **Evidence:** 11-dim label-free Z
- **Split:** DEV seeds {0,1,2}, TEST seeds {3,4} once (48 held-out cells)

## Held-out test (cross-seed calibration)
| policy | regret-to-oracle |
|--------|------------------|
| always-adapt | 0.0190 |
| always-freeze | 0.0017 |
| **KGA** | **0.0021** |

- false-adapt: **44.4%** (exceeds α=0.10)
- commit: **87.5%**
- adapt rate: **37.5%**
- **beats both:** no

## Verdict: **NOT CLEARED** under G/H cross-seed bar

### Important distinction (two calibration modes)

| Mode | Split | Tent false-adapt | KGA vs freeze | Paper table |
|------|-------|------------------|---------------|-------------|
| **Within-seed** (runner default) | ε fit on same seed’s 24 conditions | **0%** on seeds 3–4 | **ties** freeze | `decisive_tta_results.json` |
| **Cross-seed** (Protocol K = G/H template) | ε fit on seeds 0–2, test 3–4 | **44%** | slightly **worse** than freeze | this audit |

**Interpretation:** CIFAR-10.1 is a **low-margin** natural shift where benefit signals are small
(|B̄| ≈ 0.01) and **do not transfer across independent stream seeds**. The paper’s abstain-heavy,
0%-FA story is valid **within each seed** (in-domain ε). Applying the Camelyon/iWildCam
**cross-seed** protocol here fails — certificate and B_hat do not generalize across seeds.

**Paper role:** Honest regime map entry — low-margin natural shift needs in-domain calibration;
do **not** claim as a third Tier-B held-out win under the G/H cross-seed template. Keep main
results as within-seed pooled summary (`cifar101_multiseed_v1/pooled_summary.json`).

Reproduce:
```bash
~/.venv_wilds/bin/python docs/research/kbound/scripts/analyze_F.py \
  --records \
    experiments/kbound/results/cifar101_multiseed_v1/seed0/per_condition_cifar101_tent_seed0.json \
    experiments/kbound/results/cifar101_multiseed_v1/seed1/per_condition_cifar101_tent_seed1.json \
    experiments/kbound/results/cifar101_multiseed_v1/seed2/per_condition_cifar101_tent_seed2.json \
    experiments/kbound/results/cifar101_multiseed_v1/seed3/per_condition_cifar101_tent_seed3.json \
    experiments/kbound/results/cifar101_multiseed_v1/seed4/per_condition_cifar101_tent_seed4.json \
  --candidate tent --estimator gbr --conformal global \
  --dev-seeds 0 1 2 --test-seeds 3 4 \
  --output-dir experiments/kbound/results/cifar101_protocol_K_v1/
```
