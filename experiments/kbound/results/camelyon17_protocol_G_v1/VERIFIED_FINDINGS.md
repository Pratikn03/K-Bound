# Protocol G (Camelyon17 Tier-B headline): VERIFIED

*Locked analysis: `research_lock/CAMELYON17_PROTOCOL_G_v1.yaml` (2026-06-16).*
*Re-scores existing Protocol F records; no new GPU run.*

## Canonical method (headline)
- **KGA** = Algorithm 1 only: GBR `B_hat(Z)` + global conformal `eps` from dev residuals
- **Adapter:** `eata_online` (fixed)
- **Evidence:** rich 17-dim Z (Protocol F serialization)
- **Split:** DEV seeds {0,1}, TEST seeds {2,3,4} once

## Held-out test (n=54 cells)
| policy | regret-to-oracle |
|--------|------------------|
| always-adapt | 0.00132 |
| always-freeze | 0.07492 |
| **KGA** | **0.000036** |

- false-adapt: **2.56%** ≤ α=0.10
- commit: **90.7%**
- harmful base rate (this adapter): **29.6%**
- **beats both:** yes

## Why this is Tier B
- One method, one adapter, no multicandidate / PPI / Mondrian in the primary claim
- Real WILDS hospital shift, mixed harmful/helpful regime
- In-domain calibration by seed (not CIFAR ε transplant)

## Ablations (appendix only)
Pooled all adapters, GBR+global: regret 0.0019, FA 3.3%, beats both.
PPI+Mondrian: regret 0.0018, FA 3.3%, beats both.
Multicandidate router: fails (71% FA).

Reproduce:
```bash
python docs/research/kbound/scripts/analyze_F.py \
  --records experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json \
  --candidate eata_online --estimator gbr --conformal global \
  --output-dir experiments/kbound/results/camelyon17_protocol_G_v1/
```
