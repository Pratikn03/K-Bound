# Protocol H (iWildCam Tier-B replication): VERIFIED

*Locked analysis: `research_lock/IWILDCAM_PROTOCOL_H_v1.yaml` (2026-06-16).*
*Re-scores existing `iwildcam_full_test` records; no new GPU run.*

## Canonical method (headline)
- **KGA** = Algorithm 1 only: GBR `B_hat(Z)` + global conformal `eps` from dev residuals
- **Adapter:** `sar_online` (fixed before scoring; SAR harmful-stream family, online parity with Camelyon G)
- **Evidence:** 11-dim label-free Z from iWildCam full-test serialization
- **Split:** DEV seed {0}, TEST seed {1} once (72 held-out cells)

## Held-out test (n=72 cells)
| policy | regret-to-oracle |
|--------|------------------|
| always-adapt | 0.1072 |
| always-freeze | 0.0053 |
| **KGA** | **0.0049** |

- false-adapt: **0%** ≤ α=0.10
- commit: **88.9%**
- adapt rate: **1.4%** (harmful-dominated stream; gate mostly freezes)
- harmful base rate (this adapter): **83.3%**
- **beats both:** yes

## Honest framing
- **Regime:** harmful-dominated geographic shift (WILDS camera traps); primary value is
  catastrophic-harm avoidance vs always-adapt (20× regret reduction), with a **modest**
  margin over always-freeze (~8% relative).
- **Contrast with prior null:** multicandidate-route analysis abstained on 92% of cells and
  did not beat both; canonical single-adapter KGA clears the Protocol G bar.
- **Not freeze-oracle:** KGA regret (0.0049) is strictly below freeze (0.0053), but the
  margin is thin — report as damage-prevention, not a large policy flip.

## Why this is Tier B (second natural shift)
- Independent WILDS benchmark, different modality from Camelyon17 histopathology
- One method, one adapter, no multicandidate / PPI / Mondrian in primary claim
- In-domain calibration by seed (not ε transplant)

Reproduce:
```bash
python docs/research/kbound/scripts/analyze_F.py \
  --records experiments/kbound/results/iwildcam_full_test/result_e40faf29.json \
  --candidate sar_online --estimator gbr --conformal global \
  --dev-seeds 0 --test-seeds 1 \
  --output-dir experiments/kbound/results/iwildcam_protocol_H_v1/
```
