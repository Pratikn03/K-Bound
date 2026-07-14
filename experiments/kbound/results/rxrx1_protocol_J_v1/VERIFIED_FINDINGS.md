# Protocol J (RxRx1 audit): VERIFIED

*Locked analysis: `research_lock/RXRX1_PROTOCOL_J_v1.yaml` (2026-06-16).*
*Re-scores existing `rxrx1_protocol_c_9plus` records; no new GPU run.*

## Canonical method (audit — not Tier-B headline)
- **KGA** = GBR `B_hat(Z)` + global conformal `eps` from dev residuals (same as Protocol G/H)
- **Adapter:** `sar_online` (fixed; harmful-stream SAR; catastrophic always-adapt on this shift)
- **Evidence:** 11-dim label-free Z
- **Split:** DEV stream seeds {0–4}, TEST seeds {5–9} once (60 held-out cells per model seed)

## Held-out test — model seed 0 (primary)
| policy | regret-to-oracle |
|--------|------------------|
| always-adapt | 0.2531 |
| always-freeze | 0.0000 |
| **KGA** | **0.0000** |

- false-adapt: **0%** (0 adapts committed)
- commit: **100%** (all FREEZE — harmful-dominated stream)
- adapt rate: **0%**
- harmful base rate (sar_online, full grid): **100%**
- **beats both:** no (ties freeze; does not strictly beat)

## Robustness (same split, seeds 5–9)
| model seed | regret KGA | regret freeze | regret adapt | beats both |
|------------|------------|---------------|--------------|------------|
| 0 | 0.0000 | 0.0000 | 0.2531 | no |
| 1 | 0.0000 | 0.0000 | 0.2531 | no |
| 2 | 0.0000 | 0.0000 | 0.2531 | no |

## Verdict: **FREEZE-ORACLE AUDIT PASS** (not headline)

**What this confirms:** On the deployed harmful adapter, canonical KGA is a **damage-prevention**
policy — it matches always-freeze regret and avoids the SAR collapse (always-adapt regret 0.25).

**What this is not:** A Tier-B **beats-both** win. Freeze is already oracle-optimal here; KGA
correctly commits to freeze everywhere on held-out test.

**Paper role:** Supports Table RxRx1 / appendix row — harmful-and-detectable corner at 1,139-class
scale; insurance against adaptation, not a policy flip over freeze.

Reproduce:
```bash
~/.venv_wilds/bin/python docs/research/kbound/scripts/analyze_F.py \
  --records experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json \
  --candidate sar_online --estimator gbr --conformal global \
  --dev-seeds 0 1 2 3 4 --test-seeds 5 6 7 8 9 \
  --output-dir experiments/kbound/results/rxrx1_protocol_J_v1/
```
