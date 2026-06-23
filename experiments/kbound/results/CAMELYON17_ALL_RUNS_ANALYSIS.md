# Natural-shift Protocols G & H — Summary

*Updated 2026-06-16 after Protocol H lock (iWildCam second replication).*

## Headline wins (canonical KGA only)

| Protocol | Dataset | Adapter | Split | n | regret KGA | regret adapt | regret freeze | FA | beats both |
|----------|---------|---------|-------|---|------------|--------------|---------------|-----|------------|
| **G** | Camelyon17 | eata_online | dev {0,1} / test {2,3,4} | 54 | 0.000036 | 0.00132 | 0.07492 | 2.6% | **yes** |
| **H** | iWildCam test | sar_online | dev {0} / test {1} | 72 | 0.00493 | 0.10720 | 0.00534 | 0% | **yes** |

Both re-score existing GPU records; no new forward passes.

## Protocol H honest scope

- **Regime:** 83% harmful geographic shift (WILDS camera traps)
- **Primary value:** 20× regret reduction vs always-adapt (0.11 → 0.005)
- **Caveat:** thin margin vs always-freeze (0.0049 vs 0.0053); damage-prevention framing
- **Prior null:** multicandidate route on same test cells did not beat both

## Office-Home (not Protocol H)

Deployed `eata_online_mild` is net-helpful on held-out test (KGA regret 0 = ties always-adapt).
`sar_online_aggressive` beats both on n=18 but was **not** pre-registered — not promoted.

## Artifacts

- `research_lock/CAMELYON17_PROTOCOL_G_v1.yaml`
- `research_lock/IWILDCAM_PROTOCOL_H_v1.yaml`
- `experiments/kbound/results/camelyon17_protocol_G_v1/`
- `experiments/kbound/results/iwildcam_protocol_H_v1/`

## Paper framing

- **Two** independent WILDS held-out beats-both replications under one algorithm (KGA)
- Camelyon: mixed shift, rich Z, large margin vs freeze
- iWildCam: harmful-dominated, standard Z, large margin vs adapt, modest vs freeze
