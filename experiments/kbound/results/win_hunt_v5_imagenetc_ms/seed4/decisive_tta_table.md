# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-16 13:36, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 59% | 0.0183 | 0.0165 | **0.0165** | 0.421 | no | - |
| imagenetc | eata | 4% | 0.0001 | 0.0412 | **0.0001** | 0.445 | no | - |
| imagenetc | sar | 15% | 0.0561 | 0.0389 | **0.0154** | 0.443 | YES | 0.1 |
