# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-15 23:51, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 52% | 0.0199 | 0.0113 | **0.0113** | 0.414 | no | - |
| imagenetc | eata | 4% | 0.0004 | 0.0292 | **0.0000** | 0.432 | YES | - |
| imagenetc | sar | 11% | 0.0425 | 0.0284 | **0.0128** | 0.431 | YES | 0.1 |
