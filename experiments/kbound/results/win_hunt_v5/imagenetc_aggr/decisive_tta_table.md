# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-08 21:54, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 63% | 0.0216 | 0.0114 | **0.0114** | 0.421 | no | - |
| imagenetc | eata | 0% | 0.0000 | 0.0336 | **0.0000** | 0.443 | no | - |
| imagenetc | sar | 15% | 0.0625 | 0.0319 | **0.0108** | 0.441 | YES | 0.1 |
