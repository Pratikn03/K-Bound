# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-16 06:24, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 48% | 0.0175 | 0.0167 | **0.0167** | 0.421 | no | - |
| imagenetc | eata | 0% | 0.0000 | 0.0315 | **0.0000** | 0.436 | no | - |
| imagenetc | sar | 15% | 0.0441 | 0.0290 | **0.0056** | 0.434 | YES | 0.1 |
