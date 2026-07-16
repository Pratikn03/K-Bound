# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-15 16:35, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 56% | 0.0180 | 0.0163 | **0.0133** | 0.426 | YES | 0.3 |
| imagenetc | eata | 0% | 0.0000 | 0.0355 | **0.0012** | 0.445 | no | - |
| imagenetc | sar | 22% | 0.0595 | 0.0312 | **0.0091** | 0.441 | YES | 0.1 |
