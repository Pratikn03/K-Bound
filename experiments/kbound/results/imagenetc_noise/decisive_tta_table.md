# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-09 19:06, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 8% | 0.0003 | 0.0440 | **0.0042** | 0.454 | no | - |
| imagenetc | eata | 3% | 0.0001 | 0.0349 | **0.0003** | 0.444 | no | - |
| imagenetc | sar | 31% | 0.0606 | 0.0277 | **0.0086** | 0.437 | YES | 0.1 |
