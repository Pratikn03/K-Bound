# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-07 13:55, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 62% | 0.0625 | 0.0059 | **0.0059** | 0.412 | no | - |
| imagenetc | eata | 50% | 0.0645 | 0.0000 | **0.0000** | 0.406 | no | - |
| imagenetc | sar | 50% | 0.0762 | 0.0000 | **0.0000** | 0.406 | no | - |
