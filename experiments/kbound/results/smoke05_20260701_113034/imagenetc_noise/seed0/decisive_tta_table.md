# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-01 11:36, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 46% | 0.0354 | 0.0750 | **0.0000** | 0.458 | YES | 0.1 |
