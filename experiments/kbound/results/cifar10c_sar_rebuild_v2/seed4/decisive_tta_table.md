# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-23 21:07, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | sar | 10% | 0.0003 | 0.1407 | **0.0015** | 0.811 | no | - |
