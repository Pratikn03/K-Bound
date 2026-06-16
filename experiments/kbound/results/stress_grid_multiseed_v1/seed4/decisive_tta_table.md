# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-12 08:47, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 32% | 0.0078 | 0.1249 | **0.0019** | 0.796 | YES | 0.1 |
| cifar10c | eata | 23% | 0.0030 | 0.1321 | **0.0010** | 0.803 | YES | 0.0 |
| cifar10c | sar | 10% | 0.0003 | 0.1407 | **0.0015** | 0.811 | no | - |
