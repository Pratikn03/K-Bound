# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-22 21:11, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | sar | 12% | 0.0004 | 0.1404 | **0.0013** | 0.811 | no | - |
