# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-23 16:01, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | sar | 7% | 0.0003 | 0.1403 | **0.0019** | 0.811 | no | - |
