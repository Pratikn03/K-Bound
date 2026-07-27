# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-22 20:30, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | sar | 0% | 0.0000 | 0.0577 | **0.0000** | 0.834 | no | - |
