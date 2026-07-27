# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-23 00:25, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | sar | 10% | 0.0003 | 0.1400 | **0.0014** | 0.811 | no | - |
