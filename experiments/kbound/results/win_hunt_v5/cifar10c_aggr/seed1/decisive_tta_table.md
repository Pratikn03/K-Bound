# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-05 18:02, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 38% | 0.0122 | 0.0838 | **0.0012** | 0.764 | YES | 0.1 |
| cifar10c | eata | 20% | 0.0030 | 0.1043 | **0.0017** | 0.785 | YES | 0.1 |
| cifar10c | sar | 89% | 0.1963 | 0.0081 | **0.0011** | 0.688 | YES | 0.1 |
