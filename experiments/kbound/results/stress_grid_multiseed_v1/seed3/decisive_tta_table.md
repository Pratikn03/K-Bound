# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-12 04:10, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 33% | 0.0078 | 0.1239 | **0.0014** | 0.795 | YES | 0.1 |
| cifar10c | eata | 25% | 0.0030 | 0.1315 | **0.0015** | 0.802 | YES | 0.1 |
| cifar10c | sar | 7% | 0.0003 | 0.1403 | **0.0019** | 0.811 | no | - |
