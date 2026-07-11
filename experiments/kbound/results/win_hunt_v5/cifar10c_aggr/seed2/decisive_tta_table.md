# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-05 23:35, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 36% | 0.0111 | 0.0839 | **0.0018** | 0.764 | YES | 0.1 |
| cifar10c | eata | 19% | 0.0033 | 0.1038 | **0.0017** | 0.783 | YES | 0.1 |
| cifar10c | sar | 89% | 0.1888 | 0.0089 | **0.0004** | 0.689 | YES | 0.1 |
