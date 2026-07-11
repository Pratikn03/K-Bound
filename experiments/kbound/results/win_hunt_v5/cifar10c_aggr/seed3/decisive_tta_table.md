# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-06 01:17, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 38% | 0.0119 | 0.0837 | **0.0012** | 0.763 | YES | 0.0 |
| cifar10c | eata | 20% | 0.0030 | 0.1044 | **0.0019** | 0.784 | YES | 0.1 |
| cifar10c | sar | 89% | 0.1951 | 0.0113 | **0.0008** | 0.690 | YES | 0.1 |
