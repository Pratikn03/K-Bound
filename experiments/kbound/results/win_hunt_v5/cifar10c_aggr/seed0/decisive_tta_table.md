# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-05 16:13, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 37% | 0.0107 | 0.0840 | **0.0014** | 0.763 | YES | 0.0 |
| cifar10c | eata | 19% | 0.0029 | 0.1051 | **0.0021** | 0.784 | YES | 0.1 |
| cifar10c | sar | 89% | 0.1924 | 0.0116 | **0.0012** | 0.690 | YES | 0.1 |
