# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-06 02:59, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 37% | 0.0110 | 0.0839 | **0.0016** | 0.763 | YES | 0.1 |
| cifar10c | eata | 17% | 0.0022 | 0.1053 | **0.0020** | 0.784 | YES | 0.1 |
| cifar10c | sar | 90% | 0.1950 | 0.0079 | **0.0005** | 0.687 | YES | 0.1 |
