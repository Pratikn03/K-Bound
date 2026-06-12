# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-09 19:15, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 58% | 0.0188 | 0.0033 | **0.0033** | 0.774 | no | - |
| cifar101 | eata | 44% | 0.0045 | 0.0049 | **0.0049** | 0.775 | no | - |
| cifar101 | sar | 89% | 0.0914 | 0.0005 | **0.0005** | 0.771 | no | - |
