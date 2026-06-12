# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-08 19:50, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 58% | 0.0176 | 0.0025 | **0.0025** | 0.774 | no | - |
| cifar101 | eata | 46% | 0.0050 | 0.0044 | **0.0044** | 0.776 | no | - |
| cifar101 | sar | 38% | 0.0050 | 0.0060 | **0.0048** | 0.778 | YES | 0.1 |
