# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-01 11:39, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 58% | 0.0210 | 0.0021 | **0.0021** | 0.774 | no | - |
