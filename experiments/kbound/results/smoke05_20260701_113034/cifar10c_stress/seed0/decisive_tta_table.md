# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-01 11:30, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 19% | 0.0012 | 0.0383 | **0.0020** | 0.813 | no | - |
