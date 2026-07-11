# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-05 11:05, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 34% | 0.0110 | 0.1205 | **0.0016** | 0.792 | YES | 0.1 |
