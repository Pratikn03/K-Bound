# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-11 23:33, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 32% | 0.0076 | 0.1239 | **0.0016** | 0.795 | YES | 0.1 |
| cifar10c | eata | 26% | 0.0033 | 0.1312 | **0.0013** | 0.802 | YES | 0.0 |
| cifar10c | sar | 9% | 0.0003 | 0.1410 | **0.0017** | 0.812 | no | - |
