# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-05 07:29, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 34% | 0.0086 | 0.1232 | **0.0016** | 0.794 | YES | 0.1 |
| cifar10c | eata | 27% | 0.0037 | 0.1311 | **0.0015** | 0.802 | YES | 0.0 |
| cifar10c | sar | 16% | 0.0018 | 0.1372 | **0.0018** | 0.808 | no | 0.1 |
