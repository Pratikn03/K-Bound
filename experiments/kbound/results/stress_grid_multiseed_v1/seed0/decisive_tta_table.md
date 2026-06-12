# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-11 14:19, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 34% | 0.0083 | 0.1240 | **0.0013** | 0.795 | YES | 0.1 |
| cifar10c | eata | 27% | 0.0036 | 0.1313 | **0.0013** | 0.802 | YES | 0.0 |
| cifar10c | sar | 12% | 0.0004 | 0.1404 | **0.0013** | 0.811 | no | - |
