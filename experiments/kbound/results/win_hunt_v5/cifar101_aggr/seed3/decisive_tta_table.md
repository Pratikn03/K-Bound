# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-05 23:31, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 100% | 0.0530 | 0.0000 | **0.0000** | 0.769 | no | - |
| cifar101 | eata | 83% | 0.0072 | 0.0002 | **0.0002** | 0.770 | no | - |
| cifar101 | sar | 100% | 0.3227 | 0.0000 | **0.0000** | 0.769 | no | - |
