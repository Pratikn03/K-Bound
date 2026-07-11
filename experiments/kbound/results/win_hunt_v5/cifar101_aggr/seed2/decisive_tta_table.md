# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-05 23:29, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 100% | 0.0633 | 0.0000 | **0.0000** | 0.773 | no | - |
| cifar101 | eata | 100% | 0.0190 | 0.0000 | **0.0000** | 0.773 | no | - |
| cifar101 | sar | 100% | 0.3240 | 0.0000 | **0.0000** | 0.773 | no | - |
