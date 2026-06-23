# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-16 18:48, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 67% | 0.0181 | 0.0031 | **0.0031** | 0.776 | no | - |
| cifar101 | eata | 46% | 0.0065 | 0.0042 | **0.0036** | 0.777 | YES | 0.1 |
| cifar101 | sar | 4% | 0.0001 | 0.0088 | **0.0058** | 0.781 | no | - |
