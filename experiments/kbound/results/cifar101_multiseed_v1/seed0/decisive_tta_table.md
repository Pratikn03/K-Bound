# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-16 18:31, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 67% | 0.0148 | 0.0023 | **0.0023** | 0.774 | no | - |
| cifar101 | eata | 42% | 0.0050 | 0.0044 | **0.0044** | 0.776 | no | - |
| cifar101 | sar | 0% | 0.0000 | 0.0097 | **0.0020** | 0.782 | no | - |
