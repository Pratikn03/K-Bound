# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-16 18:52, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 71% | 0.0183 | 0.0020 | **0.0020** | 0.775 | no | - |
| cifar101 | eata | 50% | 0.0056 | 0.0033 | **0.0033** | 0.776 | no | - |
| cifar101 | sar | 8% | 0.0002 | 0.0082 | **0.0060** | 0.781 | no | - |
