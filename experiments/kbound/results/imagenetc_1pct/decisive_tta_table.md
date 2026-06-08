# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-08 09:02, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 44% | 0.0217 | 0.0128 | **0.0128** | 0.375 | no | - |
| imagenetc | eata | 42% | 0.0106 | 0.0111 | **0.0028** | 0.373 | YES | 0.2 |
| imagenetc | sar | 44% | 0.0372 | 0.0117 | **0.0050** | 0.374 | YES | 0.1 |
