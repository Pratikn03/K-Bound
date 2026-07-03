# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-01 23:58, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 50% | 0.0187 | 0.0396 | **0.0042** | 0.373 | YES | 0.2 |
| imagenetc | eata | 42% | 0.0187 | 0.0375 | **0.0042** | 0.371 | YES | 0.2 |
| imagenetc | sar | 46% | 0.1042 | 0.0385 | **0.0135** | 0.372 | YES | 0.1 |
