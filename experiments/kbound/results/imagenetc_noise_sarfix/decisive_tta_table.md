# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-10 15:35, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 6% | 0.0007 | 0.0494 | **0.0082** | 0.455 | no | - |
| imagenetc | eata | 3% | 0.0000 | 0.0386 | **0.0038** | 0.444 | no | - |
| imagenetc | sar | 44% | 0.1124 | 0.0267 | **0.0229** | 0.432 | YES | 0.2 |
