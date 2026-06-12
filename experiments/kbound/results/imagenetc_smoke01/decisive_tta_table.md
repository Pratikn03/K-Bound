# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-08 19:46, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| imagenetc | tent | 72% | 0.1181 | 0.0347 | **0.0000** | 0.521 | YES | 0.1 |
| imagenetc | eata | 67% | 0.1250 | 0.0417 | **0.0000** | 0.528 | YES | 0.1 |
| imagenetc | sar | 67% | 0.1250 | 0.0417 | **0.0000** | 0.528 | YES | 0.1 |
