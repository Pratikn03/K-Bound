# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-11 18:56, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar10c | tent | 34% | 0.0082 | 0.1237 | **0.0016** | 0.795 | YES | 0.1 |
| cifar10c | eata | 24% | 0.0034 | 0.1308 | **0.0013** | 0.802 | YES | 0.0 |
| cifar10c | sar | 10% | 0.0003 | 0.1401 | **0.0013** | 0.811 | no | - |
