# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-16 18:56, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 67% | 0.0197 | 0.0015 | **0.0015** | 0.774 | no | - |
| cifar101 | eata | 58% | 0.0044 | 0.0016 | **0.0016** | 0.774 | no | - |
| cifar101 | sar | 4% | 0.0002 | 0.0077 | **0.0048** | 0.780 | no | - |
