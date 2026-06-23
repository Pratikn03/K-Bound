# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-06-16 18:44, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 67% | 0.0190 | 0.0032 | **0.0032** | 0.778 | no | - |
| cifar101 | eata | 54% | 0.0067 | 0.0048 | **0.0048** | 0.779 | no | - |
| cifar101 | sar | 8% | 0.0001 | 0.0089 | **0.0038** | 0.784 | no | - |
