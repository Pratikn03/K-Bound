# Decisive deep-TTA results (KGA vs trivial policies)

_generated 2026-07-02 07:23, alpha=0.1, device=mps_

Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.

| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| cifar101 | tent | 62% | 0.0206 | 0.0023 | **0.0023** | 0.774 | no | - |
| cifar101 | eata | 46% | 0.0050 | 0.0043 | **0.0041** | 0.776 | YES | 0.1 |
| cifar101 | sar | 92% | 0.0888 | 0.0002 | **0.0002** | 0.772 | no | - |
