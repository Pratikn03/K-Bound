## Multi-seed paired-CI fold-in: camelyon17

```
source_file : /Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/camelyon17_fullscale_B_v2/MULTISEED_ANALYSIS_RESULTS.json
git_commit  : 980068f78b523b7f832987ea936d0fc0ad037737
dataset     : camelyon17 (Camelyon17-WILDS (full-scale B))
generated   : 2026-06-25 16:08:51 by scripts/foldin_multiseed_results.py
```
KGA gate regret minus trivial-policy regret (lower = gate better). CI is the 95% paired bootstrap interval; *p* is Holm-adjusted; Survives = Holm-significant AND gate strictly lower.

| Method | vs | mean diff (KGA − trivial) | 95% CI | Holm *p* | Survives |
|---|---|---|---|---|---|
| tent | always-adapt | +0.0199 | [+0.0070, +0.0331] | 7.20e-03 | no |
| tent | always-freeze | -0.0524 | [-0.0710, -0.0350] | 6.00e-04 | yes |
| eata | always-adapt | +0.0043 | [-0.0018, +0.0111] | 3.69e-01 | no |
| eata | always-freeze | -0.0827 | [-0.1077, -0.0582] | 6.00e-04 | yes |
| sar | always-adapt | -0.0010 | [-0.0026, +0.0007] | 3.69e-01 | no |
| sar | always-freeze | -0.0995 | [-0.1264, -0.0724] | 6.00e-04 | yes |

**Survives Holm (gate strictly better):** tent vs always-freeze, eata vs always-freeze, sar vs always-freeze.
