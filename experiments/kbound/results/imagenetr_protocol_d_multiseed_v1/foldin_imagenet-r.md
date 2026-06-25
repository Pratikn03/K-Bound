## Multi-seed paired-CI fold-in: imagenet-r

```
source_file : /Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json
git_commit  : 980068f78b523b7f832987ea936d0fc0ad037737
dataset     : imagenet-r (ImageNet-R (Protocol D, multi-seed))
generated   : 2026-06-25 16:10:01 by scripts/foldin_multiseed_results.py
```
KGA gate regret minus trivial-policy regret (lower = gate better). CI is the 95% paired bootstrap interval; *p* is Holm-adjusted; Survives = Holm-significant AND gate strictly lower.

| Method | vs | mean diff (KGA − trivial) | 95% CI | Holm *p* | Survives |
|---|---|---|---|---|---|
| convnext_base | always-adapt | +0.0000 | [+0.0000, +0.0000] | 1.00e+00 | no |
| convnext_base | always-freeze | -0.0683 | [-0.0733, -0.0628] | 2.00e-03 | yes |
| convnext_tiny | always-adapt | +0.0194 | [+0.0131, +0.0253] | 2.00e-03 | no |
| convnext_tiny | always-freeze | -0.0018 | [-0.0043, +0.0000] | 2.68e-01 | no |
| efficientnet_b0 | always-adapt | -0.0501 | [-0.0569, -0.0438] | 2.00e-03 | yes |
| efficientnet_b0 | always-freeze | +0.0000 | [+0.0000, +0.0000] | 1.00e+00 | no |
| efficientnet_b3 | always-adapt | +0.0104 | [+0.0042, +0.0169] | 6.30e-03 | no |
| efficientnet_b3 | always-freeze | -0.0419 | [-0.0498, -0.0347] | 2.00e-03 | yes |
| resnet101 | always-adapt | +0.0149 | [+0.0098, +0.0202] | 2.00e-03 | no |
| resnet101 | always-freeze | -0.0100 | [-0.0153, -0.0052] | 3.20e-03 | yes |
| resnet152 | always-adapt | +0.0101 | [+0.0058, +0.0142] | 2.00e-03 | no |
| resnet152 | always-freeze | -0.0313 | [-0.0369, -0.0260] | 2.00e-03 | yes |
| resnext101_32x8d | always-adapt | +0.0051 | [+0.0000, +0.0106] | 1.92e-01 | no |
| resnext101_32x8d | always-freeze | -0.0460 | [-0.0505, -0.0415] | 2.00e-03 | yes |
| swin_b | always-adapt | +0.0172 | [+0.0089, +0.0258] | 2.00e-03 | no |
| swin_b | always-freeze | -0.0198 | [-0.0257, -0.0145] | 2.00e-03 | yes |
| swin_t | always-adapt | -0.0074 | [-0.0108, -0.0028] | 1.08e-02 | yes |
| swin_t | always-freeze | -0.0006 | [-0.0017, +0.0000] | 1.00e+00 | no |
| vit_b_16 | always-adapt | +0.0137 | [+0.0082, +0.0196] | 2.00e-03 | no |
| vit_b_16 | always-freeze | -0.0103 | [-0.0156, -0.0052] | 2.00e-03 | yes |

**Survives Holm (gate strictly better):** convnext_base vs always-freeze, efficientnet_b0 vs always-adapt, efficientnet_b3 vs always-freeze, resnet101 vs always-freeze, resnet152 vs always-freeze, resnext101_32x8d vs always-freeze, swin_b vs always-freeze, swin_t vs always-adapt, vit_b_16 vs always-freeze.
