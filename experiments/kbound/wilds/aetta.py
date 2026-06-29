"""
aetta.py - label-free ACCURACY estimators used as a stronger harm detector than entropy.

Motivation: iWildCam's K-Bound null is a DETECTABILITY failure - adaptation harm is
real but invisible to ENTROPY (confident-wrong collapse under per-location label shift).
We give the certificate a label-free accuracy/risk estimator instead of (in addition to)
the entropy evidence, then ask whether it flags the harm entropy missed.

Estimators (all label-free; NO target labels):
  (a) AETTA-style MC-dropout accuracy estimate (Lee et al., CVPR 2024 core idea):
      inject dropout on the penultimate (pooled) feature, do M stochastic head passes,
      estimate accuracy by the per-sample dropout-vote PEAKEDNESS (debiased), averaged.
      A brittle/collapsed decision scatters under dropout -> lower estimated accuracy.
  (b) frozen-reference disagreement: 1 - mean[ adapted_pred != frozen_pred ] - a cheap
      proxy that treats the (decent) source model f0 as a pseudo-reference.

Harm signal for adaptation = estAcc(adapted) - estAcc(frozen).  < 0 => predicted harmful.
INTEGRITY: these are computed identically for frozen and adapted; the detector and its
threshold are calibrated on SOURCE only (see analyzer).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _features_resnet(model, x):
    """ResNet-50 forward up to the flattened penultimate (2048-d) feature."""
    m = model
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    x = m.avgpool(x)
    return torch.flatten(x, 1)


@torch.no_grad()
def aetta_estacc(model, x, M=8, p=0.4, bn_train=False, bs=64, seed=0, return_per=False):
    """AETTA-style MC-dropout accuracy estimate.

    feat computed once (BN in eval, or train=batch-stats for adapted models); then M
    dropout passes through the classifier head. Per-sample peakedness = vote share of the
    most-voted dropout class; debiased by chance (1/C) and clipped; mean = estAcc.
    """
    model.train() if bn_train else model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    peaks = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        feat = _features_resnet(model, xb)                      # [b, D]
        C = model.fc.out_features
        votes = torch.zeros(feat.shape[0], C, device=feat.device)
        for m_ in range(M):
            mask = (torch.rand(feat.shape, generator=g).to(feat.device) > p).float() / (1.0 - p)
            logit_m = model.fc(feat * mask)
            pred_m = logit_m.argmax(1)
            votes[torch.arange(feat.shape[0]), pred_m] += 1.0
        peak = (votes.max(1).values / M)                        # vote share of top class
        peaks.append(peak.cpu().numpy())
    peak = np.concatenate(peaks)
    C = model.fc.out_features
    deb = np.clip((peak - 1.0 / C) / (1.0 - 1.0 / C), 0.0, 1.0)  # debias by chance
    est = float(deb.mean())
    return (est, deb) if return_per else est


def disagree_ref_estacc(adapted_preds, frozen_preds):
    """Option (b): frozen-reference agreement as a crude label-free accuracy proxy."""
    a = np.asarray(adapted_preds, int); f = np.asarray(frozen_preds, int)
    return float((a == f).mean())
