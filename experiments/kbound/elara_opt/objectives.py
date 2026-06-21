"""objectives.py — the unlabeled objectives ELARA-Opt mixes.

Every loss here is a function of *logits only* (never labels), so the optimizer
provably cannot leak target labels through its objective.  The three mixture
objectives are {entropy, reliability-filtered entropy, augmentation-consistency};
the frozen-model KL anchor is a separate stability term.  Unit-tested in
tests/elara_opt/test_objectives.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

#: the three reliability-gated mixture objectives, in canonical order
OBJECTIVE_NAMES = ["entropy", "filtered_entropy", "aug_consistency"]


def _probs_logprobs(logits: torch.Tensor):
    logp = F.log_softmax(logits, dim=1)
    return logp.exp(), logp


def entropy_per_sample(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of softmax(logits), per sample. Label-free."""
    p, logp = _probs_logprobs(logits)
    return -(p * logp).sum(1)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean predictive entropy (the Tent objective). Label-free."""
    return entropy_per_sample(logits).mean()


def reliable_mask(logits: torch.Tensor, num_classes: int, margin_frac: float = 0.4) -> torch.Tensor:
    """Boolean mask of confident (low-entropy) samples: H < margin_frac*ln(C).
    The same reliable-sample criterion EATA/SAR use. Label-free."""
    thr = margin_frac * math.log(num_classes)
    return entropy_per_sample(logits) < thr


def filtered_entropy_loss(logits: torch.Tensor, num_classes: int, margin_frac: float = 0.4):
    """Mean entropy over reliable samples only. Returns (loss, kept_fraction).
    If nothing passes, returns a graph-connected zero so autograd stays defined."""
    e = entropy_per_sample(logits)
    thr = margin_frac * math.log(num_classes)
    keep = e < thr
    if int(keep.sum()) == 0:
        return logits.sum() * 0.0, 0.0
    return e[keep].mean(), float(keep.float().mean())


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Per-sample symmetric KL between two softmax distributions. Label-free."""
    pa, lpa = _probs_logprobs(logits_a)
    pb, lpb = _probs_logprobs(logits_b)
    kl_ab = (pa * (lpa - lpb)).sum(1)
    kl_ba = (pb * (lpb - lpa)).sum(1)
    return 0.5 * (kl_ab + kl_ba)


def aug_consistency_loss(logits: torch.Tensor, logits_aug: torch.Tensor) -> torch.Tensor:
    """Augmentation-consistency: mean symmetric KL between predictions on the
    original and an augmented (label-preserving) view. Label-free."""
    return symmetric_kl(logits, logits_aug).mean()


def frozen_kl_anchor(logits_frozen: torch.Tensor, logits_current: torch.Tensor) -> torch.Tensor:
    """KL(p0 || p_t): penalize the current model drifting from the frozen model's
    predictive distribution — the trust anchor in function space. Label-free."""
    p0, lp0 = _probs_logprobs(logits_frozen)
    _, lpt = _probs_logprobs(logits_current)
    return (p0 * (lp0 - lpt)).sum(1).mean()


def augment(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """A deterministic, label-preserving augmentation that keeps sample order.
    Images (NCHW) -> horizontal flip; otherwise -> small seeded Gaussian jitter.
    Sample-aligned so consistency is computed per original sample. Label-free."""
    if x.dim() == 4:
        return torch.flip(x, dims=[3])
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=g).to(x.device, x.dtype) * 0.05
    return x + noise


def all_mixture_losses(logits, logits_aug, num_classes, margin_frac=0.4):
    """Compute the three mixture objective losses from precomputed logits.
    Returns (dict name->scalar tensor, kept_fraction). Label-free."""
    l_ent = entropy_loss(logits)
    l_filt, kept = filtered_entropy_loss(logits, num_classes, margin_frac)
    l_aug = aug_consistency_loss(logits, logits_aug)
    return {"entropy": l_ent, "filtered_entropy": l_filt, "aug_consistency": l_aug}, kept
