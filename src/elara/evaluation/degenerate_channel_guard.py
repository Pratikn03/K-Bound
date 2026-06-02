"""Degenerate-channel guard for reliability-gated multimodal fusion.

Motivation (2026-06-02 D18 held-out audit, lego_propeller artifact)
--------------------------------------------------------------------
The D18 Real-IAD-D3 held-out run produced a pooled CW-vs-gated delta of +0.045
that was *inflated by a single degenerate point-cloud channel*. Two failure
modes were diagnosed, both visible from VALIDATION data alone (no test labels):

  1. **Sign inversion** - the channel ranks anomalies *below* normals
     (e.g. lego_propeller xyz: validation AUROC = 0.000, pos-neg separation
     -0.224). The sharpness-weighted confidence-weighted mean (CW) trusts this
     "confidently wrong" channel and collapses to ~0.0175 AUROC.
  2. **Saturation** - the channel pins every sample to one end of [0, 1]
     (e.g. lego_propeller rgb: validation std = 0.000, ~100% of scores > 0.95).
     Its AUROC is then a tie-break artifact, not real discrimination, yet CW
     still treats the near-constant value as a maximally sharp vote.

A robust fusion layer must *not* let either mode drive the fused score. This
guard returns per-channel usable-reliability weights that are forced to zero for
degenerate channels, using validation labels only. The conservative policy is to
**drop** a degenerate channel (zero weight), not to flip it: a stable inversion
is flagged for investigation but flipping on a small validation set risks
overfitting and is a stronger claim than the guard should make on its own.

This module is detector- and dataset-agnostic: it consumes per-sample modality
scores in [0, 1] and validation labels, so it composes with any upstream expert
(deep PatchCore, point-cloud geometry, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

# Validation-only degeneracy thresholds. Defaults validated against the D18
# Real-IAD-D3 cache: they drop lego_propeller rgb (saturated) and xyz (inverted)
# while keeping every genuinely informative channel (connector xyz AUROC 0.943
# std 0.101; limit_switch all channels; the trivially-easy headphone category).
DEFAULT_MIN_AUC = 0.55          # below this a channel is uninformative/inverted
DEFAULT_INVERSION_AUC = 0.45    # below this the channel is actively anti-informative
DEFAULT_MIN_STD = 0.02          # below this the channel is saturated/near-constant
DEFAULT_MAX_FRAC_EXTREME = 0.97  # fraction of scores within 0.05 of {0, 1}


@dataclass(frozen=True)
class ChannelDiagnostic:
    """Validation-only diagnostic for one modality channel."""

    index: int
    val_auc: float
    separation: float        # mean(score | y=1) - mean(score | y=0)
    std: float               # dispersion of validation scores
    frac_extreme: float      # fraction of scores within 0.05 of 0 or 1
    degenerate: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def channel_diagnostic(
    scores: np.ndarray,
    labels: np.ndarray,
    index: int = 0,
    *,
    min_auc: float = DEFAULT_MIN_AUC,
    inversion_auc: float = DEFAULT_INVERSION_AUC,
    min_std: float = DEFAULT_MIN_STD,
    max_frac_extreme: float = DEFAULT_MAX_FRAC_EXTREME,
) -> ChannelDiagnostic:
    """Diagnose a single channel from its validation scores + labels.

    A channel is flagged ``degenerate`` if it is inverted/uninformative
    (AUROC < ``min_auc``) or saturated (std < ``min_std`` or > ``max_frac_extreme``
    of scores piled at the [0, 1] boundaries).
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same length")
    if len(np.unique(y)) < 2:
        # Cannot assess discrimination without both classes -> treat as degenerate.
        return ChannelDiagnostic(index, float("nan"), 0.0, float(s.std()),
                                 float(np.mean((s < 0.05) | (s > 0.95))),
                                 True, ("single_class_validation",))

    auc = float(roc_auc_score(y, s))
    sep = float(s[y == 1].mean() - s[y == 0].mean())
    std = float(s.std())
    frac_extreme = float(np.mean((s < 0.05) | (s > 0.95)))

    reasons: list[str] = []
    if auc < inversion_auc:
        reasons.append("inverted")
    elif auc < min_auc:
        reasons.append("uninformative")
    if std < min_std:
        reasons.append("saturated_low_variance")
    if frac_extreme > max_frac_extreme:
        reasons.append("saturated_boundary_pile")

    return ChannelDiagnostic(
        index=index, val_auc=auc, separation=sep, std=std,
        frac_extreme=frac_extreme, degenerate=bool(reasons), reasons=tuple(reasons),
    )


def diagnose_channels(
    s_val: np.ndarray, y_val: np.ndarray, **kw,
) -> list[ChannelDiagnostic]:
    """Diagnose every channel (column) of a validation score matrix ``[N, M]``."""
    s_val = np.asarray(s_val, dtype=float)
    if s_val.ndim != 2:
        raise ValueError("s_val must be a 2-D array [n_samples, n_channels]")
    return [channel_diagnostic(s_val[:, j], y_val, index=j, **kw)
            for j in range(s_val.shape[1])]


def guarded_channel_mask(s_val: np.ndarray, y_val: np.ndarray, **kw) -> np.ndarray:
    """Boolean mask, ``True`` for channels safe to fuse (non-degenerate)."""
    diags = diagnose_channels(s_val, y_val, **kw)
    mask = np.array([not d.degenerate for d in diags], dtype=bool)
    if not mask.any():
        # Never zero out everything: fall back to the single best-AUROC channel
        # so fusion still has something to rank with.
        aucs = np.array([(d.val_auc if np.isfinite(d.val_auc) else 0.0) for d in diags])
        mask[int(np.argmax(aucs))] = True
    return mask


def guarded_reliability(s_val: np.ndarray, y_val: np.ndarray, **kw) -> np.ndarray:
    """Per-channel reliability weights in ``[0, 1]`` with degenerate channels zeroed.

    Reliability of a kept channel is ``clip(val_auc - 0.5, 0, 0.5) / 0.5`` so a
    perfect channel gets 1.0 and a chance channel gets 0.0; dropped channels get
    exactly 0.0. Weights are returned unnormalised (callers normalise as needed).
    """
    diags = diagnose_channels(s_val, y_val, **kw)
    mask = guarded_channel_mask(s_val, y_val, **kw)
    rel = np.zeros(len(diags), dtype=float)
    for j, d in enumerate(diags):
        if mask[j] and np.isfinite(d.val_auc):
            rel[j] = float(np.clip(d.val_auc - 0.5, 0.0, 0.5) / 0.5)
    return rel
