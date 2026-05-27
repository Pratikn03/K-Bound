"""Runtime-only inference façade for the ELARA fusion stack.

The implementation deliberately re-uses the model and estimator
classes from the research codebase rather than copy-pasting them. The
research source lives at ``src/uais/fusion/attention``; this module
imports from there to avoid forking and keep the inference surface
in lockstep with what we train. When this package is shipped as a
standalone wheel, the ``uais.fusion.attention.cross_modal_attention``
and ``uais.fusion.attention.reliability_estimator`` modules become
the package's only mandatory transitive dependency.

The class below provides:

  - InferRGA.from_checkpoint(...): load a trained fusion model and a
    fitted reliability estimator from disk.
  - InferRGA.predict_proba(features, masks): batched anomaly-score
    prediction using static-attention fusion (the deployment-grade
    path; the reliability-gated path is opt-in via use_gate=True).
  - InferRGA.reliability(features, masks): per-domain reliability
    weights, returned as a (batch, num_domains) numpy array.

This is the same inference path validated by tests/test_infer_rga.py
and is intentionally narrow: no training, no benchmarking, no
adversarial-sweep machinery. Production users who want only inference
import this package and nothing else from the research codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.reliability_estimator import (
    PerSampleReliabilityEstimator,
    ReliabilityEstimator,
)


@dataclass
class RuntimeMetadata:
    """Provenance attached to an InferRGA instance for audit trails."""

    model_path: str
    estimator_path: str | None
    num_domains: int
    input_dim: int
    estimator_kind: str
    device: str


class InferRGA:
    """Runtime-only RGA inference wrapper.

    Two prediction modes are exposed:
      - static (default): pure attention fusion (no gate).
      - gated: a hybrid of the static path and a reliability-weighted
        recomputation. The gate fires when mean per-sample reliability
        falls below ``tau`` (default ``0.66`` matches the paper).

    The two modes share the same trained model; only the post-hoc
    reliability gate differs.
    """

    def __init__(
        self,
        model: AttentionFusionModel,
        estimator: ReliabilityEstimator | None,
        *,
        device: torch.device | str = "cpu",
        tau: float = 0.66,
        metadata: RuntimeMetadata | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.estimator = estimator
        self.tau = float(tau)
        self.metadata = metadata

    # ----- construction --------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        estimator_path: str | Path | None = None,
        *,
        device: torch.device | str = "cpu",
        tau: float = 0.66,
    ) -> InferRGA:
        """Load a fusion model + fitted reliability estimator from disk."""
        model_path = Path(model_path)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model_args = ckpt.get("model_args") or {}
        # Sensible defaults if the checkpoint is bare:
        model_args.setdefault("num_domains", ckpt.get("num_domains", 2))
        model_args.setdefault("input_dim", ckpt.get("input_dim", 5))
        model = AttentionFusionModel(**model_args)
        model.load_state_dict(ckpt["state_dict"])

        estimator: ReliabilityEstimator | None = None
        estimator_kind = "none"
        if estimator_path is not None:
            estimator = _load_estimator(Path(estimator_path))
            estimator_kind = type(estimator).__name__

        metadata = RuntimeMetadata(
            model_path=str(model_path),
            estimator_path=str(estimator_path) if estimator_path else None,
            num_domains=int(model_args["num_domains"]),
            input_dim=int(model_args["input_dim"]),
            estimator_kind=estimator_kind,
            device=str(device),
        )
        return cls(model=model, estimator=estimator, device=device, tau=tau, metadata=metadata)

    # ----- inference -----------------------------------------------------

    @torch.inference_mode()
    def predict_proba(
        self,
        features: np.ndarray | torch.Tensor,
        masks: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Return fused anomaly probabilities of shape (batch,)."""
        feat_t, mask_t = self._to_tensors(features, masks)
        logits, _, _ = self.model(feat_t, key_padding_mask=mask_t)
        probs = torch.sigmoid(logits.squeeze(-1))
        return probs.detach().cpu().numpy().astype(np.float32)

    def reliability(
        self,
        features: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Return per-domain reliability weights of shape (batch, num_domains).

        Raises if no estimator was supplied at construction.
        """
        if self.estimator is None:
            raise RuntimeError(
                "InferRGA was constructed without a reliability estimator; "
                "cannot compute reliability. Pass estimator_path to from_checkpoint."
            )
        feats = np.asarray(features, dtype=np.float32)
        msks = np.asarray(masks, dtype=bool)
        return self.estimator.compute_reliability_weights(feats, msks)

    def predict_with_gate(
        self,
        features: np.ndarray,
        masks: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return both static-path probs and a gate-firing indicator.

        The deployment policy is to use ``static_probs`` always; the
        ``gate_fired`` array is logged separately for monitoring. This
        matches the paper's observe-only gate stance.
        """
        static_probs = self.predict_proba(features, masks)
        if self.estimator is None:
            gate_fired = np.zeros_like(static_probs, dtype=bool)
            mean_rel = np.ones_like(static_probs)
        else:
            rel = self.reliability(features, masks)
            # Average over present (unmasked) domains per sample.
            present = (~np.asarray(masks, dtype=bool)).astype(np.float32)
            denom = np.clip(present.sum(axis=1), 1.0, None)
            mean_rel = (rel * present).sum(axis=1) / denom
            gate_fired = mean_rel < self.tau
        return {
            "static_probs": static_probs,
            "mean_reliability": mean_rel.astype(np.float32),
            "gate_fired": gate_fired.astype(bool),
        }

    # ----- helpers -------------------------------------------------------

    def _to_tensors(
        self,
        features: np.ndarray | torch.Tensor,
        masks: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(features, np.ndarray):
            feat_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            feat_t = features.to(self.device, dtype=torch.float32)
        if isinstance(masks, np.ndarray):
            mask_t = torch.tensor(masks, dtype=torch.bool, device=self.device)
        else:
            mask_t = masks.to(self.device, dtype=torch.bool)
        return feat_t, mask_t


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_estimator(path: Path) -> ReliabilityEstimator:
    """Load a fitted ReliabilityEstimator (or PerSampleReliabilityEstimator)."""
    # The estimator's own load() handles both subclasses via the metadata
    # field stored at fit time.
    return ReliabilityEstimator.load(path)


__all__ = ["InferRGA", "RuntimeMetadata", "PerSampleReliabilityEstimator"]
