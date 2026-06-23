"""kbound_edge.tent_adapter -- EpisodicTentAdapter (safe, isolated candidate).

TENT (Wang et al. 2021) adapts a model at test time by minimising the entropy of
its predictions, updating ONLY the BatchNorm affine parameters.  This module
wraps that as an *episodic* adapter with one hard safety guarantee:

    THE FROZEN MODEL f0 IS NEVER MUTATED.

Every call to :meth:`EpisodicTentAdapter.adapt` deep-copies f0 into a fresh
candidate, configures BN-affine-only training on the COPY, takes ``steps`` Adam
steps (default 1) of entropy minimisation, and returns the adapted candidate
plus the L2 norm of the BN-affine update.  f0's parameters and buffers are
guaranteed bit-identical before and after (checked by
:mod:`tests.test_candidate_isolation`).

"Episodic" = each window starts again from f0; adaptation never accumulates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass


def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kbound_edge.tent_adapter requires PyTorch. "
            "On the host:  source ~/.venv_wilds/bin/activate"
        ) from exc


def softmax_entropy(logits):
    """Mean-free per-sample Shannon entropy of softmax(logits).  Returns (N,)."""
    torch = _torch()
    logp = torch.log_softmax(logits, dim=1)
    p = torch.softmax(logits, dim=1)
    return -(p * logp).sum(dim=1)


def configure_tent_(model):
    """Configure a model for TENT IN PLACE (call only on a COPY, never on f0).

    Freezes all parameters, then enables grad on BatchNorm affine weight/bias and
    switches BN to use *batch* statistics (track_running_stats=False).  Returns
    the list of trainable BN-affine parameters.
    """
    torch = _torch()
    import torch.nn as nn

    model.train()
    model.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            # Use batch statistics during adaptation (canonical TENT).
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


@dataclass
class AdaptResult:
    """Outcome of one episodic adaptation step."""

    model: object        # the adapted candidate (torch.nn.Module)
    upd_norm: float      # L2 norm of the BN-affine parameter update
    loss_before: float   # entropy before the step
    loss_after: float    # entropy after the step
    n_params: int        # number of BN-affine scalars updated


class EpisodicTentAdapter:
    """Episodic TENT: produce an isolated adapted candidate from a frozen f0.

    Parameters
    ----------
    f0 : torch.nn.Module
        The frozen reference model.  NEVER mutated.
    lr : float, default=1e-3
        Adam learning rate for the BN-affine parameters.
    steps : int, default=1
        Number of entropy-minimisation steps per window (1 = single step).
    device : str, default="cpu"

    Notes
    -----
    The adapter holds a reference to f0 but only ever deep-copies it; it does not
    call ``f0.train()`` or touch f0's grad state.
    """

    def __init__(self, f0, lr: float = 1e-3, steps: int = 1, device: str = "cpu") -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.f0 = f0
        self.lr = lr
        self.steps = steps
        self.device = device

    def adapt(self, x) -> AdaptResult:
        """Adapt a fresh candidate to batch ``x`` and return it (f0 untouched).

        Parameters
        ----------
        x : torch.Tensor of shape (N, 3, H, W)
            A window of preprocessed frames (N >= 2 recommended so BN batch stats
            are well-defined).

        Returns
        -------
        AdaptResult
        """
        torch = _torch()

        # 1. Deep copy -> the candidate is fully independent of f0.
        candidate = copy.deepcopy(self.f0).to(self.device)

        # 2. Configure BN-affine-only training on the COPY.
        params = configure_tent_(candidate)
        if not params:
            raise RuntimeError("No BatchNorm affine parameters found to adapt")

        before = [p.detach().clone() for p in params]
        n_params = int(sum(p.numel() for p in params))

        opt = torch.optim.Adam(params, lr=self.lr)

        x = x.to(self.device)
        with torch.no_grad():
            loss_before = float(softmax_entropy(candidate(x)).mean().cpu())
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            logits = candidate(x)
            loss = softmax_entropy(logits).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            loss_after = float(softmax_entropy(candidate(x)).mean().cpu())

        # 3. Update norm = L2 over the concatenated BN-affine deltas.
        with torch.no_grad():
            sq = 0.0
            for p, b in zip(params, before):
                sq += float(((p.detach() - b) ** 2).sum().cpu())
            upd_norm = float(sq ** 0.5)

        candidate.eval()
        return AdaptResult(
            model=candidate,
            upd_norm=upd_norm,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            n_params=n_params,
        )
