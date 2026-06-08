"""kbound.optimizer -- KBoundOptimizer: gradient-gating PyTorch optimizer.

TORCH-GUARDED: this module imports torch lazily inside a try/except block.
If torch is not installed, importing this module is safe -- only instantiating
KBoundOptimizer will raise ImportError with a clear message.

KBoundOptimizer wraps any base torch.optim.Optimizer and gates/scales the
gradient update by the EProcess decision across steps:

    - ADAPT   -> full step (scale=1.0)
    - ABSTAIN -> reduced step (scale=abstain_scale, default 0.0)
    - FREEZE  -> skip step (scale=0.0)

At each ``step()`` call:
  1. Calls ``evidence_fn()`` to get a benefit sample (float).
  2. Updates the internal EProcess.
  3. Queries the EProcess decision.
  4. Scales each parameter's gradient by the appropriate factor.
  5. Calls the base optimizer's step.

Usage
-----
    from kbound import KBoundOptimizer

    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    def evidence_fn():
        # return a float benefit estimate (e.g. entropy drop)
        return float(entropy_drop.detach().cpu())

    optimizer = KBoundOptimizer(base_opt, evidence_fn=evidence_fn)

    # In training loop:
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""

from __future__ import annotations

from typing import Callable, Optional

try:
    import torch
    import torch.optim as _torch_optim
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _torch_optim = None  # type: ignore[assignment]
    _HAS_TORCH = False

from kbound.eprocess import EProcess


if _HAS_TORCH:
    class KBoundOptimizer(_torch_optim.Optimizer):
        """Gradient-gating optimizer based on the K-Bound anytime certificate.

        Wraps a base torch optimizer and gates gradient updates by the
        EProcess adapt/freeze/abstain decision.

        Parameters
        ----------
        base_optimizer : torch.optim.Optimizer
            The wrapped optimizer (e.g. Adam, SGD).
        evidence_fn : Callable[[], float]
            Called each step to produce one benefit sample X_t in [a, b].
            Typically returns the entropy drop or a benefit proxy derived
            from the current batch (must be computed BEFORE calling step).
        alpha : float, default=0.1
            EProcess type-I error budget.
        a, b : float
            Benefit range bounds.  Default [-1.0, 1.0].
        abstain_scale : float, default=0.0
            Gradient scale factor when decision is ABSTAIN (0 = skip update,
            1.0 = full update regardless of uncertainty).
        cert_callback : Callable[[str, float], None], optional
            Optional callback called with (decision, wealth) each step.

        Examples
        --------
        (Requires torch)

            import torch, torch.nn as nn
            from kbound import KBoundOptimizer

            model = nn.Linear(10, 2)
            base = torch.optim.Adam(model.parameters(), lr=1e-3)
            opt = KBoundOptimizer(base, evidence_fn=lambda: 0.1)
            x = torch.randn(8, 10)
            loss = model(x).sum()
            loss.backward()
            opt.step()
        """

        def __init__(
            self,
            base_optimizer: "torch.optim.Optimizer",
            evidence_fn: Callable[[], float] = lambda: 0.0,
            alpha: float = 0.1,
            a: float = -1.0,
            b: float = 1.0,
            abstain_scale: float = 0.0,
            cert_callback: Optional[Callable[[str, float], None]] = None,
        ) -> None:
            if not isinstance(base_optimizer, _torch_optim.Optimizer):
                raise TypeError(
                    f"base_optimizer must be a torch.optim.Optimizer, got {type(base_optimizer)}"
                )
            self.base_optimizer = base_optimizer
            self.evidence_fn = evidence_fn
            self.abstain_scale = abstain_scale
            self.cert_callback = cert_callback

            self._eprocess = EProcess(alpha=alpha, a=a, b=b)

            # torch.optim.Optimizer requires param_groups
            super().__init__(
                base_optimizer.param_groups,
                defaults={"lr": 1.0},
            )

        @property
        def eprocess(self) -> EProcess:
            """The underlying anytime-valid e-process."""
            return self._eprocess

        def step(self, closure=None):  # type: ignore[override]
            """Gated optimizer step.

            Calls evidence_fn(), updates EProcess, scales gradients by the
            certificate decision, then calls the base optimizer step.

            Parameters
            ----------
            closure : callable, optional
                Standard torch optimizer closure (re-evaluates loss).

            Returns
            -------
            loss : optional torch.Tensor
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            # 1. Evidence sample -> EProcess update
            x_t = float(self.evidence_fn())
            self._eprocess.update(x_t)

            # 2. Decision
            d = self._eprocess.decision()
            if d == "adapt":
                scale = 1.0
            elif d == "freeze":
                scale = 0.0
            else:  # abstain
                scale = self.abstain_scale

            # 3. Scale gradients
            if scale != 1.0:
                with torch.no_grad():
                    for group in self.base_optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is not None:
                                p.grad.mul_(scale)

            # 4. Base optimizer step (skipped only if scale==0 AND no grad)
            if scale > 0.0 or d == "freeze":
                # Always call base step so state (momentum buffers etc.) stays
                # consistent; if scale==0, the zero grad is a no-op for the params.
                self.base_optimizer.step()

            if self.cert_callback is not None:
                self.cert_callback(d, self._eprocess.wealth)

            return loss

        def zero_grad(self, set_to_none: bool = True):
            """Delegate to the base optimizer's zero_grad."""
            self.base_optimizer.zero_grad(set_to_none=set_to_none)

else:
    class KBoundOptimizer:  # type: ignore[no-redef]
        """Stub KBoundOptimizer -- torch is not installed.

        Instantiating this class will raise ImportError with installation
        instructions.  This stub allows the module and package to be imported
        safely on machines without torch.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "KBoundOptimizer requires PyTorch, which is not installed. "
                "Install it with:  pip install kbound[torch]\n"
                "Or directly:      pip install torch"
            )
