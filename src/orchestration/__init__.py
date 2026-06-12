"""Production-grade Prefect orchestration for UAIS-V.

Each ``*_flow.py`` module exposes a Prefect flow (``@flow``) composed of retried
tasks (``@task``) that wraps an existing experiment/pipeline without changing its
computation. When Prefect is not installed, the :mod:`._compat` shim degrades the
decorators to no-ops so every flow remains importable and runnable.

Public surface
--------------
* The six domain flows: ``fraud_pipeline``, ``cyber_pipeline``,
  ``behavior_pipeline``, ``fusion_pipeline``, ``nlp_pipeline``,
  ``vision_pipeline``.
* :data:`FLOWS` — a name -> flow-callable registry (lazy; see
  :mod:`.runner`).
* :data:`PREFECT` — ``True`` when the real Prefect package is active.

Flows are imported lazily via :func:`__getattr__` so that pulling in one domain
(e.g. vision, which requires TensorFlow) never breaks unrelated orchestration
imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._compat import PREFECT
from .runner import FLOWS

_PIPELINE_EXPORTS = {
    "fraud_pipeline": ("fraud_flow", "fraud_pipeline"),
    "cyber_pipeline": ("cyber_flow", "cyber_pipeline"),
    "behavior_pipeline": ("behavior_flow", "behavior_pipeline"),
    "fusion_pipeline": ("fusion_flow", "fusion_pipeline"),
    "nlp_pipeline": ("nlp_flow", "nlp_pipeline"),
    "vision_pipeline": ("vision_flow", "vision_pipeline"),
}

__all__ = [
    "fraud_pipeline",
    "cyber_pipeline",
    "behavior_pipeline",
    "fusion_pipeline",
    "nlp_pipeline",
    "vision_pipeline",
    "FLOWS",
    "PREFECT",
]


def __getattr__(name: str) -> Any:
    """Lazily import and cache flow callables on first attribute access.

    Parameters
    ----------
    name:
        Attribute name being accessed on the package.

    Returns
    -------
    Any
        The resolved flow callable.

    Raises
    ------
    AttributeError
        If ``name`` is not a known lazy export.
    """

    if name not in _PIPELINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _PIPELINE_EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
