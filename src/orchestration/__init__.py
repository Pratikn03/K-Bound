"""Lightweight orchestration stubs for UAIS-V.

These wrappers call the existing experiment scripts so that references to
`src.orchestration.*` do not fail. They are intentionally thin to avoid heavy
dependencies; pipeline functions are imported lazily so optional domains such
as vision do not break unrelated orchestration imports.
"""

from __future__ import annotations

from importlib import import_module

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
]


def __getattr__(name: str):
    if name not in _PIPELINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _PIPELINE_EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
