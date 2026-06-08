"""Formal theorem stack registry and helpers."""

from .theorem_registry import (
    THEOREM_REGISTRY,
    TheoremSpec,
    artifact_status,
    list_theorems,
)

__all__ = [
    "THEOREM_REGISTRY",
    "TheoremSpec",
    "artifact_status",
    "list_theorems",
]
