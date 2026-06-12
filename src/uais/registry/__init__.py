"""Model registry, model cards, and artifact integrity for UAIS.

This subpackage provides:

* :class:`ModelCard` -- honest, structured metadata for a single model artifact.
* :class:`ModelRegistry` -- a manifest-backed registry that resolves artifacts
  and verifies their SHA-256 digests before use.
* :class:`IntegrityError` -- raised when an artifact fails verification.

The companion :mod:`uais.registry.build_manifest` module is a runnable generator
that walks ``models/`` and writes ``models/MANIFEST.json`` with real hashes.
"""

from __future__ import annotations

from .model_registry import (
    IntegrityError,
    ManifestError,
    ModelCard,
    ModelRegistry,
    sha256_file,
)

__all__ = [
    "IntegrityError",
    "ManifestError",
    "ModelCard",
    "ModelRegistry",
    "sha256_file",
]
