"""kbound_repro -- reproducibility & artifact-authority toolkit for K-Bound.

Small, dependency-light modules that make the K-Bound evidence chain
reproducible from a clean checkout:

    metrics       canonical (torch-independent) decision-metric library
    paths         repository-root discovery + external data-root resolution
    runtime       device/runtime selection (requested -> CUDA -> MPS -> CPU)
    deps          deferred hard/optional dependency imports with clear errors
    schema        versioned result-schema validators + historical migration
    authority     claim-ledger / manifest / claim-matrix authority chain
    storage       storage-policy manifest + staged-file guardrails

Submodules are intentionally NOT imported here so that importing one module
(e.g. ``kbound_repro.metrics``) never drags in another module's optional
dependency (e.g. ``jsonschema``).
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
