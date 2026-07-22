"""kbound_repro.deps -- deferred dependency imports with actionable errors.

Replaces the anti-pattern of ``try: import torch  except Exception: torch = None``
which silently degrades required scientific dependencies to ``None`` and turns a
missing-package problem into a confusing ``AttributeError`` (or, worse, wrong
results) much later.

Rules encoded here:

* **Required** scientific dependencies raise a clear ``ImportError`` *at the
  point of use* naming the package and how to install it.
* **Optional** dependencies may be absent; the failure is deferred until the
  feature is actually called, and the error still names the missing package.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["MissingDependency", "require", "optional"]

# Canonical install hints for the packages we care about.
_INSTALL_HINTS = {
    "torch": "pip install torch  (see https://pytorch.org for the right build)",
    "torchvision": "pip install torchvision",
    "numpy": "pip install numpy",
    "scipy": "pip install scipy",
    "sklearn": "pip install scikit-learn",
    "pandas": "pip install pandas",
    "jsonschema": "pip install jsonschema",
    "matplotlib": "pip install matplotlib",
    "yaml": "pip install pyyaml",
}


class MissingDependency(ImportError):
    """Raised when a required dependency is not importable."""


def _hint(package: str) -> str:
    return _INSTALL_HINTS.get(package, f"pip install {package}")


def require(package: str, *, feature: str | None = None) -> ModuleType:
    """Import and return ``package`` or raise a clear ``MissingDependency``.

    Use for dependencies that a code path genuinely needs (e.g. ``torch`` inside
    a model runner).  Call it *inside* the function that needs the package so
    unrelated imports/tests are never blocked by an absent optional stack.
    """
    try:
        return importlib.import_module(package)
    except ImportError as exc:  # narrow: only import failures, never bare except
        ctx = f" required for {feature}" if feature else ""
        raise MissingDependency(
            f"'{package}'{ctx} is not installed. Install it with: {_hint(package)}"
        ) from exc


def optional(package: str):
    """Import ``package`` if available, else return ``None``.

    Only for genuinely optional features.  Prefer :func:`require` inside the
    feature so the eventual error still names the package; use ``optional`` only
    for capability probes (e.g. "is matplotlib available for figure export?").
    """
    try:
        return importlib.import_module(package)
    except ImportError:
        return None
