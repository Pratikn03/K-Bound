"""Package entry point so ``python -m uais.training`` runs the CLI.

Delegates to :func:`uais.training.cli.main`.
"""

from __future__ import annotations

from uais.training.cli import main

if __name__ == "__main__":  # pragma: no cover - module entry guard
    raise SystemExit(main())
