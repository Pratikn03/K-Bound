"""K-Bound: the Knowability Boundary of label-free adaptation.

This package is a clean wrapper/index over the real implementations that currently
live in the monorepo (src/scripts/kbound/, src/uais/, and vendored_from_elara/).
Submodules document where each piece lives so the paper code is auditable without
moving load-bearing modules. See src/kbound/README.md for the full map.
"""
__all__ = ["evidence", "estimators", "decision", "metrics", "theory", "data", "utils"]
