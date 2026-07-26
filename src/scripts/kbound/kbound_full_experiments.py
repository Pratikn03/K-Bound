"""Thin wrapper — canonical: docs/research/kbound/scripts/kbound_full_experiments.py

Defect D10 (fix-queue items 15 and 25).  This file used to be a byte-identical
copy of the canonical script, forked onto the installed package path, carrying
its own interpolated ``np.quantile(resid, 1 - alpha)`` certificate radius.  Two
copies meant two rules.  It now delegates, exactly as
``src/scripts/kbound/cifar_tent_mps_v2.py`` already did.
"""
from __future__ import annotations

from src.scripts.kbound._canonical import run_canonical

if __name__ == "__main__":
    run_canonical("kbound_full_experiments.py")
