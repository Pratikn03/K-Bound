"""Phase 2.2B — KS-window-size sweep grid (B-MECH-4).

The locked contract sweeps window sizes {32, 64, 128, 256, 512}. The
``ReliabilityEstimator.ks_window_size`` parameter (added in Phase 2.2B
alongside the G3 top-q gate) consumes a single integer; this module
provides the locked grid for the sweep harness.
"""

from __future__ import annotations

KS_WINDOW_GRID: tuple[int, ...] = (32, 64, 128, 256, 512)

__all__ = ["KS_WINDOW_GRID"]
