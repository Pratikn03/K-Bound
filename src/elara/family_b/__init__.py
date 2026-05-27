"""Phase 2.2B — Family-B (mechanism) experiment infrastructure.

Built in response to the Phase 2.2B infrastructure audit which found
that the existing repo could only run the k=4 coherent-collapse path.
This package adds:

- `corruption` — re-exports the k-of-D corruption primitive and adds a
  validation-fold corruption injector usable for gate-threshold
  selection without test-fold visibility.
- `mixture_shift` — a pure mixture-shift sampler that varies category
  proportions while holding within-category score distributions fixed.
- `ks_window` — helpers to build KS-window-size sweep grids that
  parameterize the existing ``ReliabilityEstimator`` `ks_window_size`.
"""

from elara.family_b.corruption import (  # noqa: F401
    KOfDCorruptionResult,
    inject_corruption,
    validation_fold_corruption_grid,
)
from elara.family_b.ks_window import KS_WINDOW_GRID  # noqa: F401
from elara.family_b.mixture_shift import (  # noqa: F401
    MixtureShiftResample,
    pure_mixture_shift_resample,
)
