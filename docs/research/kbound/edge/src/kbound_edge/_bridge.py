"""kbound_edge._bridge -- Single import point for the REUSED K-Bound certificate.

The whole point of this module is to *reuse, not fork*, the published K-Bound
certificate logic.  Everything certificate-related that ``kbound_edge`` needs is
imported here from the existing ``kbound`` package (the paper reproduction
package in ``docs/research/kbound/kbound_pkg``).  No certificate math is copied
or re-implemented in the edge layer; if a guarantee needs to change, it changes
in ``kbound`` (or the canonical top-level ``kga`` package), never here.

Reused, verbatim:
    kbound.certificate.decide            -- ADAPT/FREEZE/ABSTAIN decision rule
    kbound.certificate.conformal_radius  -- population (1-alpha)-quantile radius
    kbound.evidence.evidence_vector      -- 11 label-free disagreement features
    kbound.evidence.EVIDENCE_NAMES       -- their fixed schema
    kbound.router.BenefitRouter          -- LOO-GBR + split-conformal gate

Import robustness
-----------------
On the host (``~/.venv_wilds``) ``kbound`` is already installed, so ``import
kbound`` just works.  When running from a checkout where it is not installed, we
locate the sibling ``kbound_pkg`` directory by walking up from this file and add
it to ``sys.path``.  We never modify anything inside ``kbound_pkg``.
"""

from __future__ import annotations

import os
import sys


def _ensure_kbound_importable() -> None:
    """Make the ``kbound`` package importable without modifying it.

    Strategy: try a plain import first; only if that fails do we search upward
    from this file for a ``kbound_pkg`` directory (the vendored reproduction
    package) and prepend it to ``sys.path``.
    """
    try:
        import kbound  # noqa: F401
        return
    except Exception:
        pass

    here = os.path.abspath(os.path.dirname(__file__))
    # Walk up a bounded number of levels looking for a sibling kbound_pkg.
    cur = here
    for _ in range(8):
        candidate = os.path.join(cur, "kbound_pkg")
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "kbound")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Last resort: explicit relative location (edge/src/kbound_edge -> kbound/kbound_pkg)
    guess = os.path.normpath(os.path.join(here, "..", "..", "..", "kbound_pkg"))
    if os.path.isdir(guess) and guess not in sys.path:
        sys.path.insert(0, guess)


_ensure_kbound_importable()

# --- the reused certificate surface (imported, never re-implemented) ----------
from kbound.certificate import decide, conformal_radius  # noqa: E402,F401
from kbound.evidence import evidence_vector, EVIDENCE_NAMES  # noqa: E402,F401
from kbound.router import BenefitRouter  # noqa: E402,F401

#: The paper's fixed 11-feature evidence schema (re-exported, read-only).
PAPER_EVIDENCE_NAMES = tuple(EVIDENCE_NAMES)

__all__ = [
    "decide",
    "conformal_radius",
    "evidence_vector",
    "EVIDENCE_NAMES",
    "PAPER_EVIDENCE_NAMES",
    "BenefitRouter",
]
